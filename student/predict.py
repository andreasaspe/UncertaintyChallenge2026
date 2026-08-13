"""Generate a submission.csv from a checkpoint over the test splits.

By default this covers **both** ``test_public`` and ``test_private`` — the
server uses ``test_public`` for the live leaderboard and ``test_private``
for final scoring, so a single combined submission works for both. Override
with ``--splits test_public`` if you only want leaderboard predictions.

Submission format (matches what the master evaluator expects):

- Columns: ``uid, p_0, p_1, ..., p_{K-1}``
- One row per uid across the requested splits
- Probabilities sum to ~1 per row

``--with-logits`` additionally writes a **separate** ``<output>_logits.csv``
holding the raw, uncalibrated logits (``l_0 ... l_{K-1}``) for the test splits
*and* for ``val``, so a calibration method can be fitted downstream. It is a
second file on purpose: adding rows and columns to the submission itself would
make it invalid to upload. Both files come from one inference pass, so the
extra output costs only the val split (~918 images).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from student.eval import (TTA_POLICIES, apply_calibration, collect_logits,
                          collect_logits_tta, load_checkpoint, tta_views)

DEFAULT_SPLITS: tuple[str, ...] = ("test_public", "test_private")
LOGITS_EXTRA_SPLIT = "val"


def run_split(
    model, cfg, data_root, split: str, device,
    batch_size: int = 32, num_workers: int = 4,
    temperature: float = 1.0, calibration: dict | None = None,
    tta: str = "none",
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Run one split; return ``(uids, probs, logits, labels)``.

    ``logits`` are the model's outputs *before* any calibration — what a
    downstream calibrator needs. Under TTA they are the log of the averaged
    probabilities, which is still uncalibrated and is what the calibrator was
    fitted against. ``labels`` are the true classes on a labelled split (val)
    and ``-1`` on the unlabelled test splits.
    """
    ds = cfg.dataset(data_root, split)
    logits, targets, _ = collect_logits_tta(
        model, cfg, data_root, split, device, policy=tta,
        batch_size=batch_size, num_workers=num_workers,
    )
    if ds.labels is None:
        # Unlabelled splits yield the uid as the second element.
        uids = [str(u) for u in targets]
        labels = np.full(len(uids), -1, dtype=int)
    else:
        # Labelled splits yield the *label*, not the uid. Reading uids from the
        # dataset in order is only valid because the loader is shuffle=False.
        uids = [str(u) for u in ds.uids]
        labels = np.asarray(targets, dtype=int)
    return uids, apply_calibration(logits, calibration, temperature), logits, labels


def collect_test_predictions(
    model, loader: DataLoader, device, temperature: float = 1.0,
    calibration: dict | None = None, tta_hflip: bool = False,
    progress: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Run model on the loader; return (uids in batch order, probs as np array)."""
    logits, uids = collect_logits(model, loader, device, tta_hflip=tta_hflip,
                                  progress=progress)
    return [str(u) for u in uids], apply_calibration(logits, calibration, temperature)


def write_submission(uids: list[str], probs: np.ndarray, output_path: Path) -> None:
    K = probs.shape[1]
    cols: dict[str, object] = {"uid": list(uids)}
    for k in range(K):
        cols[f"p_{k}"] = probs[:, k]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(output_path, index=False)


def write_logits(
    uids: list[str], splits: list[str], labels: np.ndarray,
    logits: np.ndarray, output_path: Path,
) -> None:
    """Write ``uid, split, y, l_0..l_{K-1}`` for downstream calibration.

    ``y`` is the true label on val and ``-1`` on the test splits, so a
    calibrator can be fitted on the val rows and applied to the rest without
    a second lookup.
    """
    K = logits.shape[1]
    cols: dict[str, object] = {"uid": list(uids), "split": list(splits), "y": labels}
    for k in range(K):
        cols[f"l_{k}"] = logits[:, k]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(output_path, index=False)


def predict(
    checkpoint: Path,
    data_root: Path,
    output: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    tta: str = "none",
    with_logits: bool = False,
    logits_output: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T, cfg = load_checkpoint(checkpoint, device)
    calibration = cfg["calibration"]
    # student.calibrate records which TTA policy won its cross-validated
    # comparison. Honour that unless the caller names one explicitly, because
    # the calibrator was fitted against that policy's outputs -- pairing a
    # calibrator with a different TTA policy would mis-calibrate.
    if tta == "auto":
        tta = str((calibration or {}).get("tta", "none"))
    n_views = len(tta_views(tta, int(cfg["img_size"])))
    print(f"TTA policy: {tta} ({n_views} view(s) per image)")

    if with_logits and n_views > 1:
        print("[note] TTA is on, so l_* hold the log of the averaged probabilities "
              "rather than single-pass logits. Still uncalibrated, and the "
              "calibrator was fitted against exactly these. Use --tta none for "
              "the plain forward pass.")

    # The logits file wants val too; the submission must never contain it.
    run_splits = list(splits)
    if with_logits and LOGITS_EXTRA_SPLIT not in run_splits:
        run_splits.append(LOGITS_EXTRA_SPLIT)

    sub_uids: list[str] = []
    sub_probs: list[np.ndarray] = []
    all_uids: list[str] = []
    all_splits: list[str] = []
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    for split in run_splits:
        # The whole input pipeline comes from the checkpoint, not from
        # defaults: resolution, box/banner cropping and normalization all
        # have to match what the model was trained on.
        uids, probs, logits, labels = run_split(
            model, cfg, data_root, split, device,
            batch_size=batch_size, num_workers=num_workers,
            temperature=T, calibration=calibration, tta=tta,
        )
        if split in splits:
            sub_uids.extend(uids)
            sub_probs.append(probs)
        all_uids.extend(uids)
        all_splits.extend([split] * len(uids))
        all_labels.append(labels)
        all_logits.append(logits)

    probs = np.concatenate(sub_probs, axis=0)
    write_submission(sub_uids, probs, output)
    method = (calibration or {}).get("method", f"temperature (T={T:.4f})")
    print(f"wrote {output} ({len(sub_uids)} rows, {probs.shape[1]} classes, "
          f"calibration={method}, tta={tta})")

    if with_logits:
        path = Path(logits_output) if logits_output else \
            output.with_name(f"{output.stem}_logits{output.suffix}")
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        write_logits(all_uids, all_splits, labels, logits, path)
        per_split = ", ".join(f"{s}={all_splits.count(s)}" for s in run_splits)
        print(f"wrote {path} ({len(all_uids)} rows: {per_split}; "
              f"raw uncalibrated l_0..l_{logits.shape[1]-1} + split + y)")
        print(f"[note] {path.name} is for analysis only — it includes val rows "
              f"and is not a submission file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a submission.csv from a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write submission.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                        help="Splits for the submission (default: test_public test_private).")
    parser.add_argument("--tta", type=str, default="auto",
                        choices=sorted(TTA_POLICIES) + ["auto"],
                        help="Test-time augmentation policy. 'auto' (default) uses "
                             "whichever policy student.calibrate selected and stored "
                             "in the checkpoint, falling back to 'none'.")
    parser.add_argument("--with-logits", action="store_true",
                        help="Also write <output>_logits.csv with raw uncalibrated "
                             "l_0..l_{K-1} for the test splits AND val, plus split/y "
                             "columns. Same inference pass; submission stays valid.")
    parser.add_argument("--logits-output", type=Path, default=None,
                        help="Override the logits file path (default: <output>_logits.csv).")
    args = parser.parse_args()
    predict(
        checkpoint=args.checkpoint, data_root=args.data_root, output=args.output,
        batch_size=args.batch_size, num_workers=args.num_workers,
        splits=tuple(args.splits), tta=args.tta,
        with_logits=args.with_logits, logits_output=args.logits_output,
    )


if __name__ == "__main__":
    main()
