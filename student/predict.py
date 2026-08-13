"""Generate a submission.csv from a checkpoint over the test splits.

By default this covers **both** ``test_public`` and ``test_private`` — the
server uses ``test_public`` for the live leaderboard and ``test_private``
for final scoring, so a single combined submission works for both. Override
with ``--splits test_public`` if you only want leaderboard predictions.

Submission format (matches what the master evaluator expects):

- Columns: ``uid, p_0, p_1, ..., p_{K-1}``
- One row per uid across the requested splits
- Probabilities sum to ~1 per row
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from student.eval import apply_calibration, collect_logits, load_checkpoint

DEFAULT_SPLITS: tuple[str, ...] = ("test_public", "test_private")


def collect_test_predictions(
    model, loader: DataLoader, device, temperature: float = 1.0,
    calibration: dict | None = None, tta_hflip: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Run model on the loader; return (uids in batch order, probs as np array)."""
    logits, uids = collect_logits(model, loader, device, tta_hflip=tta_hflip)
    return [str(u) for u in uids], apply_calibration(logits, calibration, temperature)


def write_submission(uids: list[str], probs: np.ndarray, output_path: Path) -> None:
    K = probs.shape[1]
    cols: dict[str, list] = {"uid": list(uids)}
    for k in range(K):
        cols[f"p_{k}"] = probs[:, k]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(output_path, index=False)


def predict(
    checkpoint: Path,
    data_root: Path,
    output: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    tta_hflip: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T, cfg = load_checkpoint(checkpoint, device)
    calibration = cfg["calibration"]
    # student.calibrate records whether flip-TTA won the cross-validated
    # comparison; respect that unless the caller forces it on.
    tta_hflip = tta_hflip or bool((calibration or {}).get("tta_hflip", False))

    all_uids: list[str] = []
    all_probs: list[np.ndarray] = []
    for split in splits:
        # The whole input pipeline comes from the checkpoint, not from
        # defaults: resolution, box/banner cropping and normalization all
        # have to match what the model was trained on.
        ds = cfg.dataset(data_root, split)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        uids, probs = collect_test_predictions(
            model, loader, device, temperature=T,
            calibration=calibration, tta_hflip=tta_hflip,
        )
        all_uids.extend(uids)
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    write_submission(all_uids, probs, output)
    method = (calibration or {}).get("method", f"temperature (T={T:.4f})")
    print(f"wrote {output} ({len(all_uids)} rows, {probs.shape[1]} classes, "
          f"calibration={method}, tta_hflip={tta_hflip})")


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
                        help="Test splits to predict on (default: test_public test_private).")
    parser.add_argument("--tta-hflip", action="store_true",
                        help="Average predictions over the image and its mirror.")
    args = parser.parse_args()
    predict(
        checkpoint=args.checkpoint, data_root=args.data_root, output=args.output,
        batch_size=args.batch_size, num_workers=args.num_workers,
        splits=tuple(args.splits), tta_hflip=args.tta_hflip,
    )


if __name__ == "__main__":
    main()
