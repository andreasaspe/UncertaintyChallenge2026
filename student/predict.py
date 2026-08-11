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

from student.data import IWildCamChallengeDataset, default_eval_transform
from student.eval import load_checkpoint

DEFAULT_SPLITS: tuple[str, ...] = ("test_public", "test_private")


def collect_test_predictions(
    model, loader: DataLoader, device, temperature: float = 1.0
) -> tuple[list[str], np.ndarray]:
    """Run model on the loader; return (uids in batch order, probs as np array)."""
    model.eval()
    uids: list[str] = []
    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, batch_uids in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits / temperature, dim=1)
            probs_chunks.append(probs.cpu().numpy())
            uids.extend(batch_uids)
    return uids, np.concatenate(probs_chunks, axis=0)


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
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T, img_size = load_checkpoint(checkpoint, device)

    all_uids: list[str] = []
    all_probs: list[np.ndarray] = []
    for split in splits:
        ds = IWildCamChallengeDataset(data_root, split, default_eval_transform(img_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        uids, probs = collect_test_predictions(model, loader, device, temperature=T)
        all_uids.extend(uids)
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    write_submission(all_uids, probs, output)
    print(f"wrote {output} ({len(all_uids)} rows, {probs.shape[1]} classes, T={T:.4f})")


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
    args = parser.parse_args()
    predict(
        checkpoint=args.checkpoint, data_root=args.data_root, output=args.output,
        batch_size=args.batch_size, num_workers=args.num_workers,
        splits=tuple(args.splits),
    )


if __name__ == "__main__":
    main()
