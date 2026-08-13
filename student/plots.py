"""Training curves for the five metrics the challenge actually scores.

Reads the ``history.csv`` that ``student.train`` rewrites after every epoch
and produces ``curves.png``. Runnable standalone against a live run:

    python -m student.plots --run-dir <data-root>/runs/dinov3_7b

Each metric panel shows three lines — ``overall``, ``id`` (cameras seen in
train) and ``ood`` (held-out cameras) — because the aggregate number hides
the thing we care most about: whether the model stays calibrated when the
location changes. Solid lines are post-temperature-scaling (what a submission
would score); the faint dashed line is the raw softmax, so the gap between
them shows how much work calibration is doing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in a training container / over ssh
import matplotlib.pyplot as plt
import pandas as pd

from student.train import HIGHER_IS_BETTER_METRICS, METRIC_NAMES

PRETTY = {
    "accuracy": "accuracy  (higher better)",
    "ece": "ECE  (lower better)",
    "nll": "NLL  (lower better)",
    "brier": "Brier  (lower better)",
    "misclassification_auroc": "misclassification AUROC  (higher better)",
}
SCOPE_STYLE = {
    "overall": {"color": "#1f4e79", "lw": 2.2, "zorder": 3},
    "id": {"color": "#2e8b57", "lw": 1.4, "zorder": 2},
    "ood": {"color": "#c1440e", "lw": 1.4, "zorder": 2},
}


def plot_history(
    history_path: Path,
    output_path: Path,
    best_epoch: int | None = None,
) -> Path:
    """Render ``history.csv`` to ``curves.png``; returns the output path."""
    df = pd.read_csv(history_path)
    if df.empty:
        raise ValueError(f"{history_path} has no rows yet")
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.ravel()

    # Panel 0: the optimisation itself. A train_acc that runs away from
    # val accuracy is the overfitting signature this project keeps hitting.
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color="#444", lw=2, label="train loss")
    ax.set_ylabel("train loss")
    ax.set_xlabel("epoch")
    twin = ax.twinx()
    twin.plot(epochs, df["train_acc"], color="#888", ls=":", lw=1.6, label="train acc")
    if "overall_ts_accuracy" in df:
        twin.plot(epochs, df["overall_ts_accuracy"], color="#1f4e79", lw=1.6, label="val acc")
    twin.set_ylabel("accuracy")
    ax.set_title("optimisation")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center right")

    for ax, metric in zip(axes[1:], METRIC_NAMES):
        for scope, style in SCOPE_STYLE.items():
            ts_col, raw_col = f"{scope}_ts_{metric}", f"{scope}_raw_{metric}"
            if ts_col in df:
                ax.plot(epochs, df[ts_col], label=f"{scope} (T-scaled)", **style)
            if raw_col in df:
                ax.plot(epochs, df[raw_col], color=style["color"], lw=1.0,
                        ls="--", alpha=0.35, label=f"{scope} (raw)")
        ax.set_title(PRETTY.get(metric, metric), fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)

        # Mark the epoch whose weights were actually kept.
        if best_epoch:
            ax.axvline(best_epoch, color="#999", ls="-.", lw=1.0, zorder=1)
        # Mark this metric's own best, which may be a different epoch --
        # useful for seeing when the five metrics disagree about "best".
        col = f"overall_ts_{metric}"
        if col in df and df[col].notna().any():
            idx = df[col].idxmax() if metric in HIGHER_IS_BETTER_METRICS else df[col].idxmin()
            ax.plot(df["epoch"][idx], df[col][idx], "o", color="#1f4e79", ms=6, zorder=4)

    axes[1].legend(fontsize=7, ncol=2)
    ttl = f"{history_path.parent.name} — val metrics per epoch"
    if best_epoch:
        ttl += f"  (kept epoch {best_epoch}, dash-dot line)"
    fig.suptitle(ttl, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from history.csv.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run output directory containing history.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <run-dir>/curves.png")
    parser.add_argument("--best-epoch", type=int, default=None)
    args = parser.parse_args()
    out = plot_history(
        args.run_dir / "history.csv",
        args.output or args.run_dir / "curves.png",
        best_epoch=args.best_epoch,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
