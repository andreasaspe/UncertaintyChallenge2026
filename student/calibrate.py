"""Post-hoc calibration: pick the best way to turn logits into probabilities.

Four of the five scored metrics (ece, nll, brier, misclassification_auroc)
depend on *how confident* the model claims to be, not just on which class it
picks. A neural net trained with cross-entropy is systematically
overconfident, so a cheap transform applied after training — fit on val,
touching no weights — reliably improves them. This module fits several such
transforms and picks between them honestly.

    python -m student.calibrate --checkpoint runs/dinov3_7b/model.pt \\
        --data-root <data-root>

The methods, in plain terms:

``temperature`` (NLL)
    Divide every logit by a single number T. T>1 flattens the distribution,
    T<1 sharpens it. Cannot change the argmax, so accuracy is untouched. This
    is the incumbent — it already cuts ECE from .163 to .039 on this project's
    baseline.

``temperature`` (Brier)
    The same one-parameter family, fit against Brier instead of NLL. NLL is
    dominated by the handful of confidently-wrong examples (it pays -log p),
    while Brier is bounded, so the two objectives land on different T. Worth a
    look when Brier is a scored column.

``temp_mix``
    Temperature, then blend in a little of the uniform distribution:
    ``p = (1-eps)*softmax(z/T) + eps/K``. Two parameters. The blend puts a
    floor under every class probability, which is exactly the failure NLL
    punishes hardest — one confidently-wrong prediction at p_true = 1e-6 costs
    13.8 nats on its own.

``vector``
    Per-class affine rescaling, ``z' = a*z + b``, so classes the model is
    systematically over- or under-confident about get individual treatment.
    2K = 114 parameters against 918 val images. Unlike the scalar methods it
    can also *change predictions*, so it is the one option here that can cost
    accuracy. It is included precisely because it should overfit, and the
    cross-validation should show that.

Why cross-validation matters here: val is only 918 images, and it is the same
set we early-stopped on. Fitting a calibrator on all of val and then reporting
its val metrics measures how well it memorised val, not how it will score on
the test cameras. So every candidate is scored **out-of-fold**: fit on 4/5 of
val, predict the held-out 5th, pool all five held-out chunks, then compute the
metrics once on the pooled predictions.

Test-time augmentation (``--tta``) is evaluated as a second axis, so each
calibrator is scored both with and without it. TTA is the only option here
that can move accuracy and misclassification AUROC rather than merely
reshaping confidences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from student import metrics as M
from student.eval import (TTA_POLICIES, apply_calibration, collect_logits_tta,
                          load_checkpoint, tta_views)
from student.train import HIGHER_IS_BETTER_METRICS, METRIC_NAMES


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def _t(x: np.ndarray, device="cpu") -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def fit_temperature_nll(logits: np.ndarray, labels: np.ndarray) -> dict:
    """One scalar T minimising NLL, via LBFGS on log T (keeps T > 0)."""
    z, y = _t(logits), torch.as_tensor(labels, dtype=torch.long)
    log_T = nn.Parameter(torch.zeros(1))
    opt = optim.LBFGS([log_T], lr=0.1, max_iter=200)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(z / log_T.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_T.detach().exp())
    return {"T": T if 0 < T < 1e3 else 1.0}


def fit_temperature_brier(logits: np.ndarray, labels: np.ndarray) -> dict:
    """One scalar T minimising Brier.

    Brier is not convex in log T the way NLL is, and it is cheap to evaluate,
    so a coarse-to-fine 1-D scan is more robust here than a gradient method.
    """
    onehot = np.zeros_like(logits, dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0

    def brier_at(log_T: float) -> float:
        p = apply_calibration(logits, {"method": "temperature",
                                       "params": {"T": float(np.exp(log_T))}})
        return float(((p - onehot) ** 2).sum(axis=1).mean())

    lo, hi = -2.0, 3.0
    for _ in range(4):  # coarse-to-fine: 4 passes of 41 points -> ~1e-4 in log T
        grid = np.linspace(lo, hi, 41)
        vals = [brier_at(g) for g in grid]
        i = int(np.argmin(vals))
        step = grid[1] - grid[0]
        lo, hi = grid[i] - step, grid[i] + step
    return {"T": float(np.exp((lo + hi) / 2))}


def fit_temp_mix(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Fit (T, eps) for ``p = (1-eps)*softmax(z/T) + eps/K`` against NLL.

    ``eps`` is parameterised as ``0.2 * sigmoid(raw)`` so it stays in
    [0, 0.2) — a uniform floor above that would start costing real accuracy
    in the metrics that reward confident-and-correct.
    """
    z, y = _t(logits), torch.as_tensor(labels, dtype=torch.long)
    K = logits.shape[1]
    log_T = nn.Parameter(torch.zeros(1))
    raw_eps = nn.Parameter(torch.full((1,), -4.0))  # eps ~ 0.0036 at init
    opt = optim.LBFGS([log_T, raw_eps], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        eps = 0.2 * torch.sigmoid(raw_eps)
        p = torch.softmax(z / log_T.exp(), dim=1) * (1 - eps) + eps / K
        loss = -torch.log(p[torch.arange(len(y)), y].clamp_min(1e-12)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_T.detach().exp())
    eps = float(0.2 * torch.sigmoid(raw_eps.detach()))
    if not (0 < T < 1e3):
        T, eps = 1.0, 0.0
    return {"T": T, "eps": eps}


def fit_vector(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Per-class affine rescale ``z' = a*z + b``, fit against NLL.

    Initialised at the identity (a=1, b=0) and given a small weight decay, so
    with too little data it degrades toward "do nothing" rather than toward
    nonsense.
    """
    z, y = _t(logits), torch.as_tensor(labels, dtype=torch.long)
    K = logits.shape[1]
    a = nn.Parameter(torch.ones(K))
    b = nn.Parameter(torch.zeros(K))
    opt = optim.LBFGS([a, b], lr=0.1, max_iter=200)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(z * a + b, y) + 1e-3 * ((a - 1) ** 2).sum() + 1e-3 * (b ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"a": a.detach().numpy().tolist(), "b": b.detach().numpy().tolist()}


METHODS = {
    "temperature_nll": ("temperature", fit_temperature_nll),
    "temperature_brier": ("temperature", fit_temperature_brier),
    "temp_mix": ("temp_mix", fit_temp_mix),
    "vector": ("vector", fit_vector),
}


# --------------------------------------------------------------------------
# cross-validated comparison
# --------------------------------------------------------------------------

def make_folds(n: int, k: int, groups: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Assign each of ``n`` rows a fold id in ``[0, k)``.

    When ``groups`` is given (here: the id/ood domain label) the split is
    stratified on it, so no fold ends up with a wildly different id/ood mix
    than the others — with 464 ood images out of 918, an unlucky split would
    otherwise make the fold-to-fold noise larger than the effect we're trying
    to measure.
    """
    rng = np.random.default_rng(seed)
    folds = np.empty(n, dtype=int)
    strata = [np.arange(n)] if groups is None else [
        np.flatnonzero(groups == g) for g in np.unique(groups)
    ]
    for idx in strata:
        perm = rng.permutation(idx)
        folds[perm] = np.arange(len(perm)) % k
    return folds


def oof_probs(logits: np.ndarray, labels: np.ndarray, method: str,
              folds: np.ndarray) -> np.ndarray:
    """Out-of-fold calibrated probabilities: every row predicted by a fit
    that never saw it."""
    kind, fit = METHODS[method]
    probs = np.zeros_like(logits, dtype=np.float64)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        params = fit(logits[tr], labels[tr])
        probs[te] = apply_calibration(logits[te], {"method": kind, "params": params})
    return probs


def borda(rows: list[dict]) -> list[int]:
    """Borda count across the five metrics: rank 1 = best, summed. Lower wins.

    This is the leaderboard's own scoring rule, so optimising it locally is
    optimising the thing we're actually ranked on — and it stops us picking a
    calibrator that wins NLL by a mile while quietly wrecking ECE.
    """
    n = len(rows)
    total = [0] * n
    for metric in METRIC_NAMES:
        vals = np.array([r["metrics"][metric] for r in rows], dtype=float)
        vals = np.where(np.isnan(vals), -np.inf if metric in HIGHER_IS_BETTER_METRICS else np.inf, vals)
        order = np.argsort(-vals if metric in HIGHER_IS_BETTER_METRICS else vals)
        for rank, i in enumerate(order, start=1):
            total[i] += rank
    return total


def calibrate(
    checkpoint: Path,
    data_root: Path,
    output: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    folds: int = 5,
    seed: int = 0,
    tta: bool = True,
    tta_policies: tuple[str, ...] | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_T, cfg = load_checkpoint(checkpoint, device)
    val_ds = cfg.dataset(data_root, "val")
    domains = np.asarray(val_ds.domains) if val_ds.domains else None

    # Which TTA policies to weigh up. Each costs one val pass per view, so the
    # full menu is 1 + 2 + 2 + 4 = 9 passes over 918 images.
    if tta_policies is None:
        tta_policies = tuple(TTA_POLICIES) if tta else ("none",)

    variants: dict[str, np.ndarray] = {}
    labels: np.ndarray | None = None
    for policy in tta_policies:
        n_views = len(tta_views(policy, int(cfg["img_size"])))
        print(f"running val under TTA policy {policy!r} ({n_views} view(s))")
        z, tgt, _ = collect_logits_tta(
            model, cfg, data_root, "val", device, policy=policy,
            batch_size=batch_size, num_workers=num_workers,
        )
        variants[policy] = z
        if labels is None:
            labels = np.asarray(tgt, dtype=int)

    fold_ids = make_folds(len(labels), folds, groups=domains, seed=seed)

    rows: list[dict] = []
    for use_tta, z in variants.items():
        # "uncalibrated" is the honest floor: raw softmax, nothing fitted.
        rows.append({
            "method": "none", "tta": use_tta, "params": {},
            "metrics": M.compute_all_metrics(apply_calibration(z, None, 1.0), labels),
        })
        for name in METHODS:
            probs = oof_probs(z, labels, name, fold_ids)
            rows.append({
                "method": name, "tta": use_tta,
                # Report the refit-on-all-of-val parameters (what we would
                # ship), while the metrics above stay strictly out-of-fold.
                "params": METHODS[name][1](z, labels),
                "metrics": M.compute_all_metrics(probs, labels),
            })

    for row, score in zip(rows, borda(rows)):
        row["borda"] = score
    rows.sort(key=lambda r: r["borda"])

    print(f"\n{folds}-fold cross-validated calibration on val "
          f"(n={len(labels)}, out-of-fold metrics)\n")
    header = f"{'method':<20}{'tta':<14}" + "".join(f"{m[:9]:>11}" for m in METRIC_NAMES) + f"{'borda':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = "".join(f"{r['metrics'][m]:>11.4f}" if r["metrics"][m] is not None else f"{'--':>11}"
                        for m in METRIC_NAMES)
        print(f"{r['method']:<20}{str(r['tta']):<14}{cells}{r['borda']:>8}")

    best = rows[0]
    kind = METHODS[best["method"]][0] if best["method"] != "none" else "temperature"
    params = best["params"] or {"T": 1.0}
    calibration = {"method": kind, "params": params, "tta": str(best["tta"]),
                   # kept for older consumers that only knew about a flip flag
                   "tta_hflip": "hflip" in str(best["tta"]),
                   "selected_as": best["method"]}
    scalar_T = float(params.get("T", 1.0))
    print(f"\nselected: {best['method']} (tta={best['tta']}) -> {kind}")
    print(f"  params: { {k: (v if not isinstance(v, list) else f'<{len(v)} values>') for k, v in params.items()} }")

    output = Path(output) if output else Path(checkpoint).parent / "model_calibrated.pt"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["calibration"] = calibration
    # Keep the legacy scalar in sync so anything still reading `temperature`
    # directly stays consistent with the chosen calibrator.
    ckpt["temperature"] = scalar_T
    torch.save(ckpt, output)
    print(f"wrote {output}")

    report = {"checkpoint": str(checkpoint), "folds": folds, "seed": seed,
              "selected": calibration,
              "candidates": [{k: v for k, v in r.items() if k != "params"} for r in rows]}
    (output.parent / "calibration_report.json").write_text(json.dumps(report, indent=2, default=float))
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and compare post-hoc calibration methods.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Use model.pt (T=1.0); this fits the calibration itself.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <checkpoint dir>/model_calibrated.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tta", action="store_true",
                        help="Only evaluate the no-TTA variant (fastest).")
    parser.add_argument("--tta-policies", nargs="+", default=None,
                        choices=sorted(TTA_POLICIES),
                        help="Which TTA policies to compare (default: all of them).")
    args = parser.parse_args()
    calibrate(
        checkpoint=args.checkpoint, data_root=args.data_root, output=args.output,
        batch_size=args.batch_size, num_workers=args.num_workers,
        folds=args.folds, seed=args.seed, tta=not args.no_tta,
        tta_policies=tuple(args.tta_policies) if args.tta_policies else None,
    )


if __name__ == "__main__":
    main()
