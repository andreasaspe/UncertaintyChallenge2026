"""OOD scoring in the classifier's own feature space, plus gate diagnostics.

Motivation
----------
"OOD" in this challenge means *new camera location* (covariate shift), not
new species — every class in val-ood is present in train. So we don't need
an open-set detector; we need a measure of "how far is this image from the
training distribution", computed in a space that knows about species.

The fine-tuned backbone's ``embed()`` output is exactly that space, so we
score against it directly rather than training a separate autoencoder /
ViT. Two standard scores are implemented:

- ``knn_score``         — k-th nearest-neighbour distance over L2-normalised
                          features (Sun et al. 2022, "Deep Nearest Neighbors").
- ``mahalanobis_score`` — distance to the nearest class-conditional Gaussian
                          under a shared (tied) covariance (Lee et al. 2018).

Both are *unsupervised w.r.t. domain*: they only ever look at train (which is
100% `id`). Val's `domain` labels are used to VALIDATE the score and to fit
at most two calibration parameters — never to train the detector. Training a
supervised id-vs-ood classifier on val's 464 ood images would score ~99% by
memorising those specific camera backgrounds and then transfer poorly to
test's different held-out locations.

Why an input-dependent temperature is the lever
-----------------------------------------------
``metrics.misclassification_auroc`` hard-codes the confidence score as
max-softmax, so we cannot simply "submit a better uncertainty score". The
only way to move that metric is to change the probabilities themselves.
A per-image temperature ``T(x) = exp(a * s(x) + b)`` does exactly that: it
re-ranks max-softmax confidences globally. Note ``a = 0`` recovers ordinary
temperature scaling, so this can only help if the score carries signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from student import metrics as M
from student.data import IWildCamChallengeDataset, default_eval_transform
from student.eval import load_checkpoint


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device):
    """Return ``(features, logits, targets)`` for a loader.

    Computes ``head(embed(x))`` rather than ``forward(x)`` so we get the
    penultimate features and the logits from a single pass.
    """
    model.eval()
    feats, logits, targets = [], [], []
    for imgs, tgt in tqdm(loader, desc="extract", leave=False):
        f = model.embed(imgs.to(device))
        feats.append(f.cpu().numpy())
        logits.append(model.head(f).cpu().numpy())
        targets.append(np.asarray(tgt))
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(targets)


def features_for_split(model, data_root, split, device, batch_size, num_workers, eval_cfg):
    """Features for one split, always under the *eval* transform (no augmentation)."""
    ds = eval_cfg.dataset(data_root, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, logits, targets = extract_features(model, loader, device)
    return feats, logits, targets, ds


# --------------------------------------------------------------------------
# OOD scores (higher = more out-of-distribution)
# --------------------------------------------------------------------------

def knn_score(train_feats: np.ndarray, query_feats: np.ndarray, k: int = 50) -> np.ndarray:
    """Distance to the k-th nearest training neighbour, on L2-normalised features.

    Normalising first makes this a cosine distance, which is what makes the
    method robust to the wildly varying feature norms a fine-tuned net produces.
    """
    tr = train_feats / np.clip(np.linalg.norm(train_feats, axis=1, keepdims=True), 1e-12, None)
    q = query_feats / np.clip(np.linalg.norm(query_feats, axis=1, keepdims=True), 1e-12, None)
    out = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), 512):  # chunked so the sim matrix stays small
        sim = q[i:i + 512] @ tr.T
        out[i:i + 512] = 1.0 - np.partition(sim, -k, axis=1)[:, -k]
    return out


def mahalanobis_score(
    train_feats: np.ndarray, train_labels: np.ndarray, query_feats: np.ndarray,
    num_classes: int, shrinkage: float = 1e-2,
) -> np.ndarray:
    """Min over classes of the Mahalanobis distance under a tied covariance.

    A per-class covariance would be rank-deficient (768 dims, ~330 images per
    class), so we pool residuals into one shared covariance and add ridge
    shrinkage. We whiten via a Cholesky factor and then use plain Euclidean
    distance, which is far cheaper than forming the quadratic form directly.
    """
    means = np.stack([train_feats[train_labels == c].mean(axis=0) for c in range(num_classes)])
    centered = train_feats - means[train_labels]
    cov = (centered.T @ centered) / len(train_feats)
    cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    L = np.linalg.cholesky(np.linalg.inv(cov))

    qw, mw = query_feats @ L, means @ L
    d2 = (
        (qw ** 2).sum(1)[:, None] - 2.0 * (qw @ mw.T) + (mw ** 2).sum(1)[None, :]
    )
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def auroc(scores: np.ndarray, target: np.ndarray) -> float:
    """AUROC of `scores` at predicting binary `target` (Mann-Whitney U)."""
    target = np.asarray(target).astype(int)
    n_pos = int(target.sum())
    n_neg = int(len(target) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[target == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_scalar_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Ordinary temperature scaling, returned as ``(a=0, b=log T)``."""
    lg = torch.as_tensor(logits, dtype=torch.float32)
    lb = torch.as_tensor(labels, dtype=torch.long)
    b = nn.Parameter(torch.zeros(1))
    opt = optim.LBFGS([b], lr=0.1, max_iter=200)
    crit = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = crit(lg / torch.exp(b), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return 0.0, float(b.detach())


def fit_linear_temperature(
    logits: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Fit ``T(x) = exp(a * s(x) + b)`` by minimising NLL. ``a=0`` => plain scaling."""
    lg = torch.as_tensor(logits, dtype=torch.float32)
    lb = torch.as_tensor(labels, dtype=torch.long)
    s = torch.as_tensor(scores, dtype=torch.float32).unsqueeze(1)
    a = nn.Parameter(torch.zeros(1))
    b = nn.Parameter(torch.zeros(1))
    opt = optim.LBFGS([a, b], lr=0.1, max_iter=200)
    crit = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = crit(lg / torch.exp(a * s + b), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.detach()), float(b.detach())


def apply_temperature(logits: np.ndarray, scores: np.ndarray, a: float, b: float) -> np.ndarray:
    T = np.exp(a * scores + b)[:, None]
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)          # softmax, numerically stable
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Gate diagnostics
# --------------------------------------------------------------------------

def _fmt(d: dict) -> str:
    return "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()
    )


def run_gate(checkpoint, data_root, cache, batch_size=128, num_workers=8, folds=5, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T_ckpt, eval_cfg = load_checkpoint(checkpoint, device)
    # eval_cfg rebuilds the exact input pipeline this checkpoint was trained
    # under (resolution, box/banner crop, normalization) -- see eval.EvalConfig.

    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        tr_f, tr_y = z["tr_f"], z["tr_y"]
        va_f, va_logits, va_y, va_dom = z["va_f"], z["va_logits"], z["va_y"], z["va_dom"]
        print(f"loaded cached features from {cache}")
    else:
        tr_f, _, tr_y, _ = features_for_split(model, data_root, "train", device, batch_size, num_workers, eval_cfg)
        va_f, va_logits, va_y, va_ds = features_for_split(model, data_root, "val", device, batch_size, num_workers, eval_cfg)
        va_dom = np.asarray(va_ds.domains)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, tr_f=tr_f, tr_y=tr_y, va_f=va_f,
                            va_logits=va_logits, va_y=va_y, va_dom=va_dom)
        print(f"cached features to {cache}")

    num_classes = int(tr_y.max()) + 1
    is_ood = (va_dom == "ood").astype(int)
    print(f"\ntrain={len(tr_f)}  val={len(va_f)} (id={int((1-is_ood).sum())}, ood={int(is_ood.sum())})"
          f"  feat_dim={tr_f.shape[1]}  K={num_classes}")

    # Reference point: max-softmax under the checkpoint's own temperature.
    base_probs = apply_temperature(va_logits, np.zeros(len(va_f)), 0.0, float(np.log(T_ckpt)))
    msp = base_probs.max(axis=1)
    correct = (base_probs.argmax(axis=1) == va_y).astype(int)

    scores = {"mahalanobis": mahalanobis_score(tr_f, tr_y, va_f, num_classes)}
    for k in (10, 50, 200):
        scores[f"knn_k{k}"] = knn_score(tr_f, va_f, k=k)

    # --- Diagnostic A: can the score tell id from ood at all? ---------------
    print("\n=== A. domain detection: AUROC(score ; ood vs id) ===")
    print("    0.5 = no signal, 1.0 = perfectly separates held-out locations")
    for name, s in scores.items():
        print(f"  {name:14s} {auroc(s, is_ood):.4f}")

    # --- Diagnostic B: does it predict ERRORS better than max-softmax? ------
    # This is the one that matters: the scored metric cares about knowing when
    # the classifier is wrong, not about knowing where the photo was taken.
    print("\n=== B. error prediction: AUROC( . ; correct vs incorrect) ===")
    print(f"  {'max-softmax':14s} {auroc(msp, correct):.4f}   <- incumbent, must be beaten")
    for name, s in scores.items():
        print(f"  {name:14s} {auroc(-s, correct):.4f}   (rank-corr with msp: "
              f"{np.corrcoef(pd.Series(s).rank(), pd.Series(msp).rank())[0,1]:+.3f})")

    # --- C: does an input-dependent temperature actually beat a scalar? -----
    # Cross-validated so both methods are judged out-of-fold; val is only 918
    # rows and doubles as the fitting set, so an in-sample comparison would
    # flatter the 2-parameter model.
    print(f"\n=== C. {folds}-fold CV on val: scalar T  vs  T(x)=exp(a*s+b) ===")
    rng = np.random.default_rng(seed)
    fold_id = rng.permutation(len(va_f)) % folds

    best = max(scores, key=lambda n: abs(auroc(scores[n], is_ood) - 0.5))
    print(f"    using score = {best}")
    s_raw = scores[best]

    oof = {"scalar": np.zeros_like(base_probs), "linear": np.zeros_like(base_probs)}
    a_hist = []
    for f in range(folds):
        tr_m, te_m = fold_id != f, fold_id == f
        mu, sd = s_raw[tr_m].mean(), s_raw[tr_m].std() + 1e-12   # standardise on train folds only
        s_std = (s_raw - mu) / sd

        a0, b0 = fit_scalar_temperature(va_logits[tr_m], va_y[tr_m])
        oof["scalar"][te_m] = apply_temperature(va_logits[te_m], s_std[te_m], a0, b0)

        a1, b1 = fit_linear_temperature(va_logits[tr_m], va_y[tr_m], s_std[tr_m])
        oof["linear"][te_m] = apply_temperature(va_logits[te_m], s_std[te_m], a1, b1)
        a_hist.append(a1)

    print(f"    fitted a per fold: {', '.join(f'{a:+.3f}' for a in a_hist)}"
          f"   (a=0 means the score adds nothing)")
    for name, probs in oof.items():
        print(f"\n  --- {name} ---")
        for tag, m in (("overall", slice(None)), ("id", is_ood == 0), ("ood", is_ood == 1)):
            print(f"    {tag:8s} {_fmt(M.compute_all_metrics(probs[m], va_y[m]))}")


def main() -> None:
    p = argparse.ArgumentParser(description="OOD-score gate diagnostics on val.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True,
                   help="npz path to cache extracted features (re-runs are then instant)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    run_gate(args.checkpoint, args.data_root, args.cache,
             args.batch_size, args.num_workers, args.folds)


if __name__ == "__main__":
    main()
