"""Local evaluator: load a checkpoint, run on val, report the four metrics + accuracy.

Output matches what the master evaluator computes on the server, so use this
to iterate locally before submitting.

Checkpoint format expected from ``student.train``::

    torch.save({
        "state_dict":   model.state_dict(),
        "num_classes":  K,
        "temperature":  T,            # 1.0 means no scaling
        "backbone":     backbone_name,  # timm model id, e.g. "resnet50"
    }, path)
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from student.data import IMG_SIZE, IWildCamChallengeDataset, default_eval_transform
from student.data import make_box_cropper
from student.data import IMAGENET_MEAN, IMAGENET_STD
from student.metrics import compute_all_metrics
from student.model import DEFAULT_BACKBONE, DEFAULT_LORA_TARGETS, Classifier


class EvalConfig(dict):
    """Input-pipeline settings recovered from a checkpoint.

    Anything that changes what the *pixels* look like has to travel with the
    weights, or eval/predict silently feed the model a distribution it was
    never trained on. Concretely: resolution, the top/bottom banner crop, and
    the normalization statistics.
    """

    def transform(self, img_size: int | None = None):
        # When box cropping is on it applies the banner clamp itself, so the
        # torchvision transform must not crop again.
        crop_frac = 0.0 if self["box_crop"] else self["crop_frac"]
        return default_eval_transform(
            img_size or self["img_size"], crop_frac=crop_frac,
            mean=self["mean"], std=self["std"],
        )

    def box_cropper(self, data_root):
        return make_box_cropper(
            data_root, enabled=self["box_crop"],
            conf_threshold=self["box_conf"], margin=self["box_margin"],
            min_frac=self["box_min_frac"], max_aspect=self["box_max_aspect"],
            banner_frac=self["crop_frac"] or 0.05,
            mask_banner=not self["box_no_mask_banner"],
            strict_containment=not self["box_no_strict"],
        )

    def dataset(self, data_root, split: str, img_size: int | None = None
                ) -> IWildCamChallengeDataset:
        """Build a dataset with the exact input pipeline this model was trained on.

        Single entry point so eval / predict / calibrate / ood cannot drift
        apart -- a mismatch here is silent and would only show up as an
        unexplained accuracy drop.

        ``img_size`` overrides the training resolution, which is what
        multi-scale TTA needs. It re-resizes from the *original* image rather
        than upsampling an already-downsampled tensor, so a larger scale
        actually carries more detail instead of just more pixels.
        """
        return IWildCamChallengeDataset(
            data_root, split, self.transform(img_size),
            box_cropper=self.box_cropper(data_root),
        )


def load_checkpoint(ckpt_path: Path, device) -> tuple[nn.Module, float, EvalConfig]:
    """Return ``(model, temperature, eval_config)``.

    Every architecture-shaping field is read from the checkpoint's
    ``hyperparameters`` with the *pre-LoRA* default as fallback, so
    checkpoints written before any of this existed rebuild byte-identically.

    LoRA-only checkpoints store just the adapters and the head (a few hundred
    MB rather than 27 GB for the 7B). For those the frozen base weights are
    re-fetched from timm with ``pretrained=True`` — deterministic, because
    they are the published weights and training never touched them — and the
    state dict is loaded non-strictly on top.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyperparameters", {})
    lora_only = bool(ckpt.get("lora_only", False))
    lora_r = int(hp.get("lora_r", 0))

    model = Classifier(
        int(ckpt["num_classes"]),
        backbone_name=ckpt.get("backbone", DEFAULT_BACKBONE),
        # A LoRA-only checkpoint has no base weights of its own.
        pretrained=lora_only,
        img_size=hp.get("img_size"),
        feature_pool=hp.get("feature_pool", "default"),
        head=hp.get("head", "linear"),
        head_hidden=int(hp.get("head_hidden", 1024)),
        head_dropout=float(hp.get("head_dropout", 0.0)),
        lora_r=lora_r,
        lora_alpha=float(hp.get("lora_alpha", 0.0)),
        lora_dropout=0.0,  # dropout is a no-op at eval; keep the module shapes simple
        lora_targets=tuple(hp.get("lora_targets", DEFAULT_LORA_TARGETS)),
    )

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=not lora_only)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {ckpt_path}: {sorted(unexpected)[:5]}")
    if lora_only:
        # Everything *not* stored must be a frozen base weight that timm just
        # restored. If a trainable tensor went missing, fail loudly rather than
        # silently evaluating a randomly-initialised head.
        trainable = {n for n, _ in model.trainable_parameters()}
        lost = sorted(trainable.intersection(missing))
        if lost:
            raise RuntimeError(f"LoRA checkpoint {ckpt_path} is missing trainable tensors: {lost[:5]}")

    model.to(device).eval()

    cfg = EvalConfig(
        img_size=int(hp.get("img_size", IMG_SIZE)),
        crop_frac=float(hp.get("crop_frac", 0.0)),
        mean=tuple(hp.get("mean", IMAGENET_MEAN)),
        std=tuple(hp.get("std", IMAGENET_STD)),
        box_crop=bool(hp.get("box_crop", False)),
        box_conf=float(hp.get("box_conf", 0.2)),
        box_margin=float(hp.get("box_margin", 0.15)),
        box_min_frac=float(hp.get("box_min_frac", 0.30)),
        box_max_aspect=float(hp.get("box_max_aspect", 1.6)),
        box_no_mask_banner=bool(hp.get("box_no_mask_banner", False)),
        box_no_strict=bool(hp.get("box_no_strict", False)),
        calibration=ckpt.get("calibration"),
    )
    return model, float(ckpt.get("temperature", 1.0)), cfg


def apply_calibration(logits: np.ndarray, calib: dict | None, temperature: float = 1.0) -> np.ndarray:
    """Turn raw logits into the probabilities we actually submit.

    The scalar methods never change *which* class is predicted (dividing
    every logit by the same T cannot reorder them), so accuracy is untouched
    and only ECE / NLL / Brier move. ``vector`` is the exception: it rescales
    each class separately and therefore can flip predictions — measured on
    synthetic data it moved ~24% of argmaxes, so it is a genuine risk to
    accuracy, not just a reshaping. Methods:

    - ``temperature``: divide logits by a scalar T. T>1 softens, T<1 sharpens.
    - ``temp_mix``:    temperature, then blend eps of a uniform distribution in.
      The blend puts a floor under every class probability, which is what NLL
      (``-log p_true``) punishes hardest when the model is confidently wrong.
    - ``vector``:      per-class affine rescaling of the logits, ``a * z + b``.
      2K parameters — enough to overfit a 918-image val set, which is exactly
      why student/calibrate.py cross-validates before choosing it.
    """
    if calib is None:
        calib = {"method": "temperature", "params": {"T": float(temperature)}}
    method = calib.get("method", "temperature")
    params = calib.get("params", {})
    z = np.asarray(logits, dtype=np.float64)

    if method == "vector":
        z = z * np.asarray(params["a"]) + np.asarray(params["b"])
    else:
        z = z / float(params.get("T", temperature))

    z = z - z.max(axis=1, keepdims=True)  # softmax overflow guard
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)

    if method == "temp_mix":
        eps = float(params.get("eps", 0.0))
        probs = (1.0 - eps) * probs + eps / probs.shape[1]
    return probs


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device,
    amp: bool = False,
    tta_hflip: bool = False,
    hflip_only: bool = False,
    progress: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader``; return ``(logits, targets)`` as np arrays.

    ``targets`` are int labels for train/val and uid strings for the test
    splits, whichever the dataset yields.

    ``progress`` shows a tqdm bar with that label. Off by default so library
    callers stay quiet; ``predict`` turns it on, because inference over 8,322
    test images through a 7B backbone is slow enough that silence is
    indistinguishable from a hang.

    ``amp`` defaults to **off**. bf16 carries roughly three decimal digits,
    which is plenty for a training gradient but visibly perturbs the softmax
    confidences that ECE and misclassification AUROC are computed from --
    enough to move ECE by ~.004 on this val set. Evaluation is a tiny fraction
    of total runtime, so it is not worth trading numbers for. Turn it on with
    ``--amp`` if a large model's eval pass really is the bottleneck.

    With ``tta_hflip`` the model is also run on the mirrored image and the two
    **probability** vectors are averaged (not the logits — averaging logits is
    a geometric mean, which stays overconfident). The averaged probabilities
    are returned as ``log(p)`` so that everything downstream, including
    temperature scaling, keeps working on a logit-shaped array. Test-time
    augmentation is one of the few post-hoc tricks that can improve *accuracy*
    and misclassification AUROC, not merely recalibrate.
    """
    model.eval()
    use_amp = amp and torch.device(device).type == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    all_logits, all_targets = [], []
    batches = tqdm(loader, desc=progress, unit="batch") if progress else loader
    for imgs, targets in batches:
        imgs = imgs.to(device, non_blocking=True)
        if hflip_only:
            # Run this one view mirrored. Used by collect_logits_tta, which
            # does its own averaging across views.
            imgs = torch.flip(imgs, dims=[3])
        with ctx:
            logits = model(imgs).float()
            if tta_hflip:
                flipped = model(torch.flip(imgs, dims=[3])).float()
                mean_probs = 0.5 * (torch.softmax(logits, dim=1)
                                    + torch.softmax(flipped, dim=1))
                logits = torch.log(mean_probs.clamp_min(1e-12))
        all_logits.append(logits.cpu().numpy())
        all_targets.append(np.asarray(targets))
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


# --------------------------------------------------------------------------
# test-time augmentation
# --------------------------------------------------------------------------

# Named TTA policies, cheapest first. Kept as a small fixed menu rather than
# free-form flags so that whatever `calibrate` selects can be recorded in the
# checkpoint by name and reproduced exactly by `predict`.
TTA_POLICIES: dict[str, dict] = {
    "none":        {"scales": None, "hflip": False},
    "hflip":       {"scales": None, "hflip": True},
    "scales":      {"scales": (1.0, 1.25), "hflip": False},
    "hflip+scales": {"scales": (1.0, 1.25), "hflip": True},
}


def tta_views(policy: str, base_img_size: int) -> list[tuple[int, bool]]:
    """Expand a policy name into ``[(img_size, flip), ...]``.

    Only two axes are used, both chosen because they are *label-preserving* on
    camera-trap imagery:

    - **Horizontal flip.** A mirrored animal is the same species. (Vertical
      flip is deliberately absent: nothing in this dataset is upside down, so
      it would push the model off its training distribution rather than
      probing a genuine invariance.)
    - **Scale.** Animals appear at wildly different apparent sizes, and the
      box crop already normalises that only approximately. 1.25x of the
      training resolution is a mild, safe step; the image is re-resized from
      the original file, so it genuinely carries more detail.

    Photometric augmentation (jitter, grayscale, blur) is *not* used at test
    time. It helps during training by forcing invariance, but at inference it
    only degrades each view, and averaging degraded predictions is worse than
    averaging good ones.
    """
    if policy not in TTA_POLICIES:
        raise ValueError(f"unknown TTA policy {policy!r}; choose from {sorted(TTA_POLICIES)}")
    spec = TTA_POLICIES[policy]
    scales = spec["scales"] or (1.0,)
    flips = (False, True) if spec["hflip"] else (False,)
    views = []
    for s in scales:
        # Round to a multiple of the ViT patch size so no partial patch is
        # created at the border.
        size = int(round(base_img_size * s / 16)) * 16
        for f in flips:
            views.append((size, f))
    return views


def collect_logits_tta(
    model: nn.Module,
    cfg: "EvalConfig",
    data_root,
    split: str,
    device,
    policy: str = "none",
    batch_size: int = 32,
    num_workers: int = 4,
    amp: bool = False,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average predictions over the TTA views; return ``(logits, targets, raw)``.

    Averaging is done in **probability** space (arithmetic mean), not logit
    space. Averaging logits is a geometric mean of probabilities, which keeps
    the result about as sharp as the individual views and so throws away the
    calibration benefit -- the whole reason TTA helps ECE/NLL/Brier is that
    disagreement between views *should* show up as reduced confidence. It also
    means TTA can move accuracy and misclassification AUROC, unlike a pure
    recalibration, because the argmax and the confidence ranking both change.

    The averaged probabilities are returned as ``log(p)`` so that everything
    downstream keeps working on a logit-shaped array; ``softmax(log(p)/T)``
    behaves exactly as temperature scaling should.

    ``raw`` is the single-view (first view, unflipped) logit array, so callers
    can compare against the no-TTA baseline without a second pass.
    """
    views = tta_views(policy, int(cfg["img_size"]))
    summed: np.ndarray | None = None
    targets: np.ndarray | None = None
    raw: np.ndarray | None = None

    for i, (size, flip) in enumerate(views):
        ds = cfg.dataset(data_root, split, img_size=size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
        label = f"{split} tta {i+1}/{len(views)} ({size}px{', flip' if flip else ''})"
        logits, tgt = collect_logits(
            model, loader, device, amp=amp, tta_hflip=False,
            hflip_only=flip, progress=label if progress else None,
        )
        probs = _softmax(logits)
        summed = probs if summed is None else summed + probs
        if targets is None:
            targets, raw = tgt, logits

    mean_probs = summed / len(views)
    return np.log(np.clip(mean_probs, 1e-12, None)), targets, raw


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device,
    temperature: float = 1.0,
    calibration: dict | None = None,
    tta_hflip: bool = False,
    amp: bool = False,
    progress: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and return ``(probs, labels)`` as np arrays."""
    logits, labels = collect_logits(model, loader, device, amp=amp, tta_hflip=tta_hflip,
                                    progress=progress)
    return apply_calibration(logits, calibration, temperature), labels


def evaluate(
    model: nn.Module, loader: DataLoader, device, temperature: float = 1.0,
    calibration: dict | None = None,
) -> dict:
    probs, labels = collect_predictions(model, loader, device, temperature, calibration)
    return compute_all_metrics(probs, labels)


def evaluate_by_domain(
    model: nn.Module,
    val_ds: IWildCamChallengeDataset,
    device,
    temperature: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 4,
    calibration: dict | None = None,
    tta_hflip: bool = False,
    amp: bool = False,
) -> dict[str, dict]:
    """Metrics on all of val, plus separately for each `domain` value ('id'/'ood').

    `val`'s domain column marks whether a row is from a camera location seen
    in train ('id') or a held-out location ('ood') — splitting the metrics
    this way shows whether a model is quietly overconfident on OOD data even
    when its aggregate numbers look fine.

    Uses `shuffle=False` so prediction rows line up 1:1, in order, with
    `val_ds.domains`.
    """
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, labels = collect_predictions(
        model, loader, device, temperature, calibration, tta_hflip=tta_hflip, amp=amp
    )
    labels = labels.astype(int)

    results = {"overall": compute_all_metrics(probs, labels)}
    if val_ds.domains is not None:
        domains = np.asarray(val_ds.domains)
        for domain in sorted(set(val_ds.domains)):
            mask = domains == domain
            results[domain] = compute_all_metrics(probs[mask], labels[mask])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the val split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true",
                        help="bf16 autocast during eval. Faster, but perturbs ECE/AUROC slightly.")
    parser.add_argument("--tta", type=str, default="auto",
                        choices=sorted(TTA_POLICIES) + ["auto"],
                        help="Test-time augmentation policy. 'auto' uses whatever "
                             "student.calibrate stored in the checkpoint.")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Ignore the checkpoint's fitted calibration (report raw softmax).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, temperature, cfg = load_checkpoint(args.checkpoint, device)
    if args.no_calibration:
        temperature, calibration = 1.0, None
    else:
        calibration = cfg["calibration"]

    tta = args.tta
    if tta == "auto":
        tta = str((calibration or {}).get("tta", "none"))
    val_ds = cfg.dataset(args.data_root, "val")

    logits, labels, _ = collect_logits_tta(
        model, cfg, args.data_root, "val", device, policy=tta,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=args.amp,
    )
    labels = np.asarray(labels, dtype=int)
    probs = apply_calibration(logits, calibration, temperature)

    results = {"overall": compute_all_metrics(probs, labels)}
    if val_ds.domains is not None:
        domains = np.asarray(val_ds.domains)
        for domain in sorted(set(val_ds.domains)):
            mask = domains == domain
            results[domain] = compute_all_metrics(probs[mask], labels[mask])
    print(f"TTA policy: {tta} "
          f"({len(tta_views(tta, int(cfg['img_size'])))} view(s) per image)")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
