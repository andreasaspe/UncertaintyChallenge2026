"""Trainer with early stopping, LoRA support and post-hoc temperature scaling.

Two checkpoints are written:

- ``model.pt``              — best weights (early-stopped on a val metric), ``T = 1.0``
- ``model_temp_scaled.pt``  — same weights, with scalar ``T`` learned on val

plus ``history.csv`` (all five scored metrics per epoch, split by domain) and
``curves.png``.

Things to modify:
- ``Trainer.train_epoch``    — your optimization step / loss
- ``Trainer.evaluate_val``   — your validation metrics
- ``fit_temperature``        — swap NLL for another calibration objective
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from student import metrics as M
from student.data import (
    IMG_SIZE,
    IWildCamChallengeDataset,
    default_eval_transform,
    default_train_transform,
    make_box_cropper,
)
from student.model import DEFAULT_BACKBONE, DEFAULT_LORA_TARGETS, Classifier

# All five scored metrics, in the order used for the history columns.
METRIC_NAMES = ("accuracy", "ece", "nll", "brier", "misclassification_auroc")


# --------------------------------------------------------------------------
# optimizer / schedule
# --------------------------------------------------------------------------

def _split_decay(named_params) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split (name, param) pairs into (decay, no_decay).

    Convention: 1-D parameters (biases, LayerNorm scale/shift) skip weight
    decay; 2-D+ weight matrices get it. Matches the Karpathy / nanoGPT recipe.
    Frozen parameters are dropped entirely, which is what keeps a LoRA run's
    optimizer state small — only the adapters ever reach AdamW.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def make_optimizer(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    lr_lora: float | None = None,
) -> optim.Optimizer:
    """AdamW with separate param groups for backbone / LoRA adapters / head.

    - Head gets the highest ``lr_head`` — it is randomly initialised.
    - LoRA adapters get ``lr_lora``. They tolerate a much higher LR than a
      full fine-tune would, because ``lora_B`` starts at zero and the frozen
      base weights cannot drift, so there is nothing to catastrophically
      forget.
    - Any remaining trainable backbone weight gets the conservative
      ``lr_backbone`` (only relevant for a non-LoRA full fine-tune).
    """
    if lr_lora is None:
        lr_lora = lr_backbone

    head_param_ids = {id(p) for p in model.head.parameters()}
    head_named, lora_named, backbone_named = [], [], []
    for n, p in model.named_parameters():
        if id(p) in head_param_ids:
            head_named.append((n, p))
        elif ".lora_A" in n or ".lora_B" in n:
            lora_named.append((n, p))
        else:
            backbone_named.append((n, p))

    bb_decay, bb_no_decay = _split_decay(backbone_named)
    lo_decay, lo_no_decay = _split_decay(lora_named)
    hd_decay, hd_no_decay = _split_decay(head_named)

    groups = [
        {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": lo_decay,    "lr": lr_lora,     "weight_decay": weight_decay},
        {"params": lo_no_decay, "lr": lr_lora,     "weight_decay": 0.0},
        {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay},
        {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0},
    ]
    # AdamW errors on empty param groups in some versions; drop them.
    return optim.AdamW([g for g in groups if g["params"]])


def make_scheduler(
    optimizer: optim.Optimizer, total_steps: int, warmup_frac: float = 0.03
) -> optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine decay to zero, stepped **per optimizer step**.

    Per-*epoch* cosine (what this file used to do) is far too coarse here:
    18,929 images at an effective batch of 256 is only ~74 steps per epoch, so
    the LR would move in 15 giant jumps. Warmup matters for LoRA specifically
    because ``lora_A`` is randomly initialised — the first few steps otherwise
    inject large random perturbations into an otherwise well-behaved backbone.
    """
    warmup_steps = max(1, int(round(total_steps * warmup_frac)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Metrics where a *larger* value is better. Everything else (ece, nll, brier)
# is a loss to be minimised.
HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset(
    {"accuracy", "misclassification_auroc"}
)


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------

@dataclass
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    criterion: nn.Module
    device: torch.device
    patience: int = 3
    early_stop_metric: str = "accuracy"
    grad_accum: int = 1
    grad_clip: float = 0.0
    amp_dtype: torch.dtype | None = None
    val_domains: np.ndarray | None = None
    history_path: Path | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.early_stop_metric in HIGHER_IS_BETTER_METRICS:
            self.best_val_metric: float = float("-inf")
        else:
            self.best_val_metric = float("inf")
        self.best_state_dict: dict | None = None
        self.best_epoch: int = 0
        self.epochs_no_improve: int = 0
        self._trainable = [p for p in self.model.parameters() if p.requires_grad]

    def _autocast(self):
        if self.amp_dtype is None or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def _is_improved(self, value: float) -> bool:
        if self.early_stop_metric in HIGHER_IS_BETTER_METRICS:
            return value > self.best_val_metric
        return value < self.best_val_metric

    def train_epoch(self) -> tuple[float, float]:
        """One pass over the training set with gradient accumulation.

        The dataloader yields *micro*-batches; we accumulate ``grad_accum`` of
        them before stepping, so the effective batch is
        ``micro_batch_size * grad_accum``. The loss is divided by
        ``grad_accum`` so the accumulated gradient is a mean, not a sum, and
        the learning rate keeps its usual meaning.

        No GradScaler: bf16 has the same exponent range as fp32, so unlike
        fp16 it cannot underflow gradients and needs no loss scaling.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        self.optimizer.zero_grad(set_to_none=True)
        n_batches = len(self.train_loader)

        for i, (imgs, labels) in enumerate(
            tqdm(self.train_loader, desc="train", leave=False)
        ):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self._autocast():
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            (loss / self.grad_accum).backward()

            is_last = (i + 1) == n_batches
            if (i + 1) % self.grad_accum == 0 or is_last:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._trainable, self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * imgs.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += imgs.size(0)

        return total_loss / total, total_correct / total

    @torch.no_grad()
    def collect_val_logits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, labels)`` for the whole val split, on device, fp32."""
        self.model.eval()
        logits_chunks, labels_chunks = [], []
        for imgs, labels in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(imgs)
            logits_chunks.append(logits.float())
            labels_chunks.append(labels.to(self.device))
        return torch.cat(logits_chunks), torch.cat(labels_chunks)

    def evaluate_val(self) -> tuple[dict[str, float], float]:
        """All five scored metrics on val, raw *and* temperature-scaled.

        Two things are worth understanding here:

        - We compute the metrics with ``student.metrics``, the identical file
          the server scores with, so the training curves are in the same units
          as the leaderboard.
        - We refit the temperature every epoch and report both variants. Raw
          ECE/NLL curves are dominated by the model's overconfidence, which
          post-hoc temperature scaling removes anyway — so the raw curve tells
          you almost nothing about the model you will actually submit. Fitting
          ``T`` on val is not cheating here: it is exactly what the final
          checkpoint does, and ``T`` is a single parameter.

        Returns ``(flat_metric_dict, temperature)``. Keys are
        ``{scope}_{variant}_{metric}`` with scope in overall/id/ood and
        variant in raw/ts.
        """
        logits, labels = self.collect_val_logits()
        T = fit_temperature(logits, labels)

        labels_np = labels.cpu().numpy()
        out: dict[str, float] = {}
        for variant, temp in (("raw", 1.0), ("ts", T)):
            probs = torch.softmax(logits / temp, dim=1).cpu().numpy()
            scopes = {"overall": np.ones(len(labels_np), dtype=bool)}
            if self.val_domains is not None:
                for d in sorted(set(self.val_domains.tolist())):
                    scopes[d] = self.val_domains == d
            for scope, mask in scopes.items():
                for name, value in M.compute_all_metrics(probs[mask], labels_np[mask]).items():
                    # misclassification_auroc is None when val is all-correct
                    # or all-wrong; NaN keeps the CSV column numeric.
                    out[f"{scope}_{variant}_{name}"] = float("nan") if value is None else float(value)
        return out, T

    def _write_history(self) -> None:
        """Rewrite history.csv after every epoch.

        Deliberately not deferred to the end of training: an earlier 320px run
        in this project was killed at epoch 10 and left nothing but a config
        file, losing all evidence of what it had achieved.
        """
        if self.history_path is None or not self.history:
            return
        cols = list(self.history[-1].keys())
        with self.history_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in self.history:
                w.writerow(row)

    def fit(self, epochs: int) -> None:
        stop_key = f"overall_ts_{self.early_stop_metric}"
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val, T = self.evaluate_val()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "lr": self.optimizer.param_groups[-1]["lr"],
                "temperature": T,
                **val,
            }
            self.history.append(row)
            self._write_history()

            print(
                f"epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"| T={T:.3f} | val acc={val['overall_ts_accuracy']:.4f} "
                f"ece={val['overall_ts_ece']:.4f} nll={val['overall_ts_nll']:.4f} "
                f"brier={val['overall_ts_brier']:.4f} "
                f"auroc={val['overall_ts_misclassification_auroc']:.4f}"
                + (f" | id={val.get('id_ts_accuracy', float('nan')):.4f}"
                   f" ood={val.get('ood_ts_accuracy', float('nan')):.4f}"
                   if self.val_domains is not None else "")
            )

            current = row[stop_key]
            if self._is_improved(current):
                self.best_val_metric = current
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"early stopping at epoch {epoch} (no improvement in "
                          f"val/{self.early_stop_metric} for {self.patience} epochs)")
                    break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print(f"restored best epoch {self.best_epoch} "
                  f"(val/{self.early_stop_metric}={self.best_val_metric:.4f})")


# --------------------------------------------------------------------------
# calibration + checkpointing
# --------------------------------------------------------------------------

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Optimize a single scalar ``T`` to minimize NLL on ``(logits, labels)``.

    Temperature scaling divides the logits by ``T`` before the softmax. ``T>1``
    softens the distribution (less confident), ``T<1`` sharpens it. It cannot
    change the argmax, so accuracy is untouched — it only fixes *how confident*
    the model claims to be, which is what ECE / NLL / Brier measure.

    Parameterized as ``T = exp(log_T)`` so the optimizer stays in (0, inf)
    without bounds constraints. Falls back to ``T = 1.0`` if LBFGS diverges.
    """
    logits = logits.detach().float()
    labels = labels.detach()
    log_T = nn.Parameter(torch.zeros(1, device=logits.device))
    optimizer = optim.LBFGS([log_T], lr=0.1, max_iter=100)
    # NOTE: plain CrossEntropyLoss, never label-smoothed. Calibration must be
    # fit against the true NLL, which is what the server scores.
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(log_T.exp().detach().cpu())
    if not (T > 0 and T < float("inf")):
        return 1.0
    return T


def temperature_scale(
    model: nn.Module, val_loader: DataLoader, device,
    amp_dtype: torch.dtype | None = None,
) -> float:
    """Collect val logits, then fit a single scalar T against NLL.

    ``amp_dtype`` must match training: when the frozen backbone is stored in
    bf16, a plain fp32 forward pass is a dtype mismatch, not just slower.
    """
    model.eval()
    ctx = (torch.autocast("cuda", dtype=amp_dtype)
           if amp_dtype is not None and torch.device(device).type == "cuda"
           else nullcontext())
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            with ctx:
                logits_list.append(model(imgs).float())
            labels_list.append(labels.to(device))
    return fit_temperature(torch.cat(logits_list), torch.cat(labels_list))


def save_checkpoint(
    model: nn.Module,
    num_classes: int,
    temperature: float,
    path: Path,
    hyperparameters: dict | None = None,
    lora_only: bool = False,
) -> None:
    """Write a checkpoint in the format ``eval.load_checkpoint`` expects.

    With ``lora_only=True`` only the trainable tensors (LoRA adapters + head)
    are stored — a few hundred MB instead of the 27 GB a merged 7B backbone
    would need. The frozen base weights are re-fetched from timm at load time,
    which is deterministic: they are the published pretrained weights and were
    never modified.
    """
    state_dict = model.trainable_state_dict() if lora_only else model.state_dict()
    ckpt = {
        "state_dict": state_dict,
        "num_classes": int(num_classes),
        "temperature": float(temperature),
        "backbone": getattr(model, "backbone_name", DEFAULT_BACKBONE),
        "lora_only": bool(lora_only),
    }
    if hyperparameters is not None:
        ckpt["hyperparameters"] = dict(hyperparameters)
    torch.save(ckpt, path)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_root: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    micro_batch_size: int | None = None,
    lr: float = 1e-4,
    head_lr: float = 1e-3,
    lora_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 3,
    num_workers: int = 4,
    img_size: int = IMG_SIZE,
    label_smoothing: float = 0.0,
    augment: str = "full",
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = False,
    early_stop_metric: str = "accuracy",
    crop_frac: float = 0.0,
    box_crop: bool = False,
    box_conf: float = 0.2,
    box_margin: float = 0.15,
    box_min_frac: float = 0.30,
    box_max_aspect: float = 1.6,
    box_no_mask_banner: bool = False,
    box_no_strict: bool = False,
    feature_pool: str = "default",
    head: str = "linear",
    head_hidden: int = 1024,
    head_dropout: float = 0.0,
    lora_r: int = 0,
    lora_alpha: float = 0.0,
    lora_dropout: float = 0.0,
    lora_targets: str = ",".join(DEFAULT_LORA_TARGETS),
    amp: str = "fp32",
    frozen_dtype: str = "bf16",
    grad_checkpointing: bool = False,
    grad_clip: float = 0.0,
    warmup_frac: float = 0.03,
    rrc_scale_min: float = 0.32,
    rrc_ratio: str = "0.75,1.333",
    solarize_p: float = 0.1,
    save_merged: bool = False,
    seed: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    # TF32 for the fp32 matmuls that remain (LoRA params, head, temperature fit).
    torch.set_float32_matmul_precision("high")

    micro_batch_size = int(micro_batch_size or batch_size)
    grad_accum = max(1, math.ceil(batch_size / micro_batch_size))
    lora_targets_t = tuple(t for t in str(lora_targets).split(",") if t.strip())
    rrc_ratio_t = tuple(float(v) for v in str(rrc_ratio).split(","))

    hparams = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "micro_batch_size": int(micro_batch_size),
        "grad_accum": int(grad_accum),
        "lr": float(lr),
        "head_lr": float(head_lr),
        "lora_lr": float(lora_lr),
        "weight_decay": float(weight_decay),
        "patience": int(patience),
        "num_workers": int(num_workers),
        "img_size": int(img_size),
        "label_smoothing": float(label_smoothing),
        "augment": str(augment),
        "backbone": str(backbone),
        "pretrained": bool(pretrained),
        "early_stop_metric": str(early_stop_metric),
        "crop_frac": float(crop_frac),
        "box_crop": bool(box_crop),
        "box_conf": float(box_conf),
        "box_margin": float(box_margin),
        "box_min_frac": float(box_min_frac),
        "box_max_aspect": float(box_max_aspect),
        "box_no_mask_banner": bool(box_no_mask_banner),
        "box_no_strict": bool(box_no_strict),
        "feature_pool": str(feature_pool),
        "head": str(head),
        "head_hidden": int(head_hidden),
        "head_dropout": float(head_dropout),
        "lora_r": int(lora_r),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_targets": list(lora_targets_t),
        "amp": str(amp),
        "frozen_dtype": str(frozen_dtype),
        "grad_checkpointing": bool(grad_checkpointing),
        "grad_clip": float(grad_clip),
        "warmup_frac": float(warmup_frac),
        "rrc_scale_min": float(rrc_scale_min),
        "rrc_ratio": list(rrc_ratio_t),
        "solarize_p": float(solarize_p),
        "seed": int(seed),
        "mean": None,  # filled in below from the backbone's published config
        "std": None,
        "data_root": str(data_root),
    }
    (output_dir / "config.json").write_text(json.dumps(hparams, indent=2))

    # Build the model first so its published normalization stats can be fed
    # into the transforms. Hardcoding ImageNet mean/std silently mis-normalizes
    # any backbone that was pretrained with different ones.
    model = Classifier(
        _num_classes(data_root),
        backbone_name=backbone,
        pretrained=pretrained,
        img_size=img_size,
        feature_pool=feature_pool,
        head=head,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets_t,
    )
    data_cfg = timm_data_config(model.backbone)
    mean, std = tuple(data_cfg["mean"]), tuple(data_cfg["std"])
    hparams["mean"], hparams["std"] = list(mean), list(std)
    (output_dir / "config.json").write_text(json.dumps(hparams, indent=2))

    # img_size and crop_frac are recorded in hparams -> checkpoint, so
    # eval/predict/ood all reproduce the exact input pipeline this model was
    # trained on. Evaluating a 320px model at 224px, or an edge-cropped model
    # on uncropped frames, silently wrecks its accuracy.
    tf_crop_frac = 0.0 if box_crop else crop_frac
    train_tf = default_train_transform(
        img_size, augment, crop_frac=tf_crop_frac, mean=mean, std=std,
        rrc_scale_min=rrc_scale_min, rrc_ratio=rrc_ratio_t, solarize_p=solarize_p,
    )
    eval_tf = default_eval_transform(img_size, crop_frac=tf_crop_frac, mean=mean, std=std)
    # Box cropping owns ALL cropping when enabled (it applies the banner clamp
    # itself), so crop_frac must not also fire or we would crop twice.
    cropper = make_box_cropper(
        data_root, enabled=box_crop, conf_threshold=box_conf, margin=box_margin,
        min_frac=box_min_frac, max_aspect=box_max_aspect, banner_frac=crop_frac or 0.05,
        mask_banner=not box_no_mask_banner, strict_containment=not box_no_strict,
    )
    if cropper is not None:
        print(f"box crop: {cropper}")

    train_ds = IWildCamChallengeDataset(data_root, "train", train_tf, box_cropper=cropper)
    val_ds = IWildCamChallengeDataset(data_root, "val", eval_tf, box_cropper=cropper)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True,
                              drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False,
                            **loader_kwargs)

    amp_dtype = torch.bfloat16 if amp == "bf16" else None
    if lora_r > 0 and frozen_dtype == "bf16" and amp_dtype is torch.bfloat16:
        # Frozen weights are never updated, so they don't need an fp32 master
        # copy — storing them in bf16 halves backbone memory (13.4 GB instead
        # of 26.8 GB for the 7B). The trainable LoRA/head params stay fp32,
        # where the small optimizer updates actually need the precision.
        for p in model.parameters():
            if not p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
    model.to(device)

    if grad_checkpointing:
        model.set_grad_checkpointing(True)

    n_trainable = sum(p.numel() for _, p in model.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"backbone={backbone} img_size={img_size} embed_dim={model.embed_dim}")
    print(f"trainable {n_trainable/1e6:.2f}M / {n_all/1e6:.2f}M params "
          f"({100*n_trainable/max(1,n_all):.2f}%) | lora modules: {len(model.lora_modules)}")
    print(f"effective batch {batch_size} = micro {micro_batch_size} x accum {grad_accum} "
          f"| amp={amp} grad_ckpt={grad_checkpointing}")

    optimizer = make_optimizer(model, lr_backbone=lr, lr_head=head_lr,
                               weight_decay=weight_decay, lr_lora=lora_lr)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    scheduler = make_scheduler(optimizer, total_steps=epochs * steps_per_epoch,
                               warmup_frac=warmup_frac)
    # Label smoothing: train toward (1 - eps) on the true class instead of a
    # hard 1.0. Measured harmful on this dataset (it makes the model
    # *under*-confident, so temperature scaling has to sharpen rather than
    # soften, and NLL suffers) — default 0.0.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, patience=patience, early_stop_metric=early_stop_metric,
        grad_accum=grad_accum, grad_clip=grad_clip, amp_dtype=amp_dtype,
        val_domains=np.asarray(val_ds.domains) if val_ds.domains else None,
        history_path=output_dir / "history.csv",
    )
    trainer.fit(epochs)

    lora_only = lora_r > 0 and not save_merged
    if save_merged and lora_r > 0:
        n = model.merge_lora()
        # The adapters no longer exist as separate modules, so the checkpoint
        # must not tell load_checkpoint to recreate them.
        hparams["lora_r"] = 0
        print(f"merged {n} LoRA adapters into the base weights")

    save_checkpoint(model, train_ds.num_classes, 1.0, output_dir / "model.pt",
                    hyperparameters=hparams, lora_only=lora_only)
    print("saved model.pt")

    T = temperature_scale(model, val_loader, device, amp_dtype=amp_dtype)
    save_checkpoint(model, train_ds.num_classes, T, output_dir / "model_temp_scaled.pt",
                    hyperparameters=hparams, lora_only=lora_only)
    print(f"learned T={T:.4f} -> saved model_temp_scaled.pt")

    try:
        from student.plots import plot_history
        out = plot_history(output_dir / "history.csv", output_dir / "curves.png",
                           best_epoch=trainer.best_epoch)
        print(f"wrote {out}")
    except Exception as exc:  # plotting must never lose a finished run
        print(f"[warn] could not write curves.png: {exc}")


def _num_classes(data_root) -> int:
    with (Path(data_root) / "class_mapping.json").open() as f:
        return int(json.load(f)["num_classes"])


def timm_data_config(backbone: nn.Module) -> dict:
    """Published mean/std for this backbone, falling back to ImageNet."""
    try:
        import timm.data
        return timm.data.resolve_model_data_config(backbone)
    except Exception:
        from student.data import IMAGENET_MEAN, IMAGENET_STD
        return {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier on the iWildCam challenge data.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to challenge_data/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save model.pt, model_temp_scaled.pt, history.csv, curves.png")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="EFFECTIVE batch size (micro-batches are accumulated to reach it).")
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Images per forward pass. Defaults to --batch-size. "
                             "Lower this until it fits in VRAM; --batch-size is unaffected.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Backbone learning rate (unfrozen weights only; unused with LoRA).")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="Head learning rate (higher LR for the new classification layer).")
    parser.add_argument("--lora-lr", type=float, default=1e-4,
                        help="LoRA adapter learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment", type=str, default="full",
                        choices=["full", "basic", "dinov3"],
                        help="'full' = crop/jitter/grayscale; 'basic' = starter-kit resize+flip; "
                             "'dinov3' = DINOv3's own global-crop pretraining recipe.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="CrossEntropy label smoothing eps. Measured harmful here; keep 0.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stop after this many epochs without val-metric improvement.")
    parser.add_argument("--early-stop-metric", type=str, default="accuracy",
                        choices=list(METRIC_NAMES),
                        help="Which temperature-scaled val metric drives early stopping.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE,
                        help="Input resolution. DINOv3 was pretrained at 256.")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        help="timm model id (e.g. vit_7b_patch16_dinov3.lvd1689m).")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialize the backbone from timm's pretrained weights.")
    parser.add_argument("--crop-frac", type=float, default=0.0,
                        help="Fraction of the frame cut from the top AND bottom before any "
                             "other transform. 0.05 removes the camera-trap text banner, "
                             "which is a shortcut-learning channel.")
    parser.add_argument("--box-crop", action="store_true",
                        help="Crop each image to the union of its MegaDetector animal boxes "
                             "(from <data-root>/boxes.csv), instead of the plain banner crop. "
                             "~79%% of images have a detection; the rest fall back to the "
                             "banner crop. Median linear zoom ~2.2x.")
    parser.add_argument("--box-conf", type=float, default=0.2,
                        help="Minimum MegaDetector confidence for a box to be used.")
    parser.add_argument("--box-margin", type=float, default=0.15,
                        help="Context padding, as a fraction of the union box size per side.")
    parser.add_argument("--box-min-frac", type=float, default=0.30,
                        help="Minimum crop size as a fraction of the frame, per dimension.")
    parser.add_argument("--box-max-aspect", type=float, default=1.6,
                        help="Cap on crop elongation, so squashing to square stays sane.")
    parser.add_argument("--box-no-mask-banner", action="store_true",
                        help="Do NOT black out the banner strips before cropping. "
                             "Masking is on by default because 23%% of box crops otherwise "
                             "reach the rows holding the camera-ID text.")
    parser.add_argument("--box-no-strict", action="store_true",
                        help="Let the banner clamp cut off animals rather than growing the "
                             "crop to contain them. Affects ~36%% of detected images.")
    parser.add_argument("--feature-pool", type=str, default="default",
                        choices=["default", "cls_avg"],
                        help="'cls_avg' = concat(CLS, mean patch tokens), DINOv3's linear-eval pooling.")
    parser.add_argument("--head", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--head-hidden", type=int, default=1024)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=0,
                        help="LoRA rank. 0 disables LoRA (full fine-tuning).")
    parser.add_argument("--lora-alpha", type=float, default=0.0,
                        help="LoRA scaling numerator; convention is 2*r.")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", type=str, default=",".join(DEFAULT_LORA_TARGETS),
                        help="Comma-separated Linear names inside blocks.* to adapt.")
    parser.add_argument("--amp", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="bf16 autocast. No GradScaler needed (bf16 has fp32's exponent range).")
    parser.add_argument("--frozen-dtype", type=str, default="bf16", choices=["fp32", "bf16"],
                        help="Storage dtype for frozen LoRA base weights. bf16 halves backbone memory.")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="Recompute block activations in backward: ~30%% more compute, ~10x less memory.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Global grad-norm clip. 1.0 is a safe default for large ViTs.")
    parser.add_argument("--warmup-frac", type=float, default=0.03,
                        help="Fraction of total steps spent in linear LR warmup.")
    parser.add_argument("--rrc-scale-min", type=float, default=0.32,
                        help="RandomResizedCrop lower area bound for --augment dinov3.")
    parser.add_argument("--rrc-ratio", type=str, default="0.75,1.333",
                        help="RandomResizedCrop aspect bounds for --augment dinov3.")
    parser.add_argument("--solarize-p", type=float, default=0.1,
                        help="Solarize probability for --augment dinov3. Set 0 for IR-heavy data.")
    parser.add_argument("--save-merged", action="store_true",
                        help="Fold LoRA into the base weights and save a full checkpoint "
                             "(27 GB for the 7B). Off by default: LoRA-only is ~250 MB.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
