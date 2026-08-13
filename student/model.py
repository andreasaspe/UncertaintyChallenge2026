"""Model definitions for the challenge.

Defines the ``Classifier`` nn.Module used by ``train``, ``eval``, and ``predict``.

The minimal contract (so train / eval / predict don't need to change):

- ``self.head``  is the classifier; its parameters get the higher LR.
- ``self.backbone`` is everything else; its parameters get the lower LR.
- ``forward(x)`` returns logits of shape ``(N, num_classes)``.
- ``embed(x)``   returns features of shape ``(N, embed_dim)``.

Beyond the original starter kit this file adds three things, all off by
default so every existing checkpoint keeps loading unchanged:

- ``img_size`` / ``dynamic_img_size`` plumbing, needed to run a ViT at any
  resolution other than the one it was pretrained at.
- ``feature_pool="cls_avg"``, DINOv3's own linear-evaluation pooling.
- ``lora_r > 0``, LoRA adapters on a frozen backbone (see ``LoRALinear``).
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BACKBONE = "convnext_tiny"
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("qkv", "proj")


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation of a frozen ``nn.Linear``.

    LoRA (Hu et al., 2021) freezes the pretrained weight ``W`` and learns a
    *low-rank correction* to it: instead of updating all of ``W`` (4096x12288
    numbers for one 7B attention projection), it learns two thin matrices
    ``A`` (r x in) and ``B`` (out x r) and uses ``W + (alpha/r) * B @ A``.
    With ``r = 64`` that is ~0.9% of the parameters of the 7B backbone.

    Why this is the right tool here: we have 18,929 training images. A full
    fine-tune of a 6.7B-parameter model on 19k images has vastly more
    capacity than signal and will memorise the training set. Freezing the
    pretrained features and learning only a rank-64 correction keeps the
    generalisation of the DINOv3 representation while still letting the
    network adapt to camera-trap imagery.

    Two details that matter:

    - ``lora_B`` is initialised to **zeros**, so ``B @ A == 0`` and the
      wrapped layer is *bit-identical* to the pretrained layer at step 0.
      Training therefore starts from exactly the pretrained model rather than
      from a randomly perturbed one. (``lora_A`` is randomly initialised; if
      both were zero the gradient would stay zero forever.)
    - ``scaling = alpha / r`` decouples the learning rate from the rank, so
      changing ``r`` doesn't silently change the effective step size. The
      usual convention is ``alpha = 2 * r``.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        # Two skinny matmuls (x @ A.T @ B.T) rather than materialising the
        # full (out x in) delta -- that is the whole point of the low rank.
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + delta * self.scaling

    @torch.no_grad()
    def merge_(self) -> nn.Linear:
        """Fold the adapter into the base weight and return the plain Linear.

        Used by ``--save-merged`` to emit a self-contained checkpoint that
        needs no LoRA machinery to load. Done in fp32 regardless of the
        training dtype, because a rank-64 outer product accumulated in bf16
        loses meaningful precision.
        """
        if not self.merged:
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            self.base.weight.add_(delta.to(self.base.weight.dtype))
            self.merged = True
        return self.base

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3g}, merged={self.merged}"


def _iter_named_linears(module: nn.Module) -> Iterable[tuple[str, nn.Module, str, nn.Linear]]:
    """Yield ``(full_name, parent_module, attr_name, linear)`` for every Linear."""
    for parent_name, parent in module.named_modules():
        for attr, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{parent_name}.{attr}" if parent_name else attr
                yield full, parent, attr, child


def apply_lora(
    backbone: nn.Module,
    r: int,
    alpha: float,
    dropout: float = 0.0,
    targets: Sequence[str] = DEFAULT_LORA_TARGETS,
) -> list[str]:
    """Swap every targeted ``nn.Linear`` inside ``backbone`` for a ``LoRALinear``.

    ``targets`` are matched against the *last* dotted component of the module
    path, so ``("qkv", "proj")`` hits ``blocks.7.attn.qkv`` and
    ``blocks.7.attn.proj``. The match is additionally restricted to
    ``blocks.<i>.*`` so that patch embedding, position embedding and the final
    norm stay frozen -- ``proj`` appears there too on some architectures, where
    adapting it is pointless.

    Returns the list of replaced module paths so the caller can assert the
    injection actually found something.
    """
    target_set = {t.strip() for t in targets if t.strip()}
    replaced: list[str] = []
    # Materialise the list first: we mutate the module tree while iterating.
    for full, parent, attr, linear in list(_iter_named_linears(backbone)):
        if attr not in target_set:
            continue
        if not re.match(r"^blocks\.\d+\.", full):
            continue
        setattr(parent, attr, LoRALinear(linear, r=r, alpha=alpha, dropout=dropout))
        replaced.append(full)
    return replaced


def merge_lora(module: nn.Module) -> int:
    """Merge every ``LoRALinear`` in ``module`` back into a plain ``nn.Linear``.

    After this the module's ``state_dict`` has exactly the same keys as a
    LoRA-free model, so it can be saved and reloaded without any adapter code.
    """
    n = 0
    for parent in module.modules():
        for attr, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, attr, child.merge_())
                n += 1
    return n


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------

class Classifier(nn.Module):
    """timm backbone (as feature extractor) + classifier head.

    Args:
        num_classes: K, the number of output classes.
        backbone_name: any timm model id.
        pretrained: load timm's published weights. With ``lora_r > 0`` this
            must be True at eval time too -- a LoRA-only checkpoint stores
            *only* the adapters, and the frozen base weights come from here.
        img_size: input resolution. Forwarded to timm for ViTs (together with
            ``dynamic_img_size=True``) so position embeddings are interpolated
            correctly; silently ignored by CNN backbones that don't accept it.
        feature_pool: ``"default"`` uses timm's own pooled output.
            ``"cls_avg"`` concatenates the CLS token with the mean of the
            patch tokens -- see ``embed``.
        head: ``"linear"`` or ``"mlp"``.
        lora_r: LoRA rank. ``0`` disables LoRA entirely (full fine-tuning).
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
        img_size: int | None = None,
        feature_pool: str = "default",
        head: str = "linear",
        head_hidden: int = 1024,
        head_dropout: float = 0.0,
        lora_r: int = 0,
        lora_alpha: float = 0.0,
        lora_dropout: float = 0.0,
        lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_pool = feature_pool
        self.head_type = head
        self.lora_r = int(lora_r)

        # ViTs need img_size/dynamic_img_size to run at a non-native
        # resolution; CNNs reject both kwargs. Try the rich call, fall back.
        kwargs = dict(pretrained=pretrained, num_classes=0)
        self.backbone = None
        if img_size is not None:
            try:
                self.backbone = timm.create_model(
                    backbone_name, img_size=img_size, dynamic_img_size=True, **kwargs
                )
            except TypeError:
                self.backbone = None
        if self.backbone is None:
            self.backbone = timm.create_model(backbone_name, **kwargs)

        backbone_dim = int(self.backbone.num_features)
        if feature_pool == "default":
            self.embed_dim = backbone_dim
        elif feature_pool == "cls_avg":
            if not hasattr(self.backbone, "num_prefix_tokens"):
                raise ValueError(
                    f"feature_pool='cls_avg' needs a token-based backbone; "
                    f"{backbone_name!r} has no num_prefix_tokens"
                )
            self.embed_dim = 2 * backbone_dim
        else:
            raise ValueError(f"feature_pool must be 'default' or 'cls_avg', got {feature_pool!r}")

        self.num_classes = int(num_classes)

        if head == "linear":
            # Deliberately a bare linear map. With LoRA already supplying the
            # nonlinear adaptation inside the backbone, an MLP head mostly adds
            # parameters to overfit 19k images with -- and linear heads
            # empirically produce softer, better-calibrated logits, which is
            # three of the five metrics we are scored on.
            self.head: nn.Module = nn.Linear(self.embed_dim, self.num_classes)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, self.num_classes),
            )
        else:
            raise ValueError(f"head must be 'linear' or 'mlp', got {head!r}")

        self.lora_modules: list[str] = []
        if self.lora_r > 0:
            self.freeze_backbone()
            self.lora_modules = apply_lora(
                self.backbone, r=self.lora_r, alpha=lora_alpha,
                dropout=lora_dropout, targets=lora_targets,
            )
            if not self.lora_modules:
                raise ValueError(
                    f"LoRA targets {tuple(lora_targets)} matched no Linear inside "
                    f"blocks.* of {backbone_name!r} -- nothing would train."
                )

    # -- freezing / introspection -----------------------------------------

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Only the tensors that actually train (LoRA adapters + head).

        A merged 7B checkpoint is ~27 GB; this is ~250 MB. The frozen base
        weights are recovered from timm at load time via ``pretrained=True``.
        """
        trainable = {n for n, _ in self.trainable_parameters()}
        return {k: v for k, v in self.state_dict().items() if k in trainable}

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade compute for memory: recompute block activations in backward.

        Roughly 30% more compute for ~10x less activation memory, which is
        what makes batch 256 through a 7B ViT possible at all.
        """
        if not hasattr(self.backbone, "set_grad_checkpointing"):
            raise ValueError(f"{self.backbone_name!r} does not support grad checkpointing")
        self.backbone.set_grad_checkpointing(enable)

    def merge_lora(self) -> int:
        """Fold adapters into the base weights (see module-level ``merge_lora``)."""
        n = merge_lora(self.backbone)
        self.lora_r = 0
        self.lora_modules = []
        return n

    # -- forward ------------------------------------------------------------

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample feature vectors, shape ``(N, embed_dim)``."""
        if self.feature_pool == "default":
            return self.backbone(x)
        # DINOv3's published linear-evaluation protocol: concatenate the CLS
        # token (a global summary) with the mean over patch tokens (a spatial
        # summary). They carry different information -- CLS is shaped by the
        # image-level objective, patch means retain localised evidence, which
        # matters when the animal occupies a small part of the frame. Note
        # forward_features already applies the final LayerNorm.
        tokens = self.backbone.forward_features(x)          # (N, prefix + P, D)
        p = self.backbone.num_prefix_tokens                 # dinov3: 1 CLS + 4 register
        return torch.cat([tokens[:, 0], tokens[:, p:].mean(dim=1)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
