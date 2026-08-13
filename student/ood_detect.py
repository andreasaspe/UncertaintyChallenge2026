"""OOD detection in a frozen DINOv2 latent space.

Why a *frozen self-supervised* encoder rather than our fine-tuned classifier
-----------------------------------------------------------------------------
The earlier gate (``student/ood.py``) scored OOD using the fine-tuned
classifier's ``embed()`` features and got only AUROC 0.638 at separating val-id
from val-ood. That is likely a measurement artifact: fine-tuning for *species*
classification explicitly trains the network to be **invariant** to background
and camera location — which is exactly the signal an OOD detector needs. We were
looking for location information in a space engineered to discard it.

DINOv2 is trained self-supervised on generic images and never learned to throw
scene information away, so it should preserve camera-location structure far
better. This module tests that hypothesis before building anything on top.

Nothing here is trained. DINOv2 is used strictly as a frozen feature extractor.

Supervision discipline
----------------------
``train`` is 100% ``domain == id``, so it is the reference distribution: an OOD
score is a distance *to the train feature bank*. Val's ``id``/``ood`` labels are
used **only** to measure AUROC and to set a threshold — never as input to any
scoring or clustering step. There are 464 val-ood images total, and a supervised
id-vs-ood classifier trained on them would memorise those specific camera
backgrounds and fail on test's different held-out locations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from student.data import IWildCamChallengeDataset, default_eval_transform

DINOV2_MODEL = "vit_base_patch14_dinov2.lvd142m"
SPLITS = ("train", "val", "test_public", "test_private")


def build_encoder(model_name: str = DINOV2_MODEL, img_size: int = 224, device=None):
    """Frozen DINOv2 feature extractor.

    ``num_classes=0`` makes timm return pooled features instead of logits.
    DINOv2's native input is 518x518 with patch-14; we run at 224 (a clean
    multiple of 14 -> 16x16 patches) which is ~5x cheaper.
    ``dynamic_img_size=True`` interpolates the position embeddings so the
    off-native resolution is handled correctly rather than silently breaking.
    """
    model = timm.create_model(
        model_name, pretrained=True, num_classes=0,
        dynamic_img_size=True, img_size=img_size,
    )
    model.eval().to(device)
    for p in model.parameters():          # frozen: no training, ever
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_split(model, data_root, split: str, device, img_size: int,
                 batch_size: int = 64, num_workers: int = 8):
    """Return ``(features, dataset)`` for one split under the eval transform.

    ``shuffle=False`` so row i of the feature matrix corresponds to
    ``ds.uids[i]`` / ``ds.labels[i]`` / ``ds.domains[i]``.
    """
    ds = IWildCamChallengeDataset(data_root, split, default_eval_transform(img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    chunks = []
    for imgs, _ in tqdm(loader, desc=f"encode {split}", leave=False):
        chunks.append(model(imgs.to(device)).float().cpu().numpy())
    return np.concatenate(chunks), ds


def encode_all(data_root, cache: Path, model_name: str = DINOV2_MODEL,
               img_size: int = 224, batch_size: int = 64, num_workers: int = 8) -> dict:
    """Encode every split once and cache to a single npz. Re-runs are instant."""
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"loaded cached features from {cache}")
        return {k: z[k] for k in z.files}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_encoder(model_name, img_size, device)
    out: dict[str, np.ndarray] = {}
    for split in SPLITS:
        feats, ds = encode_split(model, data_root, split, device, img_size,
                                 batch_size, num_workers)
        out[f"{split}_feats"] = feats
        out[f"{split}_uids"] = np.asarray(ds.uids)
        if ds.labels is not None:
            out[f"{split}_y"] = np.asarray(ds.labels)
        if ds.domains is not None:
            out[f"{split}_domain"] = np.asarray(ds.domains)
        print(f"  {split}: {feats.shape}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"cached features to {cache}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Encode all splits with frozen DINOv2.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=str, default=DINOV2_MODEL)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    encode_all(args.data_root, args.cache, args.model, args.img_size,
               args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
