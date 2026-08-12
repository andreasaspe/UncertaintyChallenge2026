"""Dataset and dataloaders for the iWildCam summer-school challenge.

Reads a prepared ``challenge_data/`` directory:

    challenge_data/
        train/{images/<uid>.<ext>, labels.csv}
        val/{images/<uid>.<ext>, labels.csv}
        test_public/images/<uid>.<ext>       (no labels)
        class_mapping.json

Common things to tweak:
- ``IMG_SIZE`` — image side. Lower it to speed up training.
- ``default_train_transform`` — your augmentations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_SPLITS = {"train", "val", "test_public", "test_private"}


class IWildCamChallengeDataset(Dataset):
    """One row per image. Returns ``(image_tensor, label_int)`` for train/val,
    ``(image_tensor, uid_str)`` for test_public (which has no labels).

    Exposes:
        self.num_classes : int
        self.uids        : list[str]
        self.labels      : list[int] | None
        self.domains     : list[str] | None        (val only: 'id'/'ood')
    """

    def __init__(self, root, split: str, transform: Optional[Callable] = None):
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.images_dir = self.root / split / "images"

        with (self.root / "class_mapping.json").open() as f:
            self.num_classes = int(json.load(f)["num_classes"])

        labels_path = self.root / split / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            self.uids = df["uid"].astype(str).tolist()
            self.labels = df["y"].astype(int).tolist()
            self.domains = df["domain"].astype(str).tolist() if "domain" in df.columns else None
        else:
            # test_public: images only.
            self.uids = sorted(p.stem for p in self.images_dir.iterdir() if p.is_file())
            self.labels = None
            self.domains = None

    def __len__(self) -> int:
        return len(self.uids)

    def _find_image(self, uid: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.images_dir / f"{uid}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no image for uid {uid} in {self.images_dir}")

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        img = Image.open(self._find_image(uid)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img, uid
        return img, int(self.labels[idx])


def default_train_transform(img_size: int = IMG_SIZE, augment: str = "full") -> Callable:
    """Training augmentation, tuned for camera-trap imagery.

    ``augment="basic"`` reproduces the original starter-kit pipeline (squash
    resize + horizontal flip) so the heavy augmentation below can be ablated
    against it under otherwise identical hyperparameters. Run A compared
    augmentation against the baseline while the training dynamics were also
    broken, so its effect has never actually been isolated.

    Each piece of the "full" pipeline targets a specific way this dataset
    varies across camera locations — the thing that makes held-out locations
    ("ood") hard:

    - ``RandomResizedCrop`` replaces a plain squashing ``Resize``. Animals are
      often a small blob in a corner, so random zoom/crop both preserves
      aspect ratio and forces the model to recognise species at many apparent
      scales instead of memorising "big centred animal".
    - ``ColorJitter`` covers exposure/white-balance differences between
      cameras — a major part of what changes at a new location.
    - ``RandomGrayscale`` matters because camera traps switch to infrared at
      night, so a chunk of the data is effectively greyscale. Randomly
      dropping colour stops the model leaning on colour cues that vanish
      after dark.

    ``scale=(0.6, 1.0)`` is deliberately not the ImageNet default of 0.08 —
    aggressive cropping would frequently cut the animal out of the frame
    entirely and hand the model a mislabelled image.
    """
    if augment == "basic":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if augment != "full":
        raise ValueError(f"augment must be 'full' or 'basic', got {augment!r}")
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_eval_transform(img_size: int = IMG_SIZE) -> Callable:
    """Deterministic eval transform: resize the *whole* frame, no cropping.

    Deliberately not the usual Resize-then-CenterCrop: a centre crop would
    discard the frame edges, and camera-trap animals frequently sit right at
    the edge. Keeping the full field of view costs some aspect-ratio
    distortion but never throws the subject away.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    root, batch_size: int = 32, num_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)`` with sensible defaults."""
    train_ds = IWildCamChallengeDataset(root, "train", default_train_transform())
    val_ds = IWildCamChallengeDataset(root, "val", default_eval_transform())
    test_ds = IWildCamChallengeDataset(root, "test_public", default_eval_transform())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
