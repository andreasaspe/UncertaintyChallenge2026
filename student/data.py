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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_SPLITS = {"train", "val", "test_public", "test_private"}

# DINOv3 pretraining used the two "global crop" views of the DINO multi-crop
# recipe. These are its parameters, kept here so the augmentation branch below
# reads as a recipe rather than a pile of magic numbers.
DINOV3_GLOBAL_SCALE = (0.32, 1.0)
DINOV3_RRC_RATIO = (0.75, 1.333)

# MegaDetector category ids in boxes.csv (`cls` column).
MEGADETECTOR_ANIMAL = 0


class CropBorders:
    """Drop the top and bottom ``frac`` of the frame before anything else.

    Camera traps burn a text banner into the top/bottom strips of every frame
    (timestamp, temperature, camera ID). That is a *shortcut*: camera ID
    identifies the location, location correlates with which species live
    there, and a network will happily learn to read the banner instead of
    looking at the animal. It scores well on ``id`` val images from cameras it
    memorised and then collapses on the held-out ``ood`` cameras -- exactly the
    generalisation we are scored on.

    Cutting the strips off removes the channel entirely. It must be applied
    identically at train *and* eval time, otherwise the model sees a different
    field of view at test time than it was trained on.

    At the native 796x448 camera-trap size, ``frac=0.05`` gives 796x404.
    """

    def __init__(self, frac: float = 0.05):
        if not 0.0 <= frac < 0.5:
            raise ValueError(f"frac must be in [0, 0.5), got {frac}")
        self.frac = float(frac)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.frac <= 0.0:
            return img
        w, h = img.size
        cut = round(h * self.frac)
        return img.crop((0, cut, w, h - cut))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(frac={self.frac})"


class BoxCropper:
    """Crop each image to the animals MegaDetector found in it.

    ``boxes.csv`` holds one row per detected box with normalized corners
    (``nx1, ny1, nx2, ny2``), a confidence and a class (0=animal, 1=person,
    2=vehicle). This builds, per uid, the single rectangle that contains
    **every** animal box above ``conf_threshold``, pads it for context, and
    crops to that.

    Why this beats the plain top/bottom banner crop:

    - The median union box is only ~29% x 33% of the frame, so cropping is a
      ~3x linear zoom. At a fixed 256px model input that turns a ~75px animal
      into a ~256px animal -- a large gain in effective resolution, which the
      earlier 320px experiment suggested was the strongest accuracy lever
      available.
    - It removes far more of the location-specific background than a banner
      crop does, and background is what makes held-out cameras ("ood") hard.

    Four guards, because a raw detector box is not directly usable:

    ``margin``
        MegaDetector boxes are tight around the animal, and a slightly-off box
        can clip a tail or legs -- features that distinguish species. The box
        is expanded by this fraction of its own size on every side. 0.15 keeps
        the median linear zoom at ~2.2x (vs ~1.9x at 0.25) while still giving
        the classifier a margin of error; ``min_frac`` supplies the context
        floor for small detections.
    ``min_frac``
        The 5th-percentile box is only 7% of the frame wide. Upscaling that to
        256px is pure interpolation blur with no context left, so the crop is
        expanded about its centre to cover at least this fraction of the frame
        in each dimension.
    ``max_aspect``
        Several animals spread across the frame produce a very wide union box
        -- one measured case was 560x135 pixels, a 4.2:1 strip. The model
        input is square, so squashing that would distort the animals beyond
        recognition. The short side is grown until the crop is no more
        elongated than this. For reference the full frame after a banner crop
        is about 1.4:1, and squashing *that* was measured to be the best eval
        transform, so 1.6 keeps us in known-good territory.
    ``banner_frac``
        The crop is clamped inside the banner-safe band. A detection near the
        top edge, once padded, would otherwise pull the timestamp/camera-ID
        overlay back into frame and reintroduce the exact shortcut we are
        removing.
    ``strict_containment``
        The banner clamp and "keep every animal in frame" genuinely conflict:
        **32% of union boxes extend into the bottom 5% strip**, because animals
        standing on the ground reach the bottom edge of the frame. Clamping
        those would cut off feet and tails on a third of the dataset. So when
        this is set (the default) the crop is finally unioned back with the raw
        box extent: the banner stays out wherever it can, and where it cannot,
        containment wins. The measured overhang is small -- median 0.023 of
        frame height, never more than ``banner_frac`` -- and after zooming it
        is a thin sliver that rarely contains the overlay text itself. Set it
        False to prioritise banner removal and accept clipped animals.
    ``mask_banner``
        Even with the clamp, 23% of crops end up reaching the rows where the
        overlay text actually sits. Rather than choose between losing animals
        and leaking the camera ID, this paints the banner strips solid black
        **before** cropping. That resolves the conflict outright: the text is
        destroyed, while the geometry the crop depends on is untouched. A
        black bar is itself a weak visual cue, but critically it is the *same*
        black bar at every location, so unlike the text it carries no
        information about which camera took the picture -- which is the only
        property that matters for shortcut learning.
    fallback
        MegaDetector finds nothing in ~20% of images. Those fall back to the
        full frame minus the banner strips -- i.e. the previous behaviour --
        rather than being dropped or given a garbage crop.
    """

    def __init__(
        self,
        boxes_csv,
        conf_threshold: float = 0.2,
        margin: float = 0.15,
        min_frac: float = 0.30,
        max_aspect: float = 1.6,
        banner_frac: float = 0.05,
        strict_containment: bool = True,
        mask_banner: bool = True,
        animal_only: bool = True,
    ):
        self.conf_threshold = float(conf_threshold)
        self.margin = float(margin)
        self.min_frac = float(min_frac)
        self.max_aspect = float(max_aspect)
        self.banner_frac = float(banner_frac)
        self.strict_containment = bool(strict_containment)
        self.mask_banner = bool(mask_banner)
        self.animal_only = bool(animal_only)

        df = pd.read_csv(boxes_csv)
        keep = df["conf"] >= self.conf_threshold
        if animal_only:
            keep &= df["cls"] == MEGADETECTOR_ANIMAL
        df = df[keep]
        # One rectangle per uid spanning every kept box: min of the top-left
        # corners, max of the bottom-right. This is what guarantees that all
        # animals stay in the picture when there is more than one.
        g = df.groupby("uid").agg(x1=("nx1", "min"), y1=("ny1", "min"),
                                  x2=("nx2", "max"), y2=("ny2", "max"))
        self.boxes: dict[str, tuple[float, float, float, float]] = {
            uid: (r.x1, r.y1, r.x2, r.y2) for uid, r in g.iterrows()
        }
        self.n_uids = len(self.boxes)

    def rect(self, uid: str, frame_aspect: float | None = None) -> tuple[float, float, float, float]:
        """Normalized crop rectangle ``(x1, y1, x2, y2)`` for ``uid``.

        ``frame_aspect`` is the source image's width/height. It is needed for
        the ``max_aspect`` guard, because the rectangle is in *normalized*
        coordinates: 0.7 x 0.3 of a 796x448 frame is 557x134 pixels, i.e.
        4.2:1, not 2.3:1. Pass None to skip that guard.
        """
        lo, hi = self.banner_frac, 1.0 - self.banner_frac
        box = self.boxes.get(str(uid))
        if box is None:
            return (0.0, lo, 1.0, hi)  # fallback: full frame minus banners

        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        # pad for context
        x1, x2 = x1 - self.margin * w, x2 + self.margin * w
        y1, y2 = y1 - self.margin * h, y2 + self.margin * h
        # enforce a minimum extent about the centre
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = max((x2 - x1) / 2, self.min_frac / 2)
        half_h = max((y2 - y1) / 2, self.min_frac / 2)
        # keep the crop from being an extreme strip (see max_aspect)
        if frame_aspect:
            px_aspect = (2 * half_w * frame_aspect) / (2 * half_h)
            if px_aspect > self.max_aspect:
                half_h = half_w * frame_aspect / self.max_aspect
            elif px_aspect < 1.0 / self.max_aspect:
                half_w = half_h / (frame_aspect * self.max_aspect)
        x1, x2 = cx - half_w, cx + half_w
        y1, y2 = cy - half_h, cy + half_h
        # slide (don't just clip) back inside the frame so the requested size
        # is preserved when the box sits against an edge
        x1, x2 = _shift_into(x1, x2, 0.0, 1.0)
        y1, y2 = _shift_into(y1, y2, lo, hi)

        if self.strict_containment:
            # Re-admit whatever the banner clamp just cut off. Last step, so it
            # can only ever grow the crop, never shrink it.
            bx1, by1, bx2, by2 = box
            x1, y1 = min(x1, bx1), min(y1, by1)
            x2, y2 = max(x2, bx2), max(y2, by2)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)
        return (x1, y1, x2, y2)

    def __call__(self, img: Image.Image, uid: str) -> Image.Image:
        w, h = img.size
        if self.mask_banner and self.banner_frac > 0:
            img = img.copy()
            cut = round(h * self.banner_frac)
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, w, cut], fill=(0, 0, 0))
            d.rectangle([0, h - cut, w, h], fill=(0, 0, 0))
        x1, y1, x2, y2 = self.rect(uid, frame_aspect=w / h)
        left, right = int(round(x1 * w)), int(round(x2 * w))
        top, bottom = int(round(y1 * h)), int(round(y2 * h))
        # never emit a degenerate crop
        right = max(right, left + 1)
        bottom = max(bottom, top + 1)
        return img.crop((left, top, right, bottom))

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(uids={self.n_uids}, "
                f"conf>={self.conf_threshold}, margin={self.margin}, "
                f"min_frac={self.min_frac}, max_aspect={self.max_aspect}, "
                f"banner_frac={self.banner_frac}, strict={self.strict_containment}, "
                f"mask_banner={self.mask_banner})")


def _shift_into(a: float, b: float, lo: float, hi: float) -> tuple[float, float]:
    """Slide the interval [a, b] inside [lo, hi], shrinking only if it must."""
    if b - a > hi - lo:
        return lo, hi
    if a < lo:
        b += lo - a
        a = lo
    if b > hi:
        a -= b - hi
        b = hi
    return a, b



class IWildCamChallengeDataset(Dataset):
    """One row per image. Returns ``(image_tensor, label_int)`` for train/val,
    ``(image_tensor, uid_str)`` for test_public (which has no labels).

    Exposes:
        self.num_classes : int
        self.uids        : list[str]
        self.labels      : list[int] | None
        self.domains     : list[str] | None        (val only: 'id'/'ood')
    """

    def __init__(self, root, split: str, transform: Optional[Callable] = None,
                 box_cropper: Optional["BoxCropper"] = None):
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        # Applied before `transform`, because it needs the uid and torchvision
        # transforms only ever see the pixels.
        self.box_cropper = box_cropper
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
        if self.box_cropper is not None:
            img = self.box_cropper(img, uid)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img, uid
        return img, int(self.labels[idx])


def default_train_transform(
    img_size: int = IMG_SIZE,
    augment: str = "full",
    crop_frac: float = 0.0,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    rrc_scale_min: float = DINOV3_GLOBAL_SCALE[0],
    rrc_ratio: tuple[float, float] = DINOV3_RRC_RATIO,
    solarize_p: float = 0.1,
) -> Callable:
    """Training augmentation, tuned for camera-trap imagery.

    Three pipelines, selected by ``augment``:

    ``basic``
        The original starter-kit pipeline (squash resize + horizontal flip).
        Kept so the heavier pipelines can be ablated against it under
        otherwise identical hyperparameters.

    ``full``
        The hand-tuned camera-trap pipeline. ``RandomResizedCrop`` forces the
        model to recognise species at many apparent scales instead of
        memorising "big centred animal"; ``ColorJitter`` covers the
        exposure/white-balance differences between cameras; ``RandomGrayscale``
        matters because camera traps switch to infrared at night, so randomly
        dropping colour stops the model leaning on cues that vanish after dark.
        ``scale=(0.6, 1.0)`` is deliberately not the ImageNet default of 0.08 --
        aggressive cropping would often cut the animal out of the frame and
        hand the model a mislabelled image.

    ``dinov3``
        The augmentation DINOv3 itself was pretrained under, so a DINOv3
        backbone is fine-tuned on the same input distribution its features
        were learned on. DINOv3's actual recipe is *multi-crop*: two 224px
        "global" views plus eight 96px "local" views, matched against each
        other by a self-distillation loss. Multi-crop only makes sense when
        there is a second view to match -- with a cross-entropy head there
        isn't -- so the correct single-view adaptation is the global-crop
        branch alone, which is what this builds. The blur and solarize
        probabilities below are the average of DINOv3's two global views
        (blur 1.0/0.1, solarize 0.0/0.2).

    ``crop_frac`` (see ``CropBorders``) is applied *first* in every branch,
    including ``basic``, and must match ``default_eval_transform``.
    """
    pre: list = [CropBorders(crop_frac)] if crop_frac > 0 else []
    post = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if augment == "basic":
        return transforms.Compose(pre + [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
        ] + post)

    if augment == "full":
        return transforms.Compose(pre + [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.15),
        ] + post)

    if augment == "dinov3":
        return transforms.Compose(pre + [
            transforms.RandomResizedCrop(
                img_size,
                scale=(rrc_scale_min, 1.0),
                ratio=tuple(rrc_ratio),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # RandomApply wrapper: DINOv3 applies the *whole* jitter with p=0.8,
            # rather than each channel independently. Not the same thing.
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5
            ),
            # threshold is in 0-255 because this runs on a PIL image, before
            # ToTensor(). Note for camera traps: solarize inverts bright pixels,
            # and a lot of these frames are night-time infrared. If the loss
            # curve looks strange, --solarize-p 0 is the first thing to try.
            transforms.RandomSolarize(threshold=128, p=solarize_p),
        ] + post)

    raise ValueError(f"augment must be 'full', 'basic' or 'dinov3', got {augment!r}")


def default_eval_transform(
    img_size: int = IMG_SIZE,
    crop_frac: float = 0.0,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> Callable:
    """Deterministic eval transform: resize the *whole* frame, no cropping.

    Deliberately not the usual Resize-then-CenterCrop: a centre crop would
    discard the frame edges, and camera-trap animals frequently sit right at
    the edge. Keeping the full field of view costs some aspect-ratio
    distortion but never throws the subject away. (Measured: the squash beats
    Resize+CenterCrop by 5-11 accuracy points on every checkpoint tried.)

    ``crop_frac`` must match whatever the model was *trained* with -- it is
    recorded in the checkpoint hyperparameters and threaded through
    ``eval.load_checkpoint`` for exactly this reason.
    """
    pre: list = [CropBorders(crop_frac)] if crop_frac > 0 else []
    return transforms.Compose(pre + [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
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


def make_box_cropper(data_root, enabled: bool = True, **kwargs) -> Optional[BoxCropper]:
    """Build a ``BoxCropper`` from ``<data_root>/boxes.csv``, or return None.

    Returns None when disabled, so callers can pass the result straight to
    ``IWildCamChallengeDataset(..., box_cropper=...)`` either way. Raises if
    box cropping was asked for but the file is missing -- silently falling
    back to uncropped frames would mean training and evaluating on different
    inputs without any warning.
    """
    if not enabled:
        return None
    path = Path(data_root) / "boxes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"box cropping requested but {path} does not exist. "
            f"Pass --no-box-crop to train on full frames instead."
        )
    return BoxCropper(path, **kwargs)
