"""Render before/after box-crop previews so the cropping can be eyeballed.

    python -m student.inspect_crops --data-root <root> --out-dir crop_preview

Each PNG shows the original frame with every MegaDetector box drawn (green =
kept, grey = below the confidence threshold, dashed red = the final crop
rectangle after margin / min-size / banner clamping) beside the actual cropped
image the model will be fed.

The sample deliberately covers the awkward cases, not just the easy ones:
images with several animals, very small detections, boxes touching an edge,
and images where MegaDetector found nothing at all (the fallback path).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from student.data import BoxCropper, IWildCamChallengeDataset, MEGADETECTOR_ANIMAL

PANEL_GAP = 12


def _draw_dashed(draw: ImageDraw.ImageDraw, box, color, width=3, dash=12):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x1, y2, x2, y2),
                             (x1, y1, x1, y2), (x2, y1, x2, y2)):
        length = max(abs(bx - ax), abs(by - ay))
        if length == 0:
            continue
        steps = max(1, int(length // dash))
        for i in range(0, steps, 2):
            t0, t1 = i / steps, min(1.0, (i + 1) / steps)
            draw.line([ax + (bx - ax) * t0, ay + (by - ay) * t0,
                       ax + (bx - ax) * t1, ay + (by - ay) * t1],
                      fill=color, width=width)


def render(uid: str, img: Image.Image, rows: pd.DataFrame, cropper: BoxCropper,
           out_path: Path) -> None:
    w, h = img.size
    left = img.copy()
    if cropper.mask_banner and cropper.banner_frac > 0:
        cut = round(h * cropper.banner_frac)
        ImageDraw.Draw(left).rectangle([0, 0, w, cut], fill=(0, 0, 0))
        ImageDraw.Draw(left).rectangle([0, h - cut, w, h], fill=(0, 0, 0))
    d = ImageDraw.Draw(left)

    # Every raw detection: green if it is one of the boxes we actually used.
    for r in rows.itertuples():
        kept = (r.conf >= cropper.conf_threshold and
                (not cropper.animal_only or r.cls == MEGADETECTOR_ANIMAL))
        color = (60, 220, 90) if kept else (150, 150, 150)
        d.rectangle([r.nx1 * w, r.ny1 * h, r.nx2 * w, r.ny2 * h], outline=color, width=3)
        d.text((r.nx1 * w + 4, r.ny1 * h + 3), f"{r.cls_name} {r.conf:.2f}", fill=color)

    # The banner-safe band we clamp into, and the final crop.
    for y in (cropper.banner_frac * h, (1 - cropper.banner_frac) * h):
        d.line([0, y, w, y], fill=(90, 140, 255), width=2)
    x1, y1, x2, y2 = cropper.rect(uid, frame_aspect=w / h)
    _draw_dashed(d, (x1 * w, y1 * h, x2 * w, y2 * h), (255, 70, 70))

    right = cropper(img, uid)
    canvas = Image.new("RGB", (left.width + PANEL_GAP + right.width,
                               max(left.height, right.height)), (25, 25, 25))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + PANEL_GAP, 0))
    n_kept = int(((rows.conf >= cropper.conf_threshold) &
                  (rows.cls == MEGADETECTOR_ANIMAL)).sum()) if len(rows) else 0
    ImageDraw.Draw(canvas).text(
        (6, 6),
        f"{uid}  |  {n_kept} animal box(es) kept  |  "
        f"{left.width}x{left.height} -> {right.width}x{right.height}"
        + ("  [FALLBACK: no detections]" if n_kept == 0 else ""),
        fill=(255, 255, 0),
    )
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview MegaDetector box crops.")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--per-case", type=int, default=4,
                    help="Images to render per interesting case.")
    ap.add_argument("--conf-threshold", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--min-frac", type=float, default=0.30)
    ap.add_argument("--max-aspect", type=float, default=1.6)
    ap.add_argument("--banner-frac", type=float, default=0.05)
    ap.add_argument("--no-mask-banner", action="store_true")
    ap.add_argument("--no-strict-containment", action="store_true")
    args = ap.parse_args()

    cropper = BoxCropper(args.data_root / "boxes.csv", conf_threshold=args.conf_threshold,
                         margin=args.margin, min_frac=args.min_frac,
                         max_aspect=args.max_aspect, banner_frac=args.banner_frac,
                         mask_banner=not args.no_mask_banner,
                         strict_containment=not args.no_strict_containment)
    print(cropper)

    boxes = pd.read_csv(args.data_root / "boxes.csv")
    boxes = boxes[boxes.split == args.split]
    ds = IWildCamChallengeDataset(args.data_root, args.split, None)
    by_uid = {u: g for u, g in boxes.groupby("uid")}

    kept = boxes[(boxes.conf >= args.conf_threshold) & (boxes.cls == MEGADETECTOR_ANIMAL)]
    per_uid = kept.groupby("uid")
    counts = per_uid.size()
    span = per_uid.agg(x1=("nx1", "min"), x2=("nx2", "max"))
    width = (span.x2 - span.x1)

    cases = {
        "single":      list(counts[counts == 1].index[:args.per_case]),
        "multi":       list(counts[counts >= 3].index[:args.per_case]),
        "tiny":        list(width.nsmallest(args.per_case).index),
        "large":       list(width.nlargest(args.per_case).index),
        "edge":        list(span[(span.x1 <= 0.01) | (span.x2 >= 0.99)].index[:args.per_case]),
        "no_detection": [u for u in ds.uids if u not in cropper.boxes][:args.per_case],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for case, uids in cases.items():
        for i, uid in enumerate(uids):
            img = Image.open(ds._find_image(uid)).convert("RGB")
            render(uid, img, by_uid.get(uid, boxes.iloc[:0]), cropper,
                   args.out_dir / f"{case}_{i}_{uid[:8]}.png")
            n += 1
        print(f"  {case:<13} {len(uids)} images")
    print(f"wrote {n} previews to {args.out_dir}")


if __name__ == "__main__":
    main()
