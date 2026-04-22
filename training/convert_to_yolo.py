"""
convert_to_yolo.py — Convert OmniVision3D synthetic renders to YOLOv8 format.

Bounding boxes are detected from raw white-background renders (dataset/raw/)
where the background is pure white and foreground pixels are easy to isolate.
The same bbox is applied to all sky-composited variants of the same render.

Raw render naming:   <az>_<el>_<dist>.png
Processed naming:    <az>_<el>_<dist>_sky<N>.png   (N = 0, 1, 2, ...)

Output:
    yolo_dataset/
    ├── images/train|val|test/
    ├── labels/train|val|test/
    ├── dataset.yaml
    └── sample_label_check.png

Usage:
    python training/convert_to_yolo.py
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Bounding-box detection (white-background raw renders only)
# ---------------------------------------------------------------------------

def detect_bbox_white(img_path: Path, bg_thresh: int = 240) -> tuple | None:
    """
    Return normalised YOLO bbox (cx, cy, w, h) from a white-background render.

    Pixels where ALL three channels > bg_thresh are background.
    Everything else is the drone.
    Returns None if no foreground is found (blank/invisible frame).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    bg_mask = np.all(img > bg_thresh, axis=2)
    fg_mask = ~bg_mask

    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)

    if not rows.any():
        return None

    y1 = int(np.argmax(rows))
    y2 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x1 = int(np.argmax(cols))
    x2 = int(len(cols) - 1 - np.argmax(cols[::-1]))

    # Must be at least 0.5% of image in each dimension
    if (x2 - x1) < w * 0.005 or (y2 - y1) < h * 0.005:
        return None

    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, bw)),
        max(0.0, min(1.0, bh)),
    )


def _stem_to_raw_key(stem: str) -> str:
    """
    Strip the trailing _skyN suffix from a processed image stem.
    '0.0_81.9_12.00_sky2' -> '0.0_81.9_12.00'
    """
    parts = stem.rsplit("_sky", 1)
    return parts[0]


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    raw_dir: str   = "dataset/raw/Shahed 136 Drone",
    proc_dir: str  = "dataset/processed/Shahed 136 Drone",
    out_dir: str   = "yolo_dataset",
    bg_thresh: int = 240,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed: int = 42,
) -> None:
    rng      = random.Random(seed)
    out      = Path(out_dir)
    raw_path = Path(raw_dir)
    proc_path = Path(proc_dir)

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels"  / split).mkdir(parents=True, exist_ok=True)

    # ---- Build bbox lookup from raw white-background renders ----
    print("Detecting bboxes from raw white-background renders...")
    raw_images = sorted(raw_path.glob("*.png"))
    bbox_map: dict[str, tuple] = {}
    no_detect = 0
    for raw_img in raw_images:
        bbox = detect_bbox_white(raw_img, bg_thresh)
        if bbox is None:
            no_detect += 1
            continue
        bbox_map[raw_img.stem] = bbox

    print(f"  Raw renders     : {len(raw_images)}")
    print(f"  With valid bbox : {len(bbox_map)}")
    print(f"  Skipped (tiny)  : {no_detect}")

    # ---- Collect processed images that have a matching raw bbox ----
    proc_images = sorted(proc_path.glob("*.png"))
    valid_images = []
    for p in proc_images:
        key = _stem_to_raw_key(p.stem)
        if key in bbox_map:
            valid_images.append(p)

    print(f"\nProcessed images : {len(proc_images)}")
    print(f"Matched to bbox  : {len(valid_images)}")

    # ---- Split ----
    rng.shuffle(valid_images)
    n       = len(valid_images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": valid_images[:n_train],
        "val":   valid_images[n_train : n_train + n_val],
        "test":  valid_images[n_train + n_val :],
    }

    written = {}
    for split, paths in splits.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels"  / split
        count   = 0
        for img_path in paths:
            key   = _stem_to_raw_key(img_path.stem)
            cx, cy, bw, bh = bbox_map[key]

            shutil.copy2(img_path, img_dir / img_path.name)
            lbl_file = lbl_dir / (img_path.stem + ".txt")
            lbl_file.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            count += 1
        written[split] = count

    # ---- dataset.yaml ----
    yaml_data = {
        "path":  str(out.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["dji_mini_4_pro"],
    }
    with open(out / "dataset.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # ---- sample label check ----
    _save_sample_check(out)

    print(f"\nConversion complete")
    for split, count in written.items():
        print(f"  {split:<6} : {count} images + labels")
    print(f"  dataset.yaml -> {out / 'dataset.yaml'}")


def _save_sample_check(out: Path) -> None:
    """Save the first training image with its bbox drawn in red."""
    img_dir = out / "images" / "train"
    lbl_dir = out / "labels"  / "train"
    img_files = sorted(img_dir.iterdir())
    if not img_files:
        return

    img_path = img_files[0]
    lbl_path = lbl_dir / (img_path.stem + ".txt")
    if not lbl_path.exists():
        return

    img = Image.open(img_path)
    w, h = img.size
    cls, cx, cy, bw, bh = map(float, lbl_path.read_text().split())
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    out_path = out / "sample_label_check.png"
    img.save(out_path)
    print(f"\nSample check saved -> {out_path}")
    print(f"  file  : {img_path.name}  ({w}x{h})")
    print(f"  label : class={int(cls)} cx={cx:.3f} cy={cy:.3f} w={bw:.3f} h={bh:.3f}")
    print(f"  pixels: ({x1},{y1}) -> ({x2},{y2})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",    default="dataset/raw/dji-mini-4-pro")
    parser.add_argument("--proc-dir",   default="dataset/processed/DJI_Mini_4_Pro")
    parser.add_argument("--out",        default="yolo_dataset")
    parser.add_argument("--bg-thresh",  type=int, default=240)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    convert(
        raw_dir   = args.raw_dir,
        proc_dir  = args.proc_dir,
        out_dir   = args.out,
        bg_thresh = args.bg_thresh,
        seed      = args.seed,
    )
