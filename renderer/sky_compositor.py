"""
Sky background compositor for OmniVision3D.

Generates 100 synthetic sky backgrounds and composites rendered drone images
onto them to simulate real-world aerial detection conditions.

Each render is paired with 3 randomly selected sky backgrounds, producing
3× the input image count with realistic scene variation.

Usage:
    python -m renderer.sky_compositor \
        --input  dataset/raw/Shahed_136/ \
        --output dataset/processed/Shahed_136/ \
        --backgrounds dataset/backgrounds/
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Sky background generators
# ---------------------------------------------------------------------------

def _vertical_gradient(
    top: tuple, bottom: tuple, size: int = 512
) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.float32)
    for y in range(size):
        t = y / (size - 1)
        img[y] = [top[c] * (1 - t) + bottom[c] * t for c in range(3)]
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_clear_blue(seed: int, size: int = 512) -> np.ndarray:
    """Light-blue-to-white vertical gradient with slight hue variation."""
    rng = np.random.RandomState(seed)
    top    = (rng.randint(100, 145), rng.randint(165, 210), rng.randint(220, 255))
    bottom = (rng.randint(200, 240), rng.randint(220, 245), rng.randint(240, 255))
    img = _vertical_gradient(top, bottom, size)
    noise = rng.randint(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate_overcast(seed: int, size: int = 512) -> np.ndarray:
    """Grey gradient with cloud-like noise and blur."""
    rng = np.random.RandomState(seed)
    base   = rng.randint(155, 200)
    top    = (base,      base,      min(255, base + rng.randint(0, 12)))
    bottom = (min(255, base + 35), min(255, base + 35), min(255, base + 45))
    img = _vertical_gradient(top, bottom, size)
    noise = rng.randint(-22, 23, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return np.array(
        Image.fromarray(img).filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(1.5, 3.2)))
        )
    )


def generate_sunset(seed: int, size: int = 512) -> np.ndarray:
    """Three-band blue-to-orange-to-red gradient."""
    rng = np.random.RandomState(seed)
    top    = (rng.randint(30,  80), rng.randint(20,  60), rng.randint(100, 160))
    mid    = (rng.randint(220, 255), rng.randint(100, 160), rng.randint(20,  60))
    bottom = (rng.randint(200, 240), rng.randint(60,  100), rng.randint(20,  50))
    img = np.zeros((size, size, 3), dtype=np.float32)
    half = size // 2
    for y in range(half):
        t = y / max(half - 1, 1)
        img[y] = [top[c] * (1 - t) + mid[c] * t for c in range(3)]
    for y in range(half, size):
        t = (y - half) / max(size - half - 1, 1)
        img[y] = [mid[c] * (1 - t) + bottom[c] * t for c in range(3)]
    img = np.clip(img, 0, 255).astype(np.uint8)
    noise = rng.randint(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate_night_sky(seed: int, size: int = 512) -> np.ndarray:
    """Dark blue gradient with randomly scattered star dots."""
    rng = np.random.RandomState(seed)
    top    = (rng.randint(0, 15),  rng.randint(0, 20),  rng.randint(20, 50))
    bottom = (rng.randint(5, 25),  rng.randint(5, 30),  rng.randint(30, 70))
    img = _vertical_gradient(top, bottom, size)
    num_stars = rng.randint(250, 550)
    ys = rng.randint(0, size, num_stars)
    xs = rng.randint(0, size, num_stars)
    brightness = rng.randint(140, 256, num_stars)
    for y, x, b in zip(ys, xs, brightness):
        img[y, x] = (b, b, b)
    return img


def generate_hazy_sky(seed: int, size: int = 512) -> np.ndarray:
    """Washed-out pale blue/white gradient with heavy blur."""
    rng = np.random.RandomState(seed)
    top    = (rng.randint(150, 190), rng.randint(170, 210), rng.randint(200, 235))
    bottom = (rng.randint(220, 248), rng.randint(225, 248), rng.randint(232, 252))
    img = _vertical_gradient(top, bottom, size)
    noise = rng.randint(-10, 11, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return np.array(
        Image.fromarray(img).filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(2.5, 5.5)))
        )
    )


# Generator registry: (function, count) pairs — 5 × 20 = 100 backgrounds.
_GENERATORS = [
    (generate_clear_blue,  20),
    (generate_overcast,    20),
    (generate_sunset,      20),
    (generate_night_sky,   20),
    (generate_hazy_sky,    20),
]


def generate_backgrounds(output_dir: str, size: int = 512) -> int:
    """
    Generate all 100 synthetic sky backgrounds and save to `output_dir`.

    Returns the number of files written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    seed = 0
    for gen_fn, count in _GENERATORS:
        sky_type = gen_fn.__name__.replace("generate_", "")
        for i in range(count):
            arr = gen_fn(seed=seed, size=size)
            fname = out / f"{sky_type}_{i:02d}.png"
            Image.fromarray(arr).save(fname)
            written += 1
            seed += 1

    return written


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

def composite_image(
    render: np.ndarray,
    background: np.ndarray,
    bg_threshold: int = 240,
    blur_sigma: float = 0.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Replace white background pixels in `render` with `background`.

    The Open3D renderer uses a pure-white background (255, 255, 255).
    Pixels where ALL channels exceed bg_threshold are treated as background.

    Args:
        render:       H×W×3 uint8 array from the Open3D renderer (white bg).
        background:   H×W×3 uint8 sky image (resized to match if needed).
        bg_threshold: Pixels with ALL channels above this are background (default 240).
        blur_sigma:   Gaussian blur radius applied to the mask edge (0 = none).
        brightness:   PIL brightness factor applied to the final composite.
        contrast:     PIL contrast factor applied to the final composite.

    Returns:
        H×W×3 uint8 composited image.
    """
    h, w = render.shape[:2]
    if background.shape[:2] != (h, w):
        background = np.array(Image.fromarray(background).resize((w, h)))

    # Binary mask: 1 where drone is present (not white), 0 where background (white).
    drone_mask = (render.min(axis=2) < bg_threshold).astype(np.float32)

    # Soft edge: blur the hard mask slightly to avoid pixel-sharp seams.
    if blur_sigma > 0.0:
        mask_pil = Image.fromarray((drone_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        drone_mask = np.array(mask_pil).astype(np.float32) / 255.0

    mask3 = drone_mask[:, :, np.newaxis]
    blended = (render.astype(np.float32) * mask3
               + background.astype(np.float32) * (1.0 - mask3))
    result = np.clip(blended, 0, 255).astype(np.uint8)

    pil = Image.fromarray(result)
    if brightness != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(brightness)
    if contrast != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
    return np.array(pil)


def run_compositor(
    input_dir: str,
    output_dir: str,
    bg_dir: str,
    copies_per_render: int = 3,
    seed: int = 42,
) -> int:
    """
    Composite every render in `input_dir` onto `copies_per_render` random skies.

    Returns the number of images written.
    """
    rng = random.Random(seed)

    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    bg_path  = Path(bg_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    render_files = sorted(in_path.glob("*.png"))
    bg_files     = sorted(bg_path.glob("*.png"))

    if not render_files:
        raise FileNotFoundError(f"No PNG renders found in {input_dir}")
    if not bg_files:
        raise FileNotFoundError(f"No background PNGs found in {bg_dir}")

    written = 0
    for render_path in render_files:
        render = np.array(Image.open(render_path).convert("RGB"))
        chosen_bgs = rng.choices(bg_files, k=copies_per_render)

        for idx, bg_file in enumerate(chosen_bgs):
            bg = np.array(Image.open(bg_file).convert("RGB"))
            blur_sigma  = rng.uniform(0.0, 0.8)
            brightness  = rng.uniform(0.88, 1.12)
            contrast    = rng.uniform(0.92, 1.08)

            composited = composite_image(
                render, bg,
                blur_sigma=blur_sigma,
                brightness=brightness,
                contrast=contrast,
            )
            stem = render_path.stem
            out_file = out_path / f"{stem}_sky{idx}.png"
            Image.fromarray(composited).save(out_file)
            written += 1

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Composite drone renders onto synthetic sky backgrounds"
    )
    parser.add_argument("--input",       required=True,
                        help="Directory containing raw renders (PNG)")
    parser.add_argument("--output",      required=True,
                        help="Directory to write composited images")
    parser.add_argument("--backgrounds", required=True,
                        help="Directory to save/load sky backgrounds")
    parser.add_argument("--copies",      type=int, default=3,
                        help="Sky backgrounds applied per render (default 3)")
    args = parser.parse_args()

    print("Generating sky backgrounds...")
    n_bg = generate_backgrounds(args.backgrounds)
    print(f"  {n_bg} backgrounds saved -> {args.backgrounds}")

    print("Compositing renders...")
    n_out = run_compositor(args.input, args.output, args.backgrounds,
                           copies_per_render=args.copies)
    print(f"  {n_out} composited images saved -> {args.output}")
