"""
Thermal camera simulation filter for OmniVision3D.

Converts regular RGB renders into realistic thermal-style images to support
night and low-visibility drone detection scenarios.

Pipeline per image:
  1. Convert to grayscale
  2. CLAHE contrast enhancement
  3. Apply thermal colormap (iron / rainbow / grayscale)
  4. Heat bloom: bright regions emit a soft glow
  5. Gaussian sensor noise
  6. Optional scanline effect

Usage:
    python -m renderer.thermal_filter \
        --input   dataset/processed/Shahed_136/ \
        --output  dataset/processed/Shahed_136_thermal/ \
        --palette iron
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Colormap lookup tables
# ---------------------------------------------------------------------------

def _build_lut(keypoints: list) -> np.ndarray:
    """
    Interpolate a 256×3 uint8 LUT from a list of (value, R, G, B) keypoints.
    Keypoints must be sorted by value and include 0 and 255.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(len(keypoints) - 1):
        v0, r0, g0, b0 = keypoints[i]
        v1, r1, g1, b1 = keypoints[i + 1]
        for v in range(v0, v1 + 1):
            t = (v - v0) / (v1 - v0) if v1 != v0 else 0.0
            lut[v] = (
                int(r0 + t * (r1 - r0)),
                int(g0 + t * (g1 - g0)),
                int(b0 + t * (b1 - b0)),
            )
    return lut


# Iron: black → dark red → red → orange → yellow → white
_LUT_IRON = _build_lut([
    (  0,   0,   0,   0),
    ( 50, 100,   0,   0),
    (100, 200,  30,   0),
    (150, 255, 120,   0),
    (200, 255, 210,  80),
    (230, 255, 240, 170),
    (255, 255, 255, 255),
])

# Rainbow: blue → cyan → green → yellow → red
_LUT_RAINBOW = _build_lut([
    (  0,   0,   0, 255),
    ( 64,   0, 220, 220),
    (128,   0, 210,   0),
    (192, 220, 220,   0),
    (255, 255,   0,   0),
])

# High-contrast grayscale (identity but with hard stretch)
_LUT_GRAYSCALE = _build_lut([
    (  0,   0,   0,   0),
    (255, 255, 255, 255),
])

_LUTS = {
    "iron":      _LUT_IRON,
    "rainbow":   _LUT_RAINBOW,
    "grayscale": _LUT_GRAYSCALE,
}


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def apply_thermal(
    image: np.ndarray,
    palette: str = "iron",
    scanlines: bool = True,
    noise_std: float = 3.0,
) -> np.ndarray:
    """
    Convert an RGB image to a thermal-style RGB image.

    Args:
        image:      H×W×3 uint8 numpy array (RGB).
        palette:    Colormap name: 'iron', 'rainbow', or 'grayscale'.
        scanlines:  If True, darken every 4th row to simulate sensor scanlines.
        noise_std:  Standard deviation of Gaussian sensor noise (0 to disable).

    Returns:
        H×W×3 uint8 numpy array in the chosen thermal colormap.
    """
    if palette not in _LUTS:
        raise ValueError(f"Unknown palette '{palette}'. Choose from: {list(_LUTS)}")

    # Step 1 — grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2 — CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 3 — apply colormap via LUT
    lut = _LUTS[palette]
    colored = lut[gray]  # (H, W, 3)

    # Step 4 — heat bloom: bright regions emit a soft glow
    bright = np.clip((gray.astype(np.float32) - 160) / 95.0, 0.0, 1.0)
    bloom_src = (colored.astype(np.float32) * bright[:, :, np.newaxis])
    bloom = cv2.GaussianBlur(bloom_src, (21, 21), sigmaX=7)
    bloom_mask = cv2.GaussianBlur(bright, (21, 21), sigmaX=7)[:, :, np.newaxis]
    colored = np.clip(
        colored.astype(np.float32) + bloom * bloom_mask * 0.45, 0, 255
    ).astype(np.uint8)

    # Step 5 — Gaussian sensor noise
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, colored.shape)
        colored = np.clip(colored.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Step 6 — scanline effect (every 4th row slightly darker)
    if scanlines:
        colored[::4] = (colored[::4].astype(np.float32) * 0.82).astype(np.uint8)

    return colored


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_thermal_filter(
    input_dir: str,
    output_dir: str,
    palette: str = "iron",
) -> int:
    """
    Apply thermal filter to every PNG in `input_dir` and save to `output_dir`.

    Returns the number of images processed.
    """
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    processed = 0
    for img_path in files:
        arr = np.array(Image.open(img_path).convert("RGB"))
        thermal = apply_thermal(arr, palette=palette)
        Image.fromarray(thermal).save(out_path / img_path.name)
        processed += 1
        if processed % 500 == 0:
            print(f"  {processed}/{len(files)} processed...")

    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply thermal camera filter to synthetic drone renders"
    )
    parser.add_argument("--input",   required=True,
                        help="Directory of RGB PNG images to convert")
    parser.add_argument("--output",  required=True,
                        help="Directory to write thermal images")
    parser.add_argument("--palette", default="iron",
                        choices=list(_LUTS),
                        help="Thermal colormap: iron | rainbow | grayscale")
    args = parser.parse_args()

    print(f"Applying '{args.palette}' thermal filter...")
    n = run_thermal_filter(args.input, args.output, palette=args.palette)
    print(f"Done — {n} images saved -> {args.output}")
