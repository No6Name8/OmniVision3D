"""
Unit tests for the renderer module.

Validates that the camera utility functions produce correctly shaped outputs
and that the viewpoint grid covers the expected number of positions.
All tests use the Open3D + Trimesh backend.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from renderer.camera_utils import get_viewpoint_grid, viewpoint_to_camera
from renderer.sky_compositor import composite_image, generate_backgrounds, run_compositor
from renderer.thermal_filter import apply_thermal


# ---------------------------------------------------------------------------
# get_viewpoint_grid
# ---------------------------------------------------------------------------

def test_viewpoint_grid_count():
    """Grid length must equal azimuth_steps x len(elevations) x len(distances)."""
    grid = get_viewpoint_grid(
        azimuth_steps=36,
        elevation_levels=[-30.0, 0.0, 30.0, 60.0, 90.0],
        distances=[1.5, 2.5, 4.0],
    )
    assert len(grid) == 36 * 5 * 3  # 540


def test_viewpoint_grid_azimuth_range():
    """Every azimuth value must lie within [0, 360)."""
    grid = get_viewpoint_grid(
        azimuth_steps=12,
        elevation_levels=[0.0],
        distances=[2.0],
    )
    for az, _, _ in grid:
        assert 0.0 <= az < 360.0


def test_viewpoint_grid_single_step():
    """A single azimuth step should produce exactly one tuple with azimuth == 0.0."""
    grid = get_viewpoint_grid(
        azimuth_steps=1,
        elevation_levels=[0.0],
        distances=[1.0],
    )
    assert len(grid) == 1
    assert grid[0][0] == pytest.approx(0.0)


def test_viewpoint_grid_returns_floats():
    """All values in every tuple must be plain Python floats."""
    grid = get_viewpoint_grid(
        azimuth_steps=4,
        elevation_levels=[0.0, 45.0],
        distances=[2.0],
    )
    for az, el, dist in grid:
        assert isinstance(az, float)
        assert isinstance(el, float)
        assert isinstance(dist, float)


# ---------------------------------------------------------------------------
# viewpoint_to_camera
# ---------------------------------------------------------------------------

def test_viewpoint_to_camera_returns_three_arrays():
    """Must return a (eye, center, up) tuple of three numpy arrays."""
    result = viewpoint_to_camera(azimuth=45.0, elevation=30.0, distance=2.0)
    assert len(result) == 3
    eye, center, up = result
    for arr in (eye, center, up):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)


def test_viewpoint_to_camera_center_is_origin():
    """The look-at center must always be the world origin."""
    _, center, _ = viewpoint_to_camera(azimuth=90.0, elevation=45.0, distance=3.0)
    np.testing.assert_array_equal(center, np.zeros(3))


def test_viewpoint_to_camera_eye_distance():
    """The eye position must be at the requested radial distance from the origin."""
    for dist in [1.5, 2.5, 4.0]:
        eye, _, _ = viewpoint_to_camera(azimuth=0.0, elevation=0.0, distance=dist)
        assert np.linalg.norm(eye) == pytest.approx(dist, rel=1e-5)


def test_viewpoint_to_camera_up_is_unit():
    """The up vector must be a unit vector."""
    _, _, up = viewpoint_to_camera(azimuth=30.0, elevation=20.0, distance=2.0)
    assert np.linalg.norm(up) == pytest.approx(1.0, rel=1e-5)


def test_viewpoint_to_camera_pole_fallback():
    """At elevation=90 degrees (top-down) the up vector must not be degenerate."""
    eye, center, up = viewpoint_to_camera(azimuth=0.0, elevation=90.0, distance=2.0)
    assert abs(eye[1]) == pytest.approx(2.0, rel=1e-4)
    assert np.linalg.norm(up) == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# sky_compositor
# ---------------------------------------------------------------------------

def _make_render(size: int = 64) -> np.ndarray:
    """White-background render with a grey square drone in the centre."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    q = size // 4
    img[q:size - q, q:size - q] = 120  # grey object
    return img


def _make_sky(size: int = 64) -> np.ndarray:
    """Solid blue sky background."""
    img = np.full((size, size, 3), fill_value=(100, 160, 220), dtype=np.uint8)
    return img


def test_composite_replaces_background():
    """Composited image must differ from the original black-bg render."""
    render = _make_render()
    sky    = _make_sky()
    result = composite_image(render, sky)
    assert not np.array_equal(result, render), \
        "composite_image returned an unchanged render"


def test_compositor_output_count():
    """run_compositor must write exactly input_count x copies_per_render images."""
    with tempfile.TemporaryDirectory() as tmp:
        in_dir  = Path(tmp) / "renders"
        out_dir = Path(tmp) / "composited"
        bg_dir  = Path(tmp) / "backgrounds"
        in_dir.mkdir(); out_dir.mkdir(); bg_dir.mkdir()

        # Write 4 synthetic renders and 2 backgrounds.
        for i in range(4):
            Image.fromarray(_make_render()).save(in_dir / f"render_{i}.png")
        for i in range(2):
            Image.fromarray(_make_sky()).save(bg_dir / f"sky_{i}.png")

        copies = 3
        n = run_compositor(str(in_dir), str(out_dir), str(bg_dir),
                           copies_per_render=copies)
        assert n == 4 * copies
        assert len(list(out_dir.glob("*.png"))) == 4 * copies


def test_compositor_output_dimensions():
    """Every composited image must be 512x512 when given 512x512 inputs."""
    with tempfile.TemporaryDirectory() as tmp:
        in_dir  = Path(tmp) / "renders"
        out_dir = Path(tmp) / "composited"
        bg_dir  = Path(tmp) / "backgrounds"
        in_dir.mkdir(); out_dir.mkdir(); bg_dir.mkdir()

        size = 512
        render = np.zeros((size, size, 3), dtype=np.uint8)
        render[128:384, 128:384] = 100
        sky = np.full((size, size, 3), 180, dtype=np.uint8)
        Image.fromarray(render).save(in_dir / "render_0.png")
        Image.fromarray(sky).save(bg_dir / "sky_0.png")

        run_compositor(str(in_dir), str(out_dir), str(bg_dir), copies_per_render=1)
        out_files = list(out_dir.glob("*.png"))
        assert len(out_files) == 1
        with Image.open(out_files[0]) as img:
            size_out = img.size
        assert size_out == (size, size)


# ---------------------------------------------------------------------------
# thermal_filter
# ---------------------------------------------------------------------------

def _make_rgb(size: int = 64) -> np.ndarray:
    """Synthetic RGB image: black background with a bright grey square."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    q = size // 4
    img[q:size - q, q:size - q] = 160
    return img


def test_thermal_output_dimensions():
    """Thermal output must have the same H×W×3 shape as the input."""
    rgb = _make_rgb(128)
    thermal = apply_thermal(rgb, palette="iron")
    assert thermal.shape == rgb.shape


def test_thermal_output_differs_from_input():
    """Thermal filter must change pixel values — output must not equal input."""
    rgb = _make_rgb(64)
    thermal = apply_thermal(rgb, palette="iron")
    assert not np.array_equal(thermal, rgb)


def test_thermal_all_palettes_run():
    """All three palette modes must complete without error and return uint8 arrays."""
    rgb = _make_rgb(64)
    for palette in ("iron", "rainbow", "grayscale"):
        result = apply_thermal(rgb, palette=palette)
        assert result.dtype == np.uint8, f"palette '{palette}' returned wrong dtype"
        assert result.shape == rgb.shape, f"palette '{palette}' changed image shape"
