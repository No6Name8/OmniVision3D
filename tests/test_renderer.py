"""
Unit tests for the renderer module.

Validates that the camera utility functions produce correctly shaped outputs
and that the viewpoint grid covers the expected number of positions.
All tests use the Open3D + Trimesh backend (no PyTorch3D dependency).
"""

import math

import numpy as np
import pytest

from renderer.camera_utils import get_viewpoint_grid, viewpoint_to_camera


# ---------------------------------------------------------------------------
# get_viewpoint_grid
# ---------------------------------------------------------------------------

def test_viewpoint_grid_count():
    """Grid length must equal azimuth_steps × len(elevations) × len(distances)."""
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
    """At elevation=90° (top-down) the up vector must not be degenerate."""
    eye, center, up = viewpoint_to_camera(azimuth=0.0, elevation=90.0, distance=2.0)
    # Eye should be directly above the origin.
    assert abs(eye[1]) == pytest.approx(2.0, rel=1e-4)
    # Up must still be a unit vector.
    assert np.linalg.norm(up) == pytest.approx(1.0, rel=1e-5)
