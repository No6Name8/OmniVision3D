"""
Camera utility functions for generating viewpoint parameters used in synthetic rendering.

Provides helpers to sample azimuth angles, elevation levels, and distances over a
full spherical grid, and to convert those spherical coordinates into the camera
eye-position and look-at vectors required by Open3D's offscreen renderer.
"""

import math
from typing import List, Tuple

import numpy as np


def get_viewpoint_grid(
    azimuth_steps: int,
    elevation_levels: List[float],
    distances: List[float],
) -> List[Tuple[float, float, float]]:
    """
    Return a flat list of (azimuth, elevation, distance) tuples covering the full grid.

    Azimuths are evenly spaced over [0, 360) with `azimuth_steps` steps.
    Every combination of elevation and distance is paired with every azimuth.

    Args:
        azimuth_steps:    Number of evenly-spaced azimuth samples in [0, 360).
        elevation_levels: List of elevation angles in degrees.
        distances:        List of camera-to-origin distances in scene units.

    Returns:
        List of (azimuth_deg, elevation_deg, distance) tuples.
    """
    azimuths = (np.linspace(0.0, 360.0, azimuth_steps, endpoint=False)).tolist()
    return [
        (float(az), float(el), float(dist))
        for dist in distances
        for el in elevation_levels
        for az in azimuths
    ]


def viewpoint_to_camera(
    azimuth: float,
    elevation: float,
    distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical viewpoint parameters to Open3D-compatible camera vectors.

    The camera is placed on a sphere of radius `distance` around the world origin.
    The returned vectors are suitable for use with Open3D's
    ``render.scene.camera.look_at(center, eye, up)``.

    Args:
        azimuth:   Azimuth angle in degrees, measured from the +X axis toward +Z.
        elevation: Elevation angle in degrees above the XZ plane.
        distance:  Radial distance from the world origin.

    Returns:
        eye:    Camera position as a (3,) numpy array.
        center: Look-at target — always the world origin (0, 0, 0).
        up:     Up vector as a (3,) numpy array (world +Y, or +Z at poles).
    """
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    x = distance * math.cos(el_rad) * math.cos(az_rad)
    y = distance * math.sin(el_rad)
    z = distance * math.cos(el_rad) * math.sin(az_rad)
    eye = np.array([x, y, z], dtype=np.float64)

    center = np.zeros(3, dtype=np.float64)

    # At the poles cos(elevation) ≈ 0, so the standard up vector becomes
    # degenerate. Fall back to world -Z when looking straight down.
    if abs(math.cos(el_rad)) < 1e-6:
        up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    return eye, center, up
