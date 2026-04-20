"""
Camera utility functions for generating viewpoint parameters used in synthetic rendering.

Two sampling strategies are provided:
  - get_viewpoint_grid:      classic azimuth × elevation grid (used by tests).
  - get_sphere_viewpoints:   Fibonacci sphere sampling — places N points evenly
                             across the full sphere surface, covering top, bottom,
                             sides, diagonals, and every angle in between with
                             no clustering at the poles.
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
    Return a flat list of (azimuth, elevation, distance) tuples on a regular grid.

    Azimuths are evenly spaced over [0, 360) with `azimuth_steps` steps.
    Every combination of elevation and distance is paired with every azimuth.
    """
    azimuths = (np.linspace(0.0, 360.0, azimuth_steps, endpoint=False)).tolist()
    return [
        (float(az), float(el), float(dist))
        for dist in distances
        for el in elevation_levels
        for az in azimuths
    ]


def fibonacci_sphere(n: int) -> List[Tuple[float, float]]:
    """
    Return n (azimuth, elevation) pairs evenly distributed across the full sphere.

    Uses the Fibonacci / golden-angle spiral method which spaces points uniformly
    without clustering at the poles. Covers front, rear, left, right, top-down,
    bottom-up, and every diagonal angle in between.

    Args:
        n: Number of viewpoints to generate.

    Returns:
        List of (azimuth_deg, elevation_deg) tuples.
        azimuth  in [0, 360), elevation in [-90, 90].
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    points = []
    for i in range(n):
        # elevation: uniform distribution from -90 (bottom) to +90 (top)
        elevation = math.degrees(math.asin(1.0 - 2.0 * (i + 0.5) / n))
        # azimuth: golden-angle steps ensure no two points share a longitude band
        azimuth = (360.0 * i / golden) % 360.0
        points.append((float(azimuth), float(elevation)))
    return points


def get_sphere_viewpoints(
    n_angles: int,
    distances: List[float],
) -> List[Tuple[float, float, float]]:
    """
    Return (azimuth, elevation, distance) for every sphere angle × every distance.

    For each of the n_angles evenly distributed viewpoints, all distances are
    included. Total tuples = n_angles × len(distances).

    Args:
        n_angles:  Number of sphere sample points (e.g. 100).
        distances: List of camera distances (e.g. [2.0, 3.5, 5.5, 8.0, 12.0, 18.0]).

    Returns:
        List of (azimuth_deg, elevation_deg, distance) tuples.
    """
    angles = fibonacci_sphere(n_angles)
    return [
        (az, el, float(dist))
        for az, el in angles
        for dist in distances
    ]


def viewpoint_to_camera(
    azimuth: float,
    elevation: float,
    distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical viewpoint parameters to Open3D-compatible camera vectors.

    The camera is placed on a sphere of radius `distance` around the world origin.

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

    # At the poles cos(elevation) ≈ 0 the standard up vector becomes degenerate.
    if abs(math.cos(el_rad)) < 1e-6:
        up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    return eye, center, up
