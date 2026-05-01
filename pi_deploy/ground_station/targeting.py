"""
targeting.py — Converts own GPS + compass bearing + laser distance
               into target GPS coordinates.

Also handles compass offset (user calibration) persistent in a file.
"""

import math
import os
from pathlib import Path
from typing import Tuple, Optional

from ground_station import shared_state as ss

_EARTH_R = 6_371_000.0
_OFFSET_PATH = Path("/home/bravofox/compass_offset.txt")


# ── Offset file helpers ────────────────────────────────────────────────────

def read_offset() -> float:
    try:
        return float(_OFFSET_PATH.read_text().strip())
    except Exception:
        return 0.0


def write_offset(deg: float) -> None:
    deg = _wrap_180(deg)
    try:
        _OFFSET_PATH.write_text(f"{deg:.6f}")
    except Exception as e:
        print(f"[targeting] offset write error: {e}")


def _wrap_180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


# ── Haversine destination point ────────────────────────────────────────────

def destination_point(
    lat1_deg: float,
    lon1_deg: float,
    bearing_deg: float,
    distance_m: float,
) -> Tuple[float, float]:
    """
    Given own position, bearing, and distance → target (lat, lon).
    Uses spherical Earth model (accurate to <0.1% at drone intercept ranges).
    """
    phi1   = math.radians(lat1_deg)
    lam1   = math.radians(lon1_deg)
    theta  = math.radians(bearing_deg)
    delta  = distance_m / _EARTH_R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )

    return (
        math.degrees(phi2),
        (math.degrees(lam2) + 540.0) % 360.0 - 180.0,
    )


# ── Update loop ───────────────────────────────────────────────────────────

def update_target(declination_deg: float = 0.0) -> bool:
    """
    Read shared_state, compute target coords, write back.
    Returns True when target was updated.
    Call this from any thread after receiving new sensor data.
    """
    with ss.lock():
        lat  = ss.state.lat
        lon  = ss.state.lon
        raw  = ss.state.compass_raw
        dist = ss.state.distance_m

    if any(v is None for v in (lat, lon, raw, dist)):
        return False

    offset   = read_offset()
    c_used   = (raw + declination_deg + offset) % 360.0
    tlat, tlon = destination_point(lat, lon, c_used, dist)

    with ss.lock():
        ss.state.compass_used = c_used
        ss.state.target_lat   = tlat
        ss.state.target_lon   = tlon

    return True
