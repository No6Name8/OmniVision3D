"""
shared_state.py — Single source of truth for all threads.

GPS reader, laser reader, compass reader, and YOLO detection
all write here.  The UI reads from here.  No files needed mid-flight.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class State:
    # ── GPS (own position) ────────────────────────────────
    lat:      Optional[float] = None
    lon:      Optional[float] = None
    time_utc: str             = ""

    # ── Sensors ───────────────────────────────────────────
    compass_raw:  Optional[float] = None   # raw degrees from serial
    compass_used: Optional[float] = None   # after declination + user offset
    distance_m:   Optional[float] = None   # laser range in metres

    # ── Computed target position ──────────────────────────
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None

    # ── AI detection ──────────────────────────────────────
    phase:       str   = "SCANNING"
    confidence:  float = 0.0
    consecutive: int   = 0
    bbox:        Optional[Tuple[int, int, int, int]] = None
    det_center:  Optional[Tuple[int, int]]           = None

    # ── Latest annotated frame (BGR numpy) ────────────────
    frame: object = None   # np.ndarray — written by detection thread


state = State()
_lock = threading.Lock()


def lock() -> threading.Lock:
    return _lock
