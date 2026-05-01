"""
sensors/compass.py — Text-protocol compass reader.

Protocol from laser777-new.py:  $C 123.4 A ... *HEX_CHECKSUM
Extracts the C field (heading degrees) and writes to shared_state.compass_raw.
"""

import re
import threading
import time
from typing import Optional

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

from ground_station import shared_state as ss

_STAR_RE  = re.compile(r'\$(?P<body>[^*]+)\*(?P<cs>[0-9A-Fa-f]+)')
_FIELD_RE = re.compile(r'([A-Z])\s*(-?\d+(?:\.\d+)?)')
_NUM_RE   = re.compile(r'-?\d+\.?\d*')


def _parse_compass_line(line: str) -> Optional[float]:
    """
    Try to extract a heading in degrees from a compass NMEA-like sentence.
    Primary:  $C 123.4 ... *XX  → C field value
    Fallback: first number in the line
    """
    m = _STAR_RE.search(line)
    if m:
        body = m.group("body")
        for f in _FIELD_RE.finditer(body):
            if f.group(1) == "C":
                return float(f.group(2))

    # Fallback: pull the first numeric value from the raw line
    nums = _NUM_RE.findall(line)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            pass
    return None


def thread_compass(port: str, baud: int = 9600) -> None:
    """
    Compass reader thread.  Writes bearing_deg to shared_state.compass_raw.
    """
    if not HAS_SERIAL:
        print("[Compass] pyserial not installed — compass disabled")
        return

    while True:
        try:
            with serial.Serial(port, baud, timeout=0.05) as ser:
                print(f"[Compass] opened {port}")
                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    heading = _parse_compass_line(line)
                    if heading is not None:
                        with ss.lock():
                            ss.state.compass_raw = heading
        except Exception as e:
            print(f"[Compass] error: {e} - retrying in 3s")
            time.sleep(3)
