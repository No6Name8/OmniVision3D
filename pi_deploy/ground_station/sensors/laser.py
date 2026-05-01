"""
sensors/laser.py — Binary laser rangefinder reader.

Protocol extracted from laser777-new.py.
Frame format:  0xAE 0xA7 <payload> 0xBC 0xBE
Distance = (payload[5] << 8 | payload[6]) decimetres.
"""

import threading
import time
from typing import Optional

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

from ground_station import shared_state as ss

_HEADER = bytes([0xAE, 0xA7])
_TAIL   = bytes([0xBC, 0xBE])
_CMD    = bytes([0xAE, 0xA7, 0x04, 0x00, 0x05, 0x09, 0xBC, 0xBE])


def _read_frame(ser, timeout: float = 0.03) -> Optional[bytes]:
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = ser.read(256)
        if data:
            buf.extend(data)
            i = buf.find(_HEADER)
            if i >= 0:
                j = buf.find(_TAIL, i + 2)
                if j >= 0:
                    frame = bytes(buf[i : j + 2])
                    del buf[: j + 2]
                    return frame
        else:
            time.sleep(0.001)
    return None


def _parse_distance(frame: bytes, scale: float) -> Optional[float]:
    if not (frame and frame.startswith(_HEADER) and frame.endswith(_TAIL)):
        return None
    payload = frame[2:-2]
    if len(payload) < 7:
        return None
    dm = (payload[5] << 8) | payload[6]
    if dm <= 0 or dm > 50000:
        return None
    return (dm / 10.0) * scale


def thread_laser(
    port: str,
    baud: int = 9600,
    period: float = 0.02,
    scale: float = 1.0,
) -> None:
    """
    Laser reader thread.  Fires CMD at `period` intervals,
    parses distance, writes to shared_state.distance_m.
    """
    if not HAS_SERIAL:
        print("[Laser] pyserial not installed — laser disabled")
        return

    while True:
        try:
            with serial.Serial(port, baud, timeout=0.02) as ser:
                print(f"[Laser] opened {port}")
                while True:
                    ser.reset_input_buffer()
                    ser.write(_CMD)
                    frame = _read_frame(ser)
                    d = _parse_distance(frame, scale) if frame else None
                    if d is not None:
                        with ss.lock():
                            ss.state.distance_m = d
                    if period > 0:
                        time.sleep(period)
        except Exception as e:
            print(f"[Laser] error: {e} — retrying in 3s")
            time.sleep(3)
