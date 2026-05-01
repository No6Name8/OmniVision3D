"""
sensors/gps.py — Serial GPS reader (NMEA $GPRMC / $GNRMC).

Replaces GPSTest.py.  Writes directly to shared_state instead of gpsdata.txt.
Optionally still writes gpsdata.txt for legacy compatibility.
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


def _parse_nmea_coord(value: str, hemisphere: str) -> Optional[float]:
    """Convert NMEA DDMM.MMMM + N/S/E/W to decimal degrees."""
    if not value:
        return None
    try:
        dot = value.index(".")
        deg = int(value[:dot - 2])
        mins = float(value[dot - 2:])
        decimal = deg + mins / 60.0
        if hemisphere in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def _parse_gprmc(sentence: str) -> Optional[dict]:
    """
    Parse $GPRMC or $GNRMC sentence.
    Returns dict with lat, lon, time_utc or None if invalid.
    """
    parts = sentence.strip().split(",")
    if len(parts) < 7:
        return None
    tag = parts[0].lstrip("$")
    if tag not in ("GPRMC", "GNRMC"):
        return None
    if parts[2] != "A":   # A = valid fix
        return None

    time_str = parts[1][:6]   # HHMMSS
    lat = _parse_nmea_coord(parts[3], parts[4])
    lon = _parse_nmea_coord(parts[5], parts[6])

    if lat is None or lon is None:
        return None

    return {"lat": lat, "lon": lon, "time_utc": time_str}


def thread_gps(
    port: str,
    baud: int = 9600,
    gpsdata_path: Optional[str] = None,
) -> None:
    """
    GPS reader thread.  Call as:
        t = threading.Thread(target=thread_gps, args=(port,), daemon=True)
        t.start()
    """
    if not HAS_SERIAL:
        print("[GPS] pyserial not installed — GPS disabled")
        return

    while True:
        try:
            with serial.Serial(
                port=port,
                baudrate=baud,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1,
            ) as ser:
                print(f"[GPS] opened {port}")
                while True:
                    try:
                        raw = ser.readline()
                        line = raw.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue

                    parsed = _parse_gprmc(line)
                    if parsed is None:
                        continue

                    with ss.lock():
                        ss.state.lat      = parsed["lat"]
                        ss.state.lon      = parsed["lon"]
                        ss.state.time_utc = parsed["time_utc"]

                    # Optional legacy file write
                    if gpsdata_path:
                        try:
                            with open(gpsdata_path, "w") as f:
                                # Format: h,m,s,A,lat_int,lat_frac,lon_int,lon_frac
                                t = parsed["time_utc"]
                                lat = parsed["lat"]
                                lon = parsed["lon"]
                                lat_d = int(abs(lat))
                                lat_m = (abs(lat) - lat_d) * 60
                                lon_d = int(abs(lon))
                                lon_m = (abs(lon) - lon_d) * 60
                                hemi_lat = "N" if lat >= 0 else "S"
                                f.write(
                                    f"{t[0:2]},{t[2:4]},{t[4:6]},{hemi_lat},"
                                    f"{lat_d:02d},{lat_m:.4f},"
                                    f"{lon_d:03d},{lon_m:.4f}\n"
                                )
                        except Exception:
                            pass

        except Exception as e:
            print(f"[GPS] error: {e} - retrying in 3s")
            time.sleep(3)
