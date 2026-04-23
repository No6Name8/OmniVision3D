"""
gps_nav.py — GPS navigation class for OmniVision3D.

Simulation mode only — all commands logged and printed, no hardware touched.

TODO (MAVLink wiring):
    Replace fly_to() stub with MISSION_ITEM or SET_POSITION_TARGET_GLOBAL_INT.
    Replace get_current_position() with serial GPS / dronekit vehicle.location.
"""

import logging
import math
import time
from pathlib import Path
from typing import Tuple

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(_LOG_DIR / "navigation.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)

_EARTH_R = 6_371_000.0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rl1, rlo1 = math.radians(lat1), math.radians(lon1)
    rl2, rlo2 = math.radians(lat2), math.radians(lon2)
    dlat = rl2 - rl1
    dlon = rlo2 - rlo1
    a = math.sin(dlat / 2) ** 2 + math.cos(rl1) * math.cos(rl2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(a))


class GPSNav:
    """
    GPS navigation with home-base tracking and return-to-base support.
    All methods are simulation stubs until MAVLink is wired.
    """

    def __init__(self, home_lat: float, home_lon: float) -> None:
        self.home_lat = home_lat
        self.home_lon = home_lon
        self._current_lat = home_lat
        self._current_lon = home_lon
        self._current_alt = 0.0

    # ------------------------------------------------------------------
    def get_current_position(self) -> Tuple[float, float, float]:
        """Return (lat, lon, alt). TODO: replace with real GPS read."""
        return (self._current_lat, self._current_lon, self._current_alt)

    def fly_to(self, lat: float, lon: float,
               altitude: float = 50.0, simulation: bool = True) -> None:
        """
        Command drone to fly to (lat, lon, altitude).
        Simulation: logs command and updates internal position after 2s.
        """
        msg = f"FLY_TO lat={lat:.6f} lon={lon:.6f} alt={altitude:.1f}m"
        logging.info(msg)
        print(f"  [NAV SIM] {msg}")

        if simulation:
            time.sleep(2.0)
            self._current_lat = lat
            self._current_lon = lon
            self._current_alt = altitude
            logging.info("SIM arrived at lat=%.6f lon=%.6f", lat, lon)
            print(f"  [NAV SIM] Arrived.")
        # TODO: MAVLink MISSION_ITEM command

    def is_close_enough(
        self,
        target: Tuple[float, float],
        threshold: float = 50.0,
    ) -> bool:
        """Return True when within threshold metres of target."""
        current = self.get_current_position()
        dist = _haversine(current[0], current[1], target[0], target[1])
        return dist <= threshold

    def return_to_base(self, simulation: bool = True) -> None:
        """Fly back to home coordinates."""
        msg = f"RETURN_TO_BASE lat={self.home_lat:.6f} lon={self.home_lon:.6f}"
        logging.info(msg)
        print(f"  [NAV SIM] {msg}")
        self.fly_to(self.home_lat, self.home_lon,
                    altitude=self._current_alt, simulation=simulation)

    def hold_position(self) -> None:
        """Command loiter at current position."""
        msg = f"HOLD_POSITION lat={self._current_lat:.6f} lon={self._current_lon:.6f}"
        logging.info(msg)
        print(f"  [NAV SIM] {msg}")
        # TODO: MAVLink MAV_CMD_NAV_LOITER_UNLIM
