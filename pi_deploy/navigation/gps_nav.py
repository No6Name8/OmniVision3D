"""
gps_nav.py — GPS navigation stubs for OmniVision3D.

Simulation mode only: all commands are printed and logged.
No hardware is accessed.

TODO (when GPS/compass are wired):
    - Replace fly_to() stub with MAVLink MISSION_ITEM or
      SET_POSITION_TARGET_GLOBAL_INT commands.
    - Replace get_current_position() stub with real GPS poll
      (serial read from u-blox or dronekit vehicle.location).
    - Add heading hold via compass bearing calculation.
    - Add altitude hold / barometer integration.
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

# Earth radius in metres (WGS-84 mean)
_EARTH_R = 6_371_000.0


def haversine(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Return great-circle distance in metres between two WGS-84 coordinates."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(a))


def get_current_position() -> Tuple[float, float, float]:
    """
    Return (lat, lon, alt_metres) of the drone's current position.

    Stub returns a fixed position for simulation.
    TODO: replace with real GPS read.
    """
    # TODO: read from serial GPS / dronekit vehicle.location.global_frame
    return (24.7136, 46.6753, 50.0)


def fly_to(lat: float, lon: float, altitude: float = 50.0) -> None:
    """
    Command the drone to fly to (lat, lon) at the given altitude.

    Simulation mode: logs and prints the command then returns immediately.
    TODO: replace with MAVLink waypoint or velocity setpoint command.
    """
    msg = f"FLY_TO  lat={lat:.6f}  lon={lon:.6f}  alt={altitude:.1f}m"
    logging.info(msg)
    print(f"  [NAV SIM] {msg}")

    # TODO: send via MAVLink:
    # master.mav.mission_item_send(
    #     master.target_system, master.target_component, 0,
    #     mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    #     mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
    #     2, 1, 0, 0, 0, 0, lat, lon, altitude)


def is_close_enough(
    current: Tuple[float, float, float],
    target: Tuple[float, float],
    threshold: float = 50.0,
) -> bool:
    """
    Return True when the drone is within `threshold` metres of the target.

    Args:
        current:   (lat, lon, alt) from get_current_position().
        target:    (lat, lon) destination.
        threshold: Acceptance radius in metres (default 50 m).
    """
    dist = haversine(current[0], current[1], target[0], target[1])
    return dist <= threshold


def navigate_to(
    lat: float,
    lon: float,
    altitude: float = 50.0,
    poll_interval: float = 1.0,
    threshold: float = 50.0,
    simulation: bool = True,
) -> None:
    """
    Issue a fly_to command and block until is_close_enough() is satisfied.

    In simulation mode the drone is always "close enough" after 2 seconds
    so the mission loop can proceed without real movement.
    """
    fly_to(lat, lon, altitude)
    target = (lat, lon)

    if simulation:
        print("  [NAV SIM] Simulating transit (2s)...")
        time.sleep(2.0)
        print("  [NAV SIM] Arrived at target coordinates.")
        logging.info("SIM arrival at lat=%.6f lon=%.6f", lat, lon)
        return

    # Real mode: poll GPS until within threshold
    while True:
        current = get_current_position()
        dist    = haversine(current[0], current[1], target[0], target[1])
        logging.info("En-route  dist=%.1fm", dist)
        if is_close_enough(current, target, threshold):
            logging.info("Arrived  dist=%.1fm", dist)
            return
        time.sleep(poll_interval)
