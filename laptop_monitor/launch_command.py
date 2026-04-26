"""
launch_command.py — Sends launch / abort commands to the interceptor Pi.

Fire-and-forget UDP. The drone Pi listens on port 5556.
"""

import json
import logging
import socket
import time
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fh = logging.FileHandler(str(_LOG_DIR / "launch_commands.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
_log = logging.getLogger("launch_command")
_log.setLevel(logging.INFO)
_log.addHandler(_fh)
_log.propagate = False


class LaunchCommander:
    def __init__(self, drone_ip: str, port: int = 5556) -> None:
        self._ip   = drone_ip
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[LaunchCommander] Drone Pi: {drone_ip}:{port}")

    # ------------------------------------------------------------------
    def send_launch(self, alert_data: dict) -> None:
        packet = {
            "command":           "LAUNCH",
            "target_confidence": alert_data.get("confidence", 0.0),
            "compass_heading":   alert_data.get("compass_heading"),
            "timestamp":         time.time(),
            "source":            "LAPTOP_MONITOR",
        }
        self._send(packet)
        _log.info("LAUNCH  conf=%.3f heading=%s",
                  packet["target_confidence"], packet["compass_heading"])
        print(f"LAUNCH COMMAND SENT to {self._ip}")

    def send_abort(self) -> None:
        packet = {
            "command":   "ABORT",
            "timestamp": time.time(),
            "source":    "LAPTOP_MONITOR",
        }
        self._send(packet)
        _log.info("ABORT sent")
        print(f"ABORT COMMAND SENT to {self._ip}")

    # ------------------------------------------------------------------
    def _send(self, packet: dict) -> None:
        data = json.dumps(packet).encode()
        try:
            self._sock.sendto(data, (self._ip, self._port))
        except OSError as e:
            print(f"[LaunchCommander] Send error: {e}")
            _log.error("Send error: %s", e)

    def close(self) -> None:
        self._sock.close()
