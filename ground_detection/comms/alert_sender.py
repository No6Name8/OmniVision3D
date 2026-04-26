"""
alert_sender.py — UDP alert sender for the ground detection unit.

Sends JSON packets to the laptop's listener on port 5555.
No connection state; each call is a fire-and-forget UDP send.
"""

import json
import logging
import socket
import time
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(_LOG_DIR / "alerts_sent.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
_log = logging.getLogger("alert_sender")


class AlertSender:
    def __init__(self, laptop_ip: str, port: int = 5555) -> None:
        self._ip   = laptop_ip
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[AlertSender] Target: {laptop_ip}:{port}")

    # ------------------------------------------------------------------
    def send_alert(self, detection_result, compass_heading: float = None) -> None:
        packet = {
            "type":            "DRONE_DETECTED",
            "confidence":      round(float(detection_result.confidence), 4),
            "consecutive":     int(detection_result.consecutive),
            "compass_heading": compass_heading,
            "timestamp":       time.time(),
            "source":          "GROUND_UNIT_001",
        }
        self._send(packet)
        _log.info("ALERT  conf=%.3f consec=%d heading=%s",
                  packet["confidence"], packet["consecutive"], compass_heading)
        print(f"ALERT SENT to {self._ip}  "
              f"conf={detection_result.confidence:.0%}  "
              f"consec={detection_result.consecutive}")

    def send_clear(self) -> None:
        packet = {"type": "CLEAR", "timestamp": time.time(), "source": "GROUND_UNIT_001"}
        self._send(packet)
        _log.info("CLEAR sent")
        print(f"CLEAR SENT to {self._ip}")

    # ------------------------------------------------------------------
    def _send(self, packet: dict) -> None:
        data = json.dumps(packet).encode()
        try:
            self._sock.sendto(data, (self._ip, self._port))
        except OSError as e:
            print(f"[AlertSender] Send error: {e}")
            _log.error("Send error: %s", e)

    def close(self) -> None:
        self._sock.close()
