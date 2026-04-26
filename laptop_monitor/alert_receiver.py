"""
alert_receiver.py — UDP listener for alerts from ground detection Pi.

Runs a background thread that parses JSON packets and stores the latest
alert so the main loop can poll it without blocking.
"""

import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Optional

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(_LOG_DIR / "alerts_received.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
_log = logging.getLogger("alert_receiver")


class AlertReceiver:
    def __init__(self, port: int = 5555) -> None:
        self._port   = port
        self._sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.5)          # non-blocking with short timeout
        self._sock.bind(("0.0.0.0", port))

        self._latest:  Optional[dict] = None
        self._lock     = threading.Lock()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        print(f"[AlertReceiver] Listening on 0.0.0.0:{port}")

    # ------------------------------------------------------------------
    def start(self) -> "AlertReceiver":
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    # ------------------------------------------------------------------
    def get_latest(self) -> Optional[dict]:
        with self._lock:
            pkt = self._latest
            self._latest = None          # consume — caller gets it once
        return pkt

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                pkt = json.loads(data.decode())
            except json.JSONDecodeError as e:
                _log.warning("Bad packet from %s: %s", addr, e)
                continue

            pkt["_from"] = addr[0]
            with self._lock:
                self._latest = pkt

            _log.info("RX %s  from %s", pkt.get("type"), addr[0])
