"""
camera_stream.py — Non-blocking threaded camera capture for OmniVision3D.

Always returns the latest frame immediately; never queues stale frames.
"""

import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
    ) -> None:
        self._index  = camera_index
        self._width  = width
        self._height = height
        self._fps    = fps

        self._cap:    Optional[cv2.VideoCapture] = None
        self._frame:  Optional[np.ndarray] = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._fps_window: deque = deque(maxlen=30)
        self._t_last = time.perf_counter()

    # ------------------------------------------------------------------
    def start(self) -> "CameraStream":
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                continue

            now = time.perf_counter()
            self._fps_window.append(now - self._t_last)
            self._t_last = now

            with self._lock:
                self._frame = frame

    # ------------------------------------------------------------------
    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        return len(self._fps_window) / max(sum(self._fps_window), 1e-6)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
