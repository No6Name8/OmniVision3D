"""
camera_stream.py — Camera capture in a fully isolated OS process.

Architecture
------------
The camera runs in its own OS-level process and has zero interaction with
the inference GIL.  Its only job is:
    1. cap.read()
    2. copy frame → shared memory block
    3. increment counter

The main process reads from shared memory whenever it wants a frame.
No queue, no blocking, no GIL contention.  The camera always runs at the
hardware-limited rate regardless of how long inference takes.

Shared memory layout
--------------------
One flat block of (height × width × 3) uint8 bytes.
A separate mp.Value(c_uint64) acts as a monotonic write counter so the
reader can detect new frames without polling shared memory.
A mp.Lock guards the memcpy pair (write + read) to prevent tearing on
large frames.

Usage
-----
    stream = CameraStream(camera_index=0).start()
    while True:
        frame = stream.read()   # None → no new frame yet
        if frame is not None:
            process(frame)
    stream.stop()
"""

import ctypes
import multiprocessing as mp
from collections import deque
from multiprocessing import shared_memory
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Worker — runs in its own process, imports cv2 there (not in main process)
# ---------------------------------------------------------------------------

def _camera_worker(
    shm_name:  str,
    width:     int,
    height:    int,
    fps:       int,
    index:     int,
    counter:   "mp.Value",
    lock:      "mp.Lock",
    stop:      "mp.Event",
) -> None:
    """
    Standalone capture loop.  All imports are local so the main process
    never pays their cost and the worker has its own module namespace.
    """
    import cv2           # imported inside worker process only
    import numpy as np   # separate numpy instance in worker process

    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # discard buffered frames immediately

    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            buf[:] = frame          # write to shared memory
        counter.value += 1          # signal: new frame available

    cap.release()
    shm.close()


# ---------------------------------------------------------------------------
# CameraStream — main-process handle
# ---------------------------------------------------------------------------

class CameraStream:
    """
    Non-blocking camera handle backed by a separate OS process.

    read() always returns immediately — either the latest frame or None
    if no new frame has arrived since the last call.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width:        int = 640,
        height:       int = 480,
        fps:          int = 60,
    ) -> None:
        self._width  = width
        self._height = height

        nbytes      = width * height * 3
        self._shm   = shared_memory.SharedMemory(create=True, size=nbytes)
        self._buf   = np.ndarray((height, width, 3), dtype=np.uint8,
                                 buffer=self._shm.buf)

        self._lock    = mp.Lock()
        self._counter = mp.Value(ctypes.c_uint64, 0)
        self._stop    = mp.Event()
        self._last_n  = 0

        self._fps_window: deque = deque(maxlen=30)
        self._t_last = 0.0

        self._process = mp.Process(
            target=_camera_worker,
            args=(self._shm.name, width, height, fps,
                  camera_index, self._counter, self._lock, self._stop),
            daemon=True,
        )

    # ------------------------------------------------------------------
    def start(self) -> "CameraStream":
        import time
        self._t_last = time.perf_counter()
        self._process.start()
        return self

    def read(self) -> Optional[np.ndarray]:
        """Return latest frame (copy) or None if no new frame yet."""
        import time
        n = self._counter.value
        if n == self._last_n:
            return None
        with self._lock:
            frame = self._buf.copy()
        now = time.perf_counter()
        self._fps_window.append(now - self._t_last)
        self._t_last = now
        self._last_n = n
        return frame

    def get_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        return len(self._fps_window) / max(sum(self._fps_window), 1e-6)

    def stop(self) -> None:
        self._stop.set()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
        self._shm.close()
        self._shm.unlink()
