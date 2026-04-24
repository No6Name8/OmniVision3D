"""
frame_enhancer.py — Lightweight CLAHE enhancement for OmniVision3D.

Applies CLAHE to the L channel of LAB colourspace only.
Target: under 5ms per frame on Raspberry Pi 4.
"""

import time

import cv2
import numpy as np


class FrameEnhancer:
    def __init__(self, enable: bool = True) -> None:
        self._enabled = enable
        self._clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._benchmark()

    # ------------------------------------------------------------------
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return frame
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        l_eq       = self._clahe.apply(l)
        lab_merged = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        return self._enabled

    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    def _benchmark(self) -> None:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        enabled_orig = self._enabled
        self._enabled = True

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            self.enhance(frame.copy())
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000
        self._enabled = enabled_orig

        print(f"[FrameEnhancer] CLAHE benchmark: {avg_ms:.2f}ms avg over 100 frames", flush=True)
        if avg_ms > 5.0:
            print(f"[FrameEnhancer] WARNING: {avg_ms:.2f}ms exceeds 5ms target", flush=True)
