"""
pipeline.py — Two-stage vision pipeline: YOLO detect → OmniVision3D confirm.

Phases:
    SCANNING   : no YOLO detection in current frame
    CONFIRMING : YOLO detected, waiting for 3 consecutive confirmed frames
    LOCKED     : 3+ consecutive confirmations above threshold
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from vision.yolo_detector     import YoloDetector, Detection
from vision.identity_confirmer import IdentityConfirmer, IdentityResult


class Phase(str, Enum):
    SCANNING    = "SCANNING"
    CONFIRMING  = "CONFIRMING"
    LOCKED      = "LOCKED"


@dataclass
class PipelineResult:
    phase:       Phase
    detection:   Optional[Detection]  = None
    identity:    Optional[IdentityResult] = None
    fps:         float = 0.0


class VisionPipeline:
    """
    Feeds every frame through YOLO then OmniVision3D.
    Tracks rolling FPS over the last 30 frames.
    """

    def __init__(self, config: dict) -> None:
        ycfg = config["yolo"]
        icfg = config["identity_confirmation"]
        root = config.get("_root", ".")

        import os
        self._yolo = YoloDetector(
            model_path      = os.path.join(root, ycfg["model_path"]),
            conf_threshold  = ycfg["confidence_threshold"],
            nms_threshold   = ycfg["nms_threshold"],
            input_size      = ycfg["input_size"],
        )
        self._confirmer = IdentityConfirmer(
            model_path           = os.path.join(root, icfg["model_path"]),
            conf_threshold       = icfg["confidence_threshold"],
            consecutive_required = icfg["consecutive_required"],
            reset_threshold      = icfg["reset_threshold"],
            crop_padding         = icfg["crop_padding"],
        )

        self._fps_window: deque = deque(maxlen=30)
        self._t_last = time.perf_counter()

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Run YOLO + confirmer on one frame, return PipelineResult."""
        now = time.perf_counter()
        self._fps_window.append(now - self._t_last)
        self._t_last = now
        fps = len(self._fps_window) / max(sum(self._fps_window), 1e-6)

        detections = self._yolo.detect(frame)

        if not detections:
            self._confirmer.reset()
            return PipelineResult(phase=Phase.SCANNING, fps=fps)

        best = detections[0]                   # highest confidence YOLO hit
        identity = self._confirmer.confirm(frame, best.bbox)

        if identity.is_confirmed:
            phase = Phase.LOCKED
        else:
            phase = Phase.CONFIRMING

        return PipelineResult(
            phase=phase, detection=best, identity=identity, fps=fps
        )

    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray,
                     result: PipelineResult) -> np.ndarray:
        """Draw status overlay onto a frame copy."""
        out   = frame.copy()
        h, w  = out.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        thick = 2

        # ---- Border colour ----
        if result.phase == Phase.LOCKED:
            border_color = (0, 0, 220)    # red — intercept committed
        elif result.phase == Phase.CONFIRMING:
            border_color = (0, 200, 220)  # yellow
        else:
            border_color = (0, 180, 0)    # green — scanning safe

        cv2.rectangle(out, (0, 0), (w - 1, h - 1), border_color, 4)

        # ---- Detection box ----
        if result.detection is not None:
            x1, y1, x2, y2 = result.detection.bbox
            if result.phase == Phase.LOCKED:
                box_color = (0, 220, 0)       # green box when locked
                label = f"DJI CONFIRMED  {result.identity.confidence:.0%}"
            else:
                box_color = (0, 200, 220)     # yellow when confirming
                n   = result.identity.consecutive if result.identity else 0
                req = result.identity.required    if result.identity else 3
                cf  = result.identity.confidence  if result.identity else 0.0
                label = f"CONFIRMING {n}/{req}  {cf:.0%}"

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 3)
            (tw, th), _ = cv2.getTextSize(label, font, 0.65, thick)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 10, y1),
                          box_color, -1)
            cv2.putText(out, label, (x1 + 5, y1 - 5),
                        font, 0.65, (0, 0, 0), thick, cv2.LINE_AA)

        # ---- Top-left: FPS ----
        fps_text = f"FPS: {result.fps:.1f}"
        cv2.putText(out, fps_text, (10, 28),
                    font, 0.7, (255, 255, 255), thick, cv2.LINE_AA)

        # ---- Top-right: phase ----
        phase_text = result.phase.value
        (pw, _), _ = cv2.getTextSize(phase_text, font, 0.7, thick)
        cv2.putText(out, phase_text, (w - pw - 10, 28),
                    font, 0.7, (255, 255, 255), thick, cv2.LINE_AA)

        # ---- Bottom-left: YOLO count ----
        n_det = 1 if result.detection else 0
        cv2.putText(out, f"YOLO detections: {n_det}", (10, h - 10),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        return out
