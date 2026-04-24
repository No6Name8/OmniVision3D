"""
pipeline.py — Two-stage vision pipeline: YOLO detect → OmniVision3D confirm.

Phases:
    SCANNING   : no YOLO detection in current frame
    CONFIRMING : YOLO detected, waiting for 3 consecutive confirmed frames
    LOCKED     : 3+ consecutive confirmations above threshold

YOLO skip optimisation
----------------------
When LOCKED the drone position is already confirmed.  Running the full
512→320px YOLO every frame is wasteful — the confirmer crops and classifies
a tiny region in ~7ms instead of ~27ms.

Strategy:
  • SCANNING / CONFIRMING  → always run full YOLO (need to find / verify target)
  • LOCKED                 → alternate: YOLO one frame, confirmer-only next frame
                             If confirmer confidence drops to reset threshold,
                             immediately return to full YOLO on the next frame.
"""

import time
from collections import deque
from dataclasses import dataclass
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
    phase:     Phase
    detection: Optional[Detection]     = None
    identity:  Optional[IdentityResult] = None
    fps:       float = 0.0


class VisionPipeline:
    """
    Feeds frames through YOLO then OmniVision3D.
    Tracks rolling FPS over the last 30 frames.
    Skips YOLO every other frame when LOCKED to cut CPU load by ~40%.
    """

    def __init__(self, config: dict) -> None:
        ycfg = config["yolo"]
        icfg = config["identity_confirmation"]
        root = config.get("_root", ".")

        import os
        self._yolo = YoloDetector(
            model_path     = os.path.join(root, ycfg["model_path"]),
            conf_threshold = ycfg["confidence_threshold"],
            nms_threshold  = ycfg["nms_threshold"],
            input_size     = ycfg["input_size"],
        )
        self._confirmer = IdentityConfirmer(
            model_path           = os.path.join(root, icfg["model_path"]),
            conf_threshold       = icfg["confidence_threshold"],
            consecutive_required = icfg["consecutive_required"],
            reset_threshold      = icfg["reset_threshold"],
            crop_padding         = icfg["crop_padding"],
        )
        self._reset_thresh = icfg["reset_threshold"]

        self._fps_window: deque = deque(maxlen=30)
        self._t_last = time.perf_counter()

        # YOLO-skip state
        self._phase:     Phase             = Phase.SCANNING
        self._last_bbox: Optional[tuple]   = None
        self._last_det:  Optional[Detection] = None
        self._skip_next: bool              = False

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Run YOLO + confirmer on one frame, return PipelineResult."""
        now = time.perf_counter()
        self._fps_window.append(now - self._t_last)
        self._t_last = now
        fps = len(self._fps_window) / max(sum(self._fps_window), 1e-6)

        # ---- YOLO-skip path (LOCKED only) ----
        if self._skip_next and self._last_bbox is not None:
            self._skip_next = False
            identity = self._confirmer.confirm(frame, self._last_bbox)
            if identity.confidence >= self._reset_thresh:
                # Still seeing the drone — stay LOCKED
                return PipelineResult(
                    phase=Phase.LOCKED,
                    detection=self._last_det,
                    identity=identity,
                    fps=fps,
                )
            # Confidence dropped — clear cache and fall through to full YOLO
            self._last_bbox = None
            self._last_det  = None

        # ---- Full YOLO path ----
        detections = self._yolo.detect(frame)

        if not detections:
            self._confirmer.reset()
            self._phase     = Phase.SCANNING
            self._last_bbox = None
            self._last_det  = None
            self._skip_next = False
            return PipelineResult(phase=Phase.SCANNING, fps=fps)

        best     = detections[0]
        identity = self._confirmer.confirm(frame, best.bbox)

        if identity.is_confirmed:
            self._phase     = Phase.LOCKED
            self._last_bbox = best.bbox
            self._last_det  = best
            self._skip_next = True          # skip YOLO next frame
        else:
            self._phase     = Phase.CONFIRMING
            self._last_bbox = best.bbox
            self._last_det  = best
            self._skip_next = False

        return PipelineResult(
            phase=self._phase, detection=best, identity=identity, fps=fps
        )

    def reset(self) -> None:
        self._confirmer.reset()
        self._phase     = Phase.SCANNING
        self._last_bbox = None
        self._last_det  = None
        self._skip_next = False

    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray,
                     result: PipelineResult) -> np.ndarray:
        """Draw status overlay onto a frame copy."""
        out   = frame.copy()
        h, w  = out.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        thick = 2

        if result.phase == Phase.LOCKED:
            border_color = (0, 0, 220)
        elif result.phase == Phase.CONFIRMING:
            border_color = (0, 200, 220)
        else:
            border_color = (0, 180, 0)

        cv2.rectangle(out, (0, 0), (w - 1, h - 1), border_color, 4)

        if result.detection is not None:
            x1, y1, x2, y2 = result.detection.bbox
            if result.phase == Phase.LOCKED:
                box_color = (0, 220, 0)
                label = f"DJI CONFIRMED  {result.identity.confidence:.0%}"
            else:
                box_color = (0, 200, 220)
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

        cv2.putText(out, f"FPS: {result.fps:.1f}", (10, 28),
                    font, 0.7, (255, 255, 255), thick, cv2.LINE_AA)

        phase_text = result.phase.value
        (pw, _), _ = cv2.getTextSize(phase_text, font, 0.7, thick)
        cv2.putText(out, phase_text, (w - pw - 10, 28),
                    font, 0.7, (255, 255, 255), thick, cv2.LINE_AA)

        n_det = 1 if result.detection else 0
        cv2.putText(out, f"YOLO detections: {n_det}", (10, h - 10),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        return out
