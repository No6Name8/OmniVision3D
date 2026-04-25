"""
pipeline.py — Four-phase vision pipeline.

Phases
------
SCANNING   no drone detected
ALERT      unknown drone (not in enemy list) — alert command, hold position
CONFIRMING known enemy detected, building consecutive-frame count
LOCKED     known enemy confirmed N times — intercept committed

Flow
----
DroneClassifier.classify() returns NOT_DRONE / UNKNOWN_DRONE / KNOWN_THREAT.

NOT_DRONE     → SCANNING   (reset all counters)
UNKNOWN_DRONE → ALERT      (immediately; logged after unknown_alert_frames consec.)
KNOWN_THREAT  → CONFIRMING (< consecutive_required frames)
               → LOCKED    (>= consecutive_required frames)
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from vision.yolo_detector    import Detection
from vision.identity_confirmer import IdentityResult   # kept for backward compat
from vision.drone_classifier import (
    DroneClassifier, ClassificationResult,
    NOT_DRONE, UNKNOWN_DRONE, KNOWN_THREAT,
)


class Phase(str, Enum):
    SCANNING   = "SCANNING"
    ALERT      = "ALERT"
    CONFIRMING = "CONFIRMING"
    LOCKED     = "LOCKED"


@dataclass
class PipelineResult:
    phase:          Phase
    detection:      Optional[Detection]          = None   # backward compat
    identity:       Optional[IdentityResult]     = None   # backward compat
    fps:            float                        = 0.0
    classification: Optional[ClassificationResult] = None


class VisionPipeline:
    """
    Wraps DroneClassifier with phase-transition logic and consecutive counters.
    Maintains backward-compatible PipelineResult fields for tracker / main.py.
    """

    def __init__(self, config: dict) -> None:
        icfg = config["identity_confirmation"]
        ccfg = config.get("drone_classifier", {})

        self._classifier = DroneClassifier(config)
        self._required   = icfg["consecutive_required"]         # frames to reach LOCKED
        self._alert_req  = ccfg.get("unknown_alert_frames", 3)  # frames to confirm ALERT

        self._fps_window: deque = deque(maxlen=30)
        self._t_last = time.perf_counter()

        # Phase state
        self._phase:         Phase = Phase.SCANNING
        self._known_consec:  int   = 0
        self._unknown_consec: int  = 0

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        now = time.perf_counter()
        self._fps_window.append(now - self._t_last)
        self._t_last = now
        fps = len(self._fps_window) / max(sum(self._fps_window), 1e-6)

        cr = self._classifier.classify(frame)

        # ---- NOT_DRONE ----
        if cr.status == NOT_DRONE:
            self._known_consec   = 0
            self._unknown_consec = 0
            self._phase          = Phase.SCANNING
            return PipelineResult(phase=Phase.SCANNING, fps=fps, classification=cr)

        # ---- UNKNOWN_DRONE ----
        if cr.status == UNKNOWN_DRONE:
            self._known_consec    = 0
            self._unknown_consec += 1
            self._phase           = Phase.ALERT

            det = _make_detection(cr)
            # Synthetic IdentityResult — keeps CSV / UI code working
            identity = IdentityResult(
                is_confirmed = False,
                confidence   = cr.confidence,
                consecutive  = self._unknown_consec,
                required     = self._alert_req,
            )
            return PipelineResult(
                phase=Phase.ALERT, detection=det,
                identity=identity, fps=fps, classification=cr,
            )

        # ---- KNOWN_THREAT ----
        self._unknown_consec  = 0
        self._known_consec   += 1

        if self._known_consec >= self._required:
            self._phase = Phase.LOCKED
        else:
            self._phase = Phase.CONFIRMING

        det      = _make_detection(cr)
        identity = IdentityResult(
            is_confirmed = self._phase == Phase.LOCKED,
            confidence   = cr.confidence,
            consecutive  = self._known_consec,
            required     = self._required,
        )
        return PipelineResult(
            phase=self._phase, detection=det,
            identity=identity, fps=fps, classification=cr,
        )

    def reset(self) -> None:
        self._classifier.reset()
        self._phase          = Phase.SCANNING
        self._known_consec   = 0
        self._unknown_consec = 0

    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray,
                     result: PipelineResult) -> np.ndarray:
        out  = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        tk   = 2

        # Border colour by phase
        border = {
            Phase.SCANNING:   (0, 180, 0),
            Phase.ALERT:      (0, 140, 255),
            Phase.CONFIRMING: (0, 200, 220),
            Phase.LOCKED:     (0, 0,   220),
        }.get(result.phase, (180, 180, 180))
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), border, 4)

        # Detection box
        if result.detection is not None:
            x1, y1, x2, y2 = result.detection.bbox

            if result.phase == Phase.LOCKED:
                box_c = (0, 220, 0)
                cr    = result.classification
                name  = cr.threat_name if cr else "KNOWN"
                label = f"KNOWN THREAT: {name}  {result.identity.confidence:.0%}"
            elif result.phase == Phase.CONFIRMING:
                box_c = (0, 200, 220)
                n     = result.identity.consecutive if result.identity else 0
                req   = result.identity.required    if result.identity else 3
                cf    = result.identity.confidence  if result.identity else 0.0
                label = f"CONFIRMING {n}/{req}  {cf:.0%}"
            else:  # ALERT
                box_c = (0, 140, 255)
                label = "UNKNOWN DRONE"

            cv2.rectangle(out, (x1, y1), (x2, y2), box_c, 3)
            (tw, th), _ = cv2.getTextSize(label, font, 0.60, tk)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_c, -1)
            cv2.putText(out, label, (x1 + 5, y1 - 5),
                        font, 0.60, (0, 0, 0), tk, cv2.LINE_AA)

        # FPS — top-left
        cv2.putText(out, f"FPS: {result.fps:.1f}", (10, 28),
                    font, 0.7, (255, 255, 255), tk, cv2.LINE_AA)

        # Phase — top-right
        pt = result.phase.value
        (pw, _), _ = cv2.getTextSize(pt, font, 0.7, tk)
        cv2.putText(out, pt, (w - pw - 10, 28),
                    font, 0.7, (255, 255, 255), tk, cv2.LINE_AA)

        # Detection count — bottom-left
        n_det = 1 if result.detection else 0
        cv2.putText(out, f"Detections: {n_det}", (10, h - 10),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        return out


# ---------------------------------------------------------------------------
def _make_detection(cr: ClassificationResult) -> Optional[Detection]:
    """Create a Detection from a ClassificationResult for backward compat."""
    if cr.bbox is None:
        return None
    return Detection(
        bbox       = cr.bbox,
        confidence = cr.confidence,
        center     = cr.center or (
            (cr.bbox[0] + cr.bbox[2]) // 2,
            (cr.bbox[1] + cr.bbox[3]) // 2,
        ),
        area = float(
            (cr.bbox[2] - cr.bbox[0]) * (cr.bbox[3] - cr.bbox[1])
        ),
    )
