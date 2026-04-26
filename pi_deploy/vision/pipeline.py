"""
pipeline.py — Four-phase vision pipeline.

Phases
------
SCANNING   no drone detected
CONFIRMING drone detected, building consecutive-frame count (< 3 frames)
LOCKED     3+ consecutive frames at 50%+ — intercept committed
SEARCHING  drone lost after LOCKED (< lost_timeout seconds)

Flow
----
DroneClassifier.classify() returns NOT_DRONE / CONFIRMING / LOCKED / IDENTIFIED.
SEARCHING is managed here via a lost timer; falls back to SCANNING after timeout.
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from vision.yolo_detector      import Detection
from vision.identity_confirmer import IdentityResult
from vision.drone_classifier   import (
    DroneClassifier, ClassificationResult,
    NOT_DRONE,
    CONFIRMING as CR_CONFIRMING,
    LOCKED     as CR_LOCKED,
    IDENTIFIED,
)


class Phase(str, Enum):
    SCANNING   = "SCANNING"
    CONFIRMING = "CONFIRMING"
    LOCKED     = "LOCKED"
    SEARCHING  = "SEARCHING"


@dataclass
class PipelineResult:
    phase:          Phase
    detection:      Optional[Detection]            = None
    identity:       Optional[IdentityResult]       = None
    fps:            float                          = 0.0
    classification: Optional[ClassificationResult] = None


class VisionPipeline:
    """
    Wraps DroneClassifier with phase-transition logic and a SEARCHING timer.
    Maintains backward-compatible PipelineResult fields for tracker / main.py.
    """

    def __init__(self, config: dict) -> None:
        ccfg = config.get("drone_classifier", {})
        self._classifier   = DroneClassifier(config)
        self._lost_timeout = ccfg.get("lost_timeout_seconds", 2.0)

        self._fps_window: deque = deque(maxlen=30)
        self._t_last  = time.perf_counter()
        self._phase   = Phase.SCANNING
        self._lost_since: Optional[float] = None

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        now = time.perf_counter()
        self._fps_window.append(now - self._t_last)
        self._t_last = now
        fps = len(self._fps_window) / max(sum(self._fps_window), 1e-6)

        cr = self._classifier.classify(frame)

        # ---- NOT_DRONE ----
        if cr.status == NOT_DRONE:
            if self._phase in (Phase.LOCKED, Phase.SEARCHING):
                if self._lost_since is None:
                    self._lost_since = now
                lost = now - self._lost_since
                if lost < self._lost_timeout:
                    self._phase = Phase.SEARCHING
                    return PipelineResult(phase=Phase.SEARCHING, fps=fps, classification=cr)
                else:
                    self._phase      = Phase.SCANNING
                    self._lost_since = None
            else:
                self._phase      = Phase.SCANNING
                self._lost_since = None
            return PipelineResult(phase=Phase.SCANNING, fps=fps, classification=cr)

        # ---- Drone detected — reset lost timer ----
        self._lost_since = None

        det = _make_detection(cr)
        identity = IdentityResult(
            is_confirmed = cr.status in (CR_LOCKED, IDENTIFIED),
            confidence   = cr.confidence,
            consecutive  = cr.consecutive,
            required     = cr.required,
        )

        if cr.status == CR_CONFIRMING:
            self._phase = Phase.CONFIRMING
        else:
            self._phase = Phase.LOCKED

        return PipelineResult(
            phase=self._phase, detection=det,
            identity=identity, fps=fps, classification=cr,
        )

    def reset(self) -> None:
        self._classifier.reset()
        self._phase      = Phase.SCANNING
        self._lost_since = None

    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray,
                     result: PipelineResult) -> np.ndarray:
        out  = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cx_f, cy_f = w // 2, h // 2

        if result.phase == Phase.SCANNING:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 180, 0), 2)
            cv2.line(out, (cx_f - 20, cy_f), (cx_f + 20, cy_f), (180, 180, 180), 1)
            cv2.line(out, (cx_f, cy_f - 20), (cx_f, cy_f + 20), (180, 180, 180), 1)

        elif result.phase == Phase.SEARCHING:
            lost = (time.perf_counter() - self._lost_since) if self._lost_since else 0.0
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
            cv2.putText(out, f"SEARCHING {lost:.1f}s", (10, 40),
                        font, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

        elif result.phase == Phase.CONFIRMING:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 220, 220), 3)
            det = result.detection
            if det is not None:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 220), 2)
                n   = result.identity.consecutive if result.identity else 0
                req = result.identity.required    if result.identity else 3
                cv2.putText(out, f"CONFIRMING {n}/{req}", (x1, y1 - 8),
                            font, 0.65, (0, 220, 220), 2, cv2.LINE_AA)

        else:  # LOCKED
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)
            cv2.putText(out, "LOCKED ON", (10, 40),
                        font, 0.9, (0, 0, 220), 2, cv2.LINE_AA)
            det = result.detection
            if det is not None:
                x1, y1, x2, y2 = det.bbox
                cx_t, cy_t = det.center
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 3)
                cr   = result.classification
                name = cr.threat_name if (cr and cr.threat_name) else None
                conf = result.identity.confidence if result.identity else 0.0
                label = f"LOCKED: {name}  {conf:.0%}" if name else f"LOCKED ON  {conf:.0%}"
                cv2.putText(out, label, (x1, y1 - 8),
                            font, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
                if name:
                    cv2.putText(out, name, (x1, y2 + 20),
                                font, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
                cv2.line(out, (cx_f, cy_f), (cx_t, cy_t), (0, 220, 0), 1)
            cv2.line(out, (cx_f - 20, cy_f), (cx_f + 20, cy_f), (0, 220, 0), 1)
            cv2.line(out, (cx_f, cy_f - 20), (cx_f, cy_f + 20), (0, 220, 0), 1)

        cv2.putText(out, f"FPS: {result.fps:.1f}", (10, h - 10),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        pt = result.phase.value
        (pw, _), _ = cv2.getTextSize(pt, font, 0.65, 2)
        cv2.putText(out, pt, (w - pw - 10, 28),
                    font, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        return out


# ---------------------------------------------------------------------------
def _make_detection(cr: ClassificationResult) -> Optional[Detection]:
    if cr.bbox is None:
        return None
    return Detection(
        bbox       = cr.bbox,
        confidence = cr.confidence,
        center     = cr.center or (
            (cr.bbox[0] + cr.bbox[2]) // 2,
            (cr.bbox[1] + cr.bbox[3]) // 2,
        ),
        area = float((cr.bbox[2] - cr.bbox[0]) * (cr.bbox[3] - cr.bbox[1])),
    )
