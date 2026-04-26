"""
drone_classifier.py — YOLO-only locking with background enemy identification.

Locking logic
-------------
YOLO confidence >= 50% for N consecutive frames → LOCKED
No second classifier needed to lock.

Enemy identification
--------------------
Runs every enemy_id_every_n_frames frames after lock is established.
Result shown on screen but does NOT block the intercept.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from vision.yolo_detector    import YoloDetector
from vision.enemy_identifier import EnemyIdentifier

NOT_DRONE  = "NOT_DRONE"
CONFIRMING = "CONFIRMING"
LOCKED     = "LOCKED"
IDENTIFIED = "IDENTIFIED"


@dataclass
class ClassificationResult:
    status:       str
    confidence:   float            = 0.0
    bbox:         Optional[tuple]  = None
    center:       Optional[tuple]  = None
    consecutive:  int              = 0
    required:     int              = 3
    threat_name:  Optional[str]    = None
    threat_level: Optional[str]    = None
    message:      str              = ""
    all_scores:   Dict[str, float] = field(default_factory=dict)


class DroneClassifier:
    """
    YOLO at 50%+ for N consecutive frames → LOCKED.
    Enemy identifier runs non-blocking every N frames after lock.
    """

    def __init__(self, config: dict) -> None:
        ccfg = config.get("drone_classifier", {})
        ycfg = config["yolo"]
        root = config.get("_root", ".")

        self._threshold = ccfg.get("generic_threshold", 0.50)
        self._required  = ccfg.get("consecutive_required", 3)
        self._id_every  = ccfg.get("enemy_id_every_n_frames", 5)

        self._detector = YoloDetector(
            model_path     = os.path.join(root, ycfg["model_path"]),
            conf_threshold = self._threshold,
            nms_threshold  = ycfg["nms_threshold"],
            input_size     = ycfg["input_size"],
        )

        enemies_file = ccfg.get("enemies_file", "enemies/enemies.yaml")
        self._enemy_id = EnemyIdentifier(os.path.join(root, enemies_file))

        # State
        self._consecutive:  int           = 0
        self._id_counter:   int           = 0
        self._threat_name:  Optional[str] = None
        self._threat_level: Optional[str] = None

        print(f"[DroneClassifier] Lock threshold:  {self._threshold:.0%} x {self._required} frames")
        print(f"[DroneClassifier] Enemy ID:        every {self._id_every} frames (non-blocking)")
        print(f"[DroneClassifier] Active enemies:  {len(self._enemy_id.get_active_enemies())}")

    # ------------------------------------------------------------------
    def classify(self, frame: np.ndarray) -> ClassificationResult:
        detections = self._detector.detect(frame)

        if not detections:
            self._consecutive  = 0
            self._threat_name  = None
            self._threat_level = None
            self._id_counter   = 0
            return ClassificationResult(
                status  = NOT_DRONE,
                message = "No drone detected",
            )

        best = detections[0]
        self._consecutive += 1
        ran_id = False

        # Once locked, run enemy ID every N frames (non-blocking background task)
        id_scores: Dict[str, float] = {}
        if self._consecutive >= self._required:
            self._id_counter += 1
            if self._id_counter >= self._id_every:
                self._id_counter = 0
                ran_id = True
                id_res = self._enemy_id.identify(frame, best.bbox)
                id_scores = id_res.all_scores
                if id_res.is_known_threat:
                    self._threat_name  = id_res.threat_name
                    self._threat_level = id_res.threat_level

            status = IDENTIFIED if self._threat_name else LOCKED
            msg    = f"LOCKED: {self._threat_name}" if self._threat_name else "LOCKED ON"
            return ClassificationResult(
                status       = status,
                confidence   = best.confidence,
                bbox         = best.bbox,
                center       = best.center,
                consecutive  = self._consecutive,
                required     = self._required,
                threat_name  = self._threat_name,
                threat_level = self._threat_level,
                message      = msg,
                all_scores   = id_scores,
            )

        # Still confirming
        return ClassificationResult(
            status      = CONFIRMING,
            confidence  = best.confidence,
            bbox        = best.bbox,
            center      = best.center,
            consecutive = self._consecutive,
            required    = self._required,
            message     = f"CONFIRMING {self._consecutive}/{self._required}",
        )

    def reset(self) -> None:
        self._consecutive  = 0
        self._id_counter   = 0
        self._threat_name  = None
        self._threat_level = None
