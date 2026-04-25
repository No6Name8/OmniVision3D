"""
drone_classifier.py — Two-part drone identification.

Part 1 — Generic drone detection
    YOLO at 30% threshold (wide net — catches anything drone-shaped)

Part 2 — Enemy identification
    Crop is checked against every active model in enemies/enemies.yaml

Outputs
-------
NOT_DRONE     YOLO found nothing above generic threshold
UNKNOWN_DRONE YOLO detected something, but no enemy model matched
KNOWN_THREAT  YOLO detected + enemy model matched above its threshold
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from vision.yolo_detector    import YoloDetector
from vision.enemy_identifier import EnemyIdentifier

NOT_DRONE     = "NOT_DRONE"
UNKNOWN_DRONE = "UNKNOWN_DRONE"
KNOWN_THREAT  = "KNOWN_THREAT"


@dataclass
class ClassificationResult:
    status:       str
    threat_name:  Optional[str]         = None
    threat_level: Optional[str]         = None
    confidence:   float                 = 0.0
    bbox:         Optional[tuple]       = None
    center:       Optional[tuple]       = None
    message:      str                   = ""
    all_scores:   Dict[str, float]      = field(default_factory=dict)


class DroneClassifier:
    """
    Combines generic YOLO detection with enemy-list identification.
    Call classify(frame) — returns ClassificationResult.
    """

    def __init__(self, config: dict) -> None:
        ccfg = config.get("drone_classifier", {})
        ycfg = config["yolo"]
        root = config.get("_root", ".")

        generic_thresh = ccfg.get("generic_threshold", 0.30)

        # Generic detector — lower threshold to cast a wide net
        self._detector = YoloDetector(
            model_path     = os.path.join(root, ycfg["model_path"]),
            conf_threshold = generic_thresh,
            nms_threshold  = ycfg["nms_threshold"],
            input_size     = ycfg["input_size"],
        )

        # Enemy identifier — checks crop against every active enemy model
        enemies_file = ccfg.get("enemies_file", "enemies/enemies.yaml")
        self._enemy_id = EnemyIdentifier(os.path.join(root, enemies_file))

        # YOLO-skip optimisation: when KNOWN_THREAT, alternate YOLO and ID-only
        self._last_bbox:   Optional[tuple] = None
        self._last_center: Optional[tuple] = None
        self._skip_next:   bool            = False

        print(f"[DroneClassifier] Generic detection threshold: {generic_thresh:.0%}")
        print(f"[DroneClassifier] Active enemies: "
              f"{len(self._enemy_id.get_active_enemies())}")

    # ------------------------------------------------------------------
    def classify(self, frame: np.ndarray) -> ClassificationResult:
        """
        Full two-part classification on one frame.

        When the last result was KNOWN_THREAT the YOLO step is skipped every
        other frame — the enemy identifier runs on the cached bbox instead
        (much faster).  If confidence drops the cache is cleared and the next
        call runs full YOLO again.
        """
        # ---- YOLO-skip path (only when recently LOCKED onto a known threat) ----
        if self._skip_next and self._last_bbox is not None:
            self._skip_next = False
            id_res = self._enemy_id.identify(frame, self._last_bbox)
            if id_res.is_known_threat:
                return ClassificationResult(
                    status       = KNOWN_THREAT,
                    threat_name  = id_res.threat_name,
                    threat_level = id_res.threat_level,
                    confidence   = id_res.confidence,
                    bbox         = self._last_bbox,
                    center       = self._last_center,
                    message      = f"KNOWN THREAT: {id_res.threat_name}",
                    all_scores   = id_res.all_scores,
                )
            # Confidence dropped — clear cache, fall through to full YOLO
            self._last_bbox   = None
            self._last_center = None

        # ---- Full YOLO detection ----
        detections = self._detector.detect(frame)

        if not detections:
            self._last_bbox   = None
            self._last_center = None
            self._skip_next   = False
            return ClassificationResult(
                status  = NOT_DRONE,
                message = "No drone detected",
            )

        best   = detections[0]
        id_res = self._enemy_id.identify(frame, best.bbox)

        if id_res.is_known_threat:
            self._last_bbox   = best.bbox
            self._last_center = best.center
            self._skip_next   = True       # skip YOLO next frame
            return ClassificationResult(
                status       = KNOWN_THREAT,
                threat_name  = id_res.threat_name,
                threat_level = id_res.threat_level,
                confidence   = id_res.confidence,
                bbox         = best.bbox,
                center       = best.center,
                message      = f"KNOWN THREAT: {id_res.threat_name}",
                all_scores   = id_res.all_scores,
            )

        # Detected by YOLO but not in enemy list
        self._last_bbox   = None
        self._last_center = None
        self._skip_next   = False
        return ClassificationResult(
            status     = UNKNOWN_DRONE,
            confidence = best.confidence,
            bbox       = best.bbox,
            center     = best.center,
            message    = "UNKNOWN DRONE DETECTED",
            all_scores = id_res.all_scores,
        )

    def reset(self) -> None:
        self._last_bbox   = None
        self._last_center = None
        self._skip_next   = False
