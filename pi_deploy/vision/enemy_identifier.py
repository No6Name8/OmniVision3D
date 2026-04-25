"""
enemy_identifier.py — Checks a drone crop against every model in the enemy list.

Loads all active entries from enemies/enemies.yaml at startup.
To add a new enemy: drop its .onnx in enemies/ and add an entry to the yaml.
No code changes needed.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import yaml

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class IdentificationResult:
    is_known_threat: bool
    threat_name:     Optional[str]
    threat_level:    Optional[str]
    confidence:      float
    all_scores:      Dict[str, float] = field(default_factory=dict)


class EnemyIdentifier:
    """
    Runs each active enemy's ONNX classifier on a cropped drone region.
    Returns the highest-confidence match above that enemy's threshold.
    """

    def __init__(self, enemies_yaml_path: str) -> None:
        self._yaml_path      = Path(enemies_yaml_path).resolve()
        self._sessions:      Dict[str, ort.InferenceSession] = {}
        self._thresholds:    Dict[str, float] = {}
        self._threat_levels: Dict[str, str]   = {}
        self._padding = 0.20

        with open(self._yaml_path) as f:
            self._cfg = yaml.safe_load(f)

        print("ENEMY LIST LOADED:")
        for i, target in enumerate(self._cfg.get("targets", []), start=1):
            name = target["name"]
            if not target.get("active", False):
                print(f"  [{i}] {name} — INACTIVE")
                continue

            # Model lives in the same folder as the yaml
            model_file = Path(target["model"]).name
            model_path = self._yaml_path.parent / model_file

            self._load_model(
                name, str(model_path),
                target.get("confidence_threshold", 0.85),
                target.get("threat_level", "unknown"),
            )
            level = target.get("threat_level", "unknown").upper()
            print(f"  [{i}] {name} — {level} THREAT — ACTIVE")

        if not self._sessions:
            print("  (no active enemies)")

    # ------------------------------------------------------------------
    def _load_model(self, name: str, model_path: str,
                    threshold: float, threat_level: str) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = max(4, os.cpu_count() or 4)
        opts.inter_op_num_threads     = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

        self._sessions[name]      = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._thresholds[name]    = threshold
        self._threat_levels[name] = threat_level

    # ------------------------------------------------------------------
    def identify(self, frame: np.ndarray, bbox: tuple) -> IdentificationResult:
        """
        Crop bbox from frame and run it through every active enemy model.
        Returns the best match above threshold, or not-known if nothing matches.
        """
        if not self._sessions:
            return IdentificationResult(False, None, None, 0.0)

        crop = self._preprocess(frame, bbox)
        all_scores: Dict[str, float] = {}
        best_name = None
        best_conf = 0.0

        for name, sess in self._sessions.items():
            in_name  = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            logits   = sess.run([out_name], {in_name: crop})[0][0]

            exp   = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            drone_conf = float(probs[1])      # class 1 = drone
            all_scores[name] = drone_conf

            if drone_conf >= self._thresholds[name] and drone_conf > best_conf:
                best_conf = drone_conf
                best_name = name

        return IdentificationResult(
            is_known_threat = best_name is not None,
            threat_name     = best_name,
            threat_level    = self._threat_levels.get(best_name) if best_name else None,
            confidence      = best_conf if best_name else max(all_scores.values(), default=0.0),
            all_scores      = all_scores,
        )

    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * self._padding)
        pad_y = int((y2 - y1) * self._padding)

        crop = frame[
            max(0, y1 - pad_y) : min(h, y2 + pad_y),
            max(0, x1 - pad_x) : min(w, x2 + pad_x),
        ]
        if crop.size == 0:
            crop = frame

        rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor  = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        return tensor.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 224, 224)

    # ------------------------------------------------------------------
    def add_enemy(self, name: str, model_path: str,
                  threat_level: str = "unknown") -> None:
        """Copy model to enemies/ folder, update yaml, load into memory."""
        import shutil
        dest = self._yaml_path.parent / Path(model_path).name
        shutil.copy(model_path, dest)

        entry = {
            "name":                 name,
            "model":                f"enemies/{dest.name}",
            "threat_level":         threat_level,
            "description":          "",
            "confidence_threshold": 0.85,
            "active":               True,
        }
        self._cfg.setdefault("targets", []).append(entry)
        with open(self._yaml_path, "w") as f:
            yaml.dump(self._cfg, f, default_flow_style=False, sort_keys=False)

        self._load_model(name, str(dest), 0.85, threat_level)
        print(f"NEW THREAT ADDED: {name}")

    def get_active_enemies(self) -> List[str]:
        return list(self._sessions.keys())
