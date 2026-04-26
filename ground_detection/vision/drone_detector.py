"""
drone_detector.py — Lightweight YOLO ONNX wrapper for the ground detection unit.

Parses YOLOv8 output tensor [1, 5, 8400]:
  row 0-3: cx, cy, w, h (normalised to input_size)
  row 4:   objectness/class score
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort


@dataclass
class DetectionResult:
    detected:    bool
    confidence:  float
    bbox:        Optional[Tuple[int, int, int, int]]
    center:      Optional[Tuple[int, int]]
    consecutive: int


class DroneDetector:
    def __init__(
        self,
        model_path:     str,
        conf_threshold: float = 0.50,
        input_size:     int   = 320,
    ) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads     = 4
        opts.inter_op_num_threads     = 1

        self._sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._conf        = conf_threshold
        self._size        = input_size
        self._consecutive = 0

        print(f"[DroneDetector] Model:     {os.path.basename(model_path)}")
        print(f"[DroneDetector] Threshold: {conf_threshold:.0%}  input: {input_size}px")

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> DetectionResult:
        import cv2

        h, w = frame.shape[:2]
        scale_x = w / self._size
        scale_y = h / self._size

        # Preprocess
        img = cv2.resize(frame, (self._size, self._size))
        img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, normalise
        blob = np.expand_dims(img.transpose(2, 0, 1), 0)   # HWC→NCHW

        raw = self._sess.run([self._output_name], {self._input_name: blob})[0]
        # raw shape: [1, 5, num_anchors]
        preds = raw[0].T   # [num_anchors, 5]

        scores = preds[:, 4]
        mask   = scores >= self._conf
        if not mask.any():
            self._consecutive = 0
            return DetectionResult(False, 0.0, None, None, 0)

        filtered = preds[mask]
        scores_f = filtered[:, 4]

        # NMS
        cx = filtered[:, 0] * scale_x
        cy = filtered[:, 1] * scale_y
        bw = filtered[:, 2] * scale_x
        bh = filtered[:, 3] * scale_y
        x1 = cx - bw / 2;  y1 = cy - bh / 2
        x2 = cx + bw / 2;  y2 = cy + bh / 2

        import cv2 as _cv2
        boxes_cv = np.stack([x1, y1, bw, bh], axis=1).tolist()
        keep = _cv2.dnn.NMSBoxes(boxes_cv, scores_f.tolist(), self._conf, 0.45)
        if len(keep) == 0:
            self._consecutive = 0
            return DetectionResult(False, 0.0, None, None, 0)

        idx  = int(keep[0]) if isinstance(keep[0], (int, np.integer)) else int(keep[0][0])
        conf = float(scores_f[idx])
        bbox = (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]))
        cx_i = int((x1[idx] + x2[idx]) / 2)
        cy_i = int((y1[idx] + y2[idx]) / 2)

        self._consecutive += 1
        return DetectionResult(True, conf, bbox, (cx_i, cy_i), self._consecutive)

    def reset(self) -> None:
        self._consecutive = 0
