"""
yolo_detector.py — Lightweight YOLOv8 wrapper using onnxruntime only.

Parses the YOLOv8 ONNX output [1, 5, N] where:
    N   = number of anchor boxes (5376 for imgsz=512)
    5   = cx, cy, w, h, class_confidence  (1-class model)

Coordinates are returned in original frame pixel space.
"""

import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

os.environ.setdefault("OMP_NUM_THREADS", "4")


@dataclass
class Detection:
    bbox:       tuple   # (x1, y1, x2, y2) in original frame pixels
    confidence: float
    center:     tuple   # (cx, cy) in original frame pixels
    area:       float   # pixel area


class YoloDetector:
    """
    Runs YOLOv8 nano ONNX inference and returns bounding-box detections.
    Designed for 1-class models exported with imgsz=512.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.50,
        nms_threshold:  float = 0.45,
        input_size:     int   = 512,
    ) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._conf_thresh = conf_threshold
        self._nms_thresh  = nms_threshold
        self._size        = input_size

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR frame.

        Returns a list of Detection objects sorted by confidence (highest first).
        Returns an empty list when nothing is detected.
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        resized = cv2.resize(frame, (self._size, self._size),
                             interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = (rgb.astype(np.float32) / 255.0)
        tensor  = tensor.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)

        # Inference
        raw = self._sess.run(
            [self._output_name], {self._input_name: tensor}
        )[0]                                               # (1, 5, N)

        # Parse — shape [5, N] -> [N, 5]
        preds = raw[0].T                                   # (N, 5)
        confs = preds[:, 4]

        mask = confs >= self._conf_thresh
        if not mask.any():
            return []

        preds = preds[mask]
        confs = confs[mask]

        # cx, cy, w, h are in input_size space — scale to original
        scale_x = orig_w / self._size
        scale_y = orig_h / self._size

        cx = preds[:, 0] * scale_x
        cy = preds[:, 1] * scale_y
        w  = preds[:, 2] * scale_x
        h  = preds[:, 3] * scale_y

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Clamp to frame
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = self._nms(boxes, confs)

        detections = []
        for i in keep:
            bx1, by1, bx2, by2 = boxes[i]
            c = float(confs[i])
            detections.append(Detection(
                bbox=(int(bx1), int(by1), int(bx2), int(by2)),
                confidence=c,
                center=(int((bx1 + bx2) / 2), int((by1 + by2) / 2)),
                area=float((bx2 - bx1) * (by2 - by1)),
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------
    def draw_detections(self, frame: np.ndarray,
                        detections: List[Detection]) -> np.ndarray:
        """Draw yellow bounding boxes and confidence labels onto a frame copy."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 220), 2)
            label = f"DJI? {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 220, 220), -1)
            cv2.putText(out, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray,
             iou_threshold: float = 0.45) -> List[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_threshold]
        return keep
