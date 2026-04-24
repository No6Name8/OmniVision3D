"""
identity_confirmer.py — OmniVision3D crop-and-classify confirmer.

Crops the YOLO bbox from the frame, runs the MobileNetV2 classifier,
and counts consecutive high-confidence frames before committing to LOCKED.

Thresholds:
    >= conf_threshold  (0.85) : count += 1
    <  reset_threshold (0.50) : count  = 0
    between                   : count unchanged (hold)
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

os.environ.setdefault("OMP_NUM_THREADS", "4")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class IdentityResult:
    is_confirmed:  bool
    confidence:    float
    consecutive:   int
    required:      int


class IdentityConfirmer:
    """
    Two-stage confirmation: only declare LOCKED after `consecutive_required`
    frames each with confidence >= conf_threshold.
    """

    def __init__(
        self,
        model_path:           str,
        conf_threshold:       float = 0.85,
        consecutive_required: int   = 3,
        reset_threshold:      float = 0.50,
        crop_padding:         float = 0.20,
    ) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads          = max(4, os.cpu_count() or 4)
        opts.inter_op_num_threads          = 1
        opts.graph_optimization_level      = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode                = ort.ExecutionMode.ORT_SEQUENTIAL
        self._sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

        self._conf_thresh  = conf_threshold
        self._reset_thresh = reset_threshold
        self._required     = consecutive_required
        self._padding      = crop_padding
        self._counter      = 0

    # ------------------------------------------------------------------
    def confirm(self, frame: np.ndarray,
                bbox: tuple) -> IdentityResult:
        """
        Classify the cropped bbox region.

        Args:
            frame: Full BGR frame.
            bbox:  (x1, y1, x2, y2) from YoloDetector in frame pixel coords.

        Returns:
            IdentityResult with updated consecutive counter.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add padding
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * self._padding)
        pad_y = int(bh * self._padding)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return IdentityResult(False, 0.0, self._counter, self._required)

        # Preprocess: BGR -> RGB, resize 224x224, normalize ImageNet
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor  = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor  = tensor.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 224, 224)

        # Inference
        logits = self._sess.run(
            [self._output_name], {self._input_name: tensor}
        )[0][0]                                            # (2,)

        exp   = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        # Class 1 = drone confidence
        drone_conf = float(probs[1])

        # Update counter
        if drone_conf >= self._conf_thresh:
            self._counter += 1
        elif drone_conf < self._reset_thresh:
            self._counter = 0
        # else: hold — no change

        return IdentityResult(
            is_confirmed = self._counter >= self._required,
            confidence   = drone_conf,
            consecutive  = self._counter,
            required     = self._required,
        )

    def reset(self) -> None:
        self._counter = 0
