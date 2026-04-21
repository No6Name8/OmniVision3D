"""
pi_predict.py — OmniVision3D camera inference for Raspberry Pi.

Reads from the Pi camera (or a video file), runs the ONNX model on every
frame, and returns a DetectionResult with label, confidence, and pixel
offset from frame centre to the target.

Usage:
    python vision/pi_predict.py                          # live camera
    python vision/pi_predict.py --video path/to/vid.mp4  # video file
    python vision/pi_predict.py --show                   # display window
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# onnxruntime is the only heavy dependency — keeps Pi's memory usage low
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(_LOG_DIR / "detections.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)

# ImageNet normalisation constants (float32)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["no_drone", "drone"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    label:      str          # "drone" or "no_drone"
    confidence: float        # 0.0 – 1.0
    offset:     Tuple[float, float]  # (dx, dy) pixels from frame centre


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class PiPredictor:
    """
    Loads the ONNX model once and exposes predict_frame() for the mission loop.
    Designed to run at 15+ FPS on Raspberry Pi 4 (4-thread ONNX CPU runtime).
    """

    def __init__(self, model_path: str, inference_size: int = 224) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._size        = inference_size

    def predict_frame(self, bgr: np.ndarray) -> DetectionResult:
        """
        Run one inference pass on a BGR frame.

        Returns:
            DetectionResult with label, confidence, and (dx, dy) offset.
            Offset is (0, 0) when no drone is detected.
        """
        h, w = bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Preprocess: resize → RGB float → normalize → NCHW
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._size, self._size),
                             interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        logits = self._sess.run(
            [self._output_name], {self._input_name: tensor}
        )[0][0]                                         # shape (2,)

        # Softmax
        exp   = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        idx        = int(probs.argmax())
        label      = CLASS_NAMES[idx]
        confidence = float(probs[idx])

        # Pixel offset: use full frame centre when drone detected
        # (bounding-box regression will be wired here once det_head is exported)
        if label == "drone":
            offset = (0.0, 0.0)  # TODO: replace with det_head bbox centre
        else:
            offset = (0.0, 0.0)

        return DetectionResult(label=label, confidence=confidence, offset=offset)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    result: DetectionResult,
    fps: float,
    confidence_threshold: float = 0.85,
) -> np.ndarray:
    """Draw detection status, confidence, and FPS counter onto a BGR frame."""
    out  = frame.copy()
    h, w = out.shape[:2]
    is_drone = result.label == "drone" and result.confidence >= confidence_threshold

    if is_drone:
        status      = "LOCKED ON"
        box_color   = (0, 220, 0)
        text_bg     = (0, 160, 0)
        inset       = 12
        cv2.rectangle(out, (inset, inset), (w - inset, h - inset), box_color, 3)
        conf_text   = f"DRONE  {result.confidence:.1%}"
    else:
        status      = "SCANNING..."
        box_color   = (0, 0, 200)
        text_bg     = (0, 0, 160)
        conf_text   = f"NO DRONE  {result.confidence:.1%}"

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 2
    pad   = 6

    # Status banner (top-left)
    for i, txt in enumerate([status, conf_text]):
        (tw, th), bl = cv2.getTextSize(txt, font, scale, thick)
        y_off = i * (th + 2 * pad + bl + 4)
        cv2.rectangle(out, (0, y_off), (tw + 2 * pad, y_off + th + 2 * pad + bl),
                      text_bg, -1)
        cv2.putText(out, txt, (pad, y_off + th + pad),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # FPS counter (bottom-right)
    fps_text = f"{fps:.1f} FPS"
    (fw, fh), _ = cv2.getTextSize(fps_text, font, 0.55, 1)
    cv2.rectangle(out, (w - fw - 12, h - fh - 16), (w, h), (30, 30, 30), -1)
    cv2.putText(out, fps_text, (w - fw - 6, h - 8),
                font, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

    # Crosshair at frame centre
    cv2.drawMarker(out, (w // 2, h // 2), (200, 200, 200),
                   cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(
    model_path: str,
    camera_index: int = 0,
    video_path: Optional[str] = None,
    frame_width: int = 640,
    frame_height: int = 480,
    inference_size: int = 224,
    confidence_threshold: float = 0.85,
    show: bool = False,
) -> None:
    predictor = PiPredictor(model_path, inference_size)

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        sys.exit("ERROR: could not open camera / video source")

    print("OmniVision3D  |  press Ctrl-C or Q to quit")

    fps_acc  = 0.0
    fps_disp = 0.0
    frame_count = 0
    t_fps = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0     = time.perf_counter()
            result = predictor.predict_frame(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Rolling FPS (updated every 30 frames)
            frame_count += 1
            if frame_count % 30 == 0:
                fps_disp = 30.0 / (time.time() - t_fps)
                t_fps    = time.time()

            if result.label == "drone" and result.confidence >= confidence_threshold:
                dx, dy = result.offset
                logging.info(
                    "DRONE  conf=%.3f  offset=(%.1f, %.1f)  ms=%.1f",
                    result.confidence, dx, dy, elapsed_ms,
                )

            annotated = draw_overlay(frame, result, fps_disp, confidence_threshold)

            if show:
                cv2.imshow("OmniVision3D", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                status = "LOCKED ON" if (
                    result.label == "drone"
                    and result.confidence >= confidence_threshold
                ) else "SCANNING"
                print(
                    f"\r{status:<12}  conf={result.confidence:.1%}"
                    f"  {fps_disp:.1f} FPS  {elapsed_ms:.1f}ms",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        print("\nCamera released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVision3D Pi inference")
    parser.add_argument("--model",  default=str(_HERE / "omnivision3d.onnx"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video",  default=None)
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--show",   action="store_true")
    args = parser.parse_args()

    run(
        model_path=args.model,
        camera_index=args.camera,
        video_path=args.video,
        frame_width=args.width,
        frame_height=args.height,
        show=args.show,
    )
