"""
detection.py — YOLO detection thread for the ground station.

Grabs frames from the camera, runs YOLO, writes annotated frame
and detection state to shared_state.  No display here — UI reads
from shared_state.frame.
"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent   # pi_deploy/
sys.path.insert(0, str(_ROOT))

from vision.yolo_detector import YoloDetector
from ground_station import shared_state as ss

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw(frame: np.ndarray, phase: str, conf: float,
          consecutive: int, required: int,
          bbox, center) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cx_f, cy_f = w // 2, h // 2

    if phase == "LOCKED":
        cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 0, 220), 4)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 3)
            if center:
                cv2.line(out, (cx_f, cy_f), center, (0, 220, 0), 1)
            cv2.putText(out, f"LOCKED {conf:.0%}", (x1, max(y1-8, 16)),
                        _FONT, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.line(out, (cx_f-20, cy_f), (cx_f+20, cy_f), (0, 220, 0), 1)
        cv2.line(out, (cx_f, cy_f-20), (cx_f, cy_f+20), (0, 220, 0), 1)

    elif phase == "CONFIRMING":
        cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 220, 220), 3)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 220), 2)
            cv2.putText(out, f"CONFIRM {consecutive}/{required}",
                        (x1, max(y1-8, 16)),
                        _FONT, 0.6, (0, 220, 220), 2, cv2.LINE_AA)

    elif phase == "SEARCHING":
        cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 165, 255), 3)
        cv2.putText(out, "SEARCHING", (10, 36),
                    _FONT, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    else:  # SCANNING
        cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 180, 0), 2)
        cv2.line(out, (cx_f-20, cy_f), (cx_f+20, cy_f), (180, 180, 180), 1)
        cv2.line(out, (cx_f, cy_f-20), (cx_f, cy_f+20), (180, 180, 180), 1)

    cv2.putText(out, phase, (w - 140, 28),
                _FONT, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def thread_detection(
    camera_index: int,
    model_path: str,
    conf_threshold: float = 0.50,
    nms_threshold:  float = 0.45,
    input_size:     int   = 320,
    consecutive_required: int = 3,
    lost_timeout_secs:    float = 2.0,
) -> None:
    """
    Detection thread.  Opens camera, runs YOLO every frame,
    maintains phase state, writes to shared_state.
    """
    yolo = YoloDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        input_size=input_size,
    )

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[Detection] ERROR: cannot open camera {camera_index}")
        return

    print(f"[Detection] camera {camera_index} opened")

    consecutive   = 0
    lost_since    = None
    phase         = "SCANNING"

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        detections = yolo.detect(frame)
        now = time.perf_counter()

        if detections:
            best = detections[0]
            consecutive += 1
            lost_since   = None

            if consecutive >= consecutive_required:
                phase = "LOCKED"
            else:
                phase = "CONFIRMING"

            bbox   = best.bbox
            center = best.center
            conf   = best.confidence
        else:
            if phase in ("LOCKED", "SEARCHING"):
                if lost_since is None:
                    lost_since = now
                if now - lost_since < lost_timeout_secs:
                    phase = "SEARCHING"
                else:
                    phase       = "SCANNING"
                    lost_since  = None
                    consecutive = 0
            else:
                phase       = "SCANNING"
                consecutive = 0
                lost_since  = None

            bbox   = None
            center = None
            conf   = 0.0

        annotated = _draw(frame, phase, conf, consecutive,
                          consecutive_required, bbox, center)

        with ss.lock():
            ss.state.phase       = phase
            ss.state.confidence  = conf
            ss.state.consecutive = consecutive
            ss.state.bbox        = bbox
            ss.state.det_center  = center
            ss.state.frame       = annotated

    cap.release()
