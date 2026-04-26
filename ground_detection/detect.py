"""
detect.py — Ground Detection Unit main loop.

Usage:
    python detect.py --laptop 192.168.1.100
    python detect.py --laptop 192.168.1.100 --camera 1
    python detect.py --laptop 192.168.1.100 --sim
    python detect.py --laptop 192.168.1.100 --model path/to/model.onnx
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from vision.drone_detector import DroneDetector
from comms.alert_sender    import AlertSender

_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(_LOG_DIR / "detections.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
_log = logging.getLogger("detect")

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_THICK = 2


# ---------------------------------------------------------------------------
def _banner(camera: int, laptop_ip: str, model: str, sim: bool) -> None:
    print("═" * 36)
    print("  GROUND DETECTION UNIT")
    print("═" * 36)
    print(f"  Camera:    index {camera}")
    print(f"  Laptop:    {laptop_ip}")
    print(f"  Model:     {model}")
    print(f"  Threshold: 50%")
    print(f"  Lock:      3 consecutive frames")
    if sim:
        print(f"  Mode:      SIMULATION (alerts → 127.0.0.1)")
    print("═" * 36 + "\n")


def _draw(frame: np.ndarray, result, required: int) -> np.ndarray:
    out  = frame.copy()
    h, w = out.shape[:2]
    cx_f, cy_f = w // 2, h // 2

    if not result.detected:
        # SCANNING
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 180, 0), 2)
        cv2.line(out, (cx_f - 20, cy_f), (cx_f + 20, cy_f), (180, 180, 180), 1)
        cv2.line(out, (cx_f, cy_f - 20), (cx_f, cy_f + 20), (180, 180, 180), 1)
        cv2.putText(out, "SCANNING", (10, 30), _FONT, 0.7, (0, 180, 0), _THICK, cv2.LINE_AA)
        return out

    x1, y1, x2, y2 = result.bbox

    if result.consecutive < required:
        # CONFIRMING
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 220, 220), 3)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 220), 2)
        cv2.putText(out, f"CONFIRMING {result.consecutive}/{required}", (x1, y1 - 8),
                    _FONT, 0.65, (0, 220, 220), _THICK, cv2.LINE_AA)
        cv2.putText(out, "CONFIRMING", (10, 30), _FONT, 0.7, (0, 220, 220), _THICK, cv2.LINE_AA)
    else:
        # LOCKED
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(out, f"DRONE DETECTED  {result.confidence:.0%}", (x1, y1 - 8),
                    _FONT, 0.65, (0, 220, 0), _THICK, cv2.LINE_AA)
        cv2.putText(out, "DRONE DETECTED", (10, 30), _FONT, 0.8, (0, 0, 220), _THICK, cv2.LINE_AA)
        cx_t, cy_t = result.center
        cv2.line(out, (cx_f, cy_f), (cx_t, cy_t), (0, 220, 0), 1)

    return out


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(open(_ROOT / "config.yaml"))

    model_path  = args.model or str(_ROOT / cfg["detection"]["model_path"])
    laptop_ip   = args.laptop or cfg["comms"]["laptop_ip"]
    camera_idx  = args.camera if args.camera is not None else cfg["camera"]["index"]
    required    = cfg["detection"]["consecutive_required"]
    sim         = args.sim or cfg.get("simulation_mode", False)

    alert_ip = "127.0.0.1" if sim else laptop_ip
    _banner(camera_idx, laptop_ip, model_path, sim)

    detector = DroneDetector(model_path, conf_threshold=cfg["detection"]["conf_threshold"])
    sender   = AlertSender(alert_ip, port=cfg["comms"]["alert_port"])

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    if not cap.isOpened():
        print(f"ERROR: could not open camera {camera_idx}")
        sys.exit(1)

    csv_path = _LOG_DIR / "detections.csv"
    csv_f    = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_f)
    csv_w.writerow(["timestamp", "detected", "confidence", "consecutive", "bbox"])

    was_locked = False

    print("Camera active — press Q to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect(frame)
            locked = result.detected and result.consecutive >= required

            # State transitions
            if locked and not was_locked:
                print(f"DRONE DETECTED — confidence:{result.confidence:.0%}  "
                      f"consecutive:{result.consecutive}")
                _log.info("LOCKED  conf=%.3f consec=%d bbox=%s",
                          result.confidence, result.consecutive, result.bbox)
                sender.send_alert(result)
                was_locked = True

            elif locked and was_locked:
                # Still locked — resend alert every 30 consecutive frames (~1 s at 30 fps)
                if result.consecutive % 30 == 0:
                    sender.send_alert(result)

            elif not result.detected and was_locked:
                print("TARGET LOST")
                _log.info("CLEAR")
                sender.send_clear()
                was_locked = False

            # CSV
            csv_w.writerow([
                f"{time.time():.3f}",
                result.detected,
                f"{result.confidence:.3f}",
                result.consecutive,
                result.bbox,
            ])

            overlay = _draw(frame, result, required)
            cv2.imshow("Ground Detection Unit", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        csv_f.close()
        sender.close()
        cv2.destroyAllWindows()
        print("\nGround detection unit stopped.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Detection Unit")
    parser.add_argument("--camera", type=int,   default=None)
    parser.add_argument("--laptop", type=str,   default=None,
                        help="Laptop IP address to send alerts to")
    parser.add_argument("--model",  type=str,   default=None,
                        help="Path to YOLO ONNX model")
    parser.add_argument("--sim",    action="store_true",
                        help="Simulation mode — alerts sent to localhost")
    args = parser.parse_args()
    run(args)
