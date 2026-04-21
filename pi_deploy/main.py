"""
main.py — OmniVision3D mission loop for Raspberry Pi.

Stages:
    STARTUP     — load config, initialise camera and ONNX model
    NAVIGATION  — fly to provided GPS coordinates (simulation or real)
    SCANNING    — run inference on every frame, wait for confident detection
    TRACKING    — pass detections to PID tracker, log corrections
    EXIT        — release resources on Ctrl-C

Usage:
    python main.py --lat 24.7136 --lon 46.6753
    python main.py --lat 24.7136 --lon 46.6753 --video path/to/test.mp4
"""

import argparse
import csv
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import yaml

# ---------------------------------------------------------------------------
# Bootstrap: make sure pi_deploy/ itself is importable regardless of CWD
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from vision.pi_predict   import PiPredictor, draw_overlay
from control.tracker     import TargetTracker
from navigation.gps_nav  import navigate_to, get_current_position, is_close_enough


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging(log_dir: str) -> logging.Logger:
    ld = Path(log_dir)
    ld.mkdir(exist_ok=True)
    logger = logging.getLogger("mission")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(ld / "mission.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------
SCANNING  = "SCANNING"
TRACKING  = "TRACKING"


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------
def run_mission(
    lat: float,
    lon: float,
    config_path: str = "config.yaml",
    video_path: str  = None,
) -> None:
    # ---- STARTUP ----
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log_dir = str(_ROOT / cfg["log_dir"])
    logger  = _setup_logging(log_dir)

    conf_thresh = cfg["confidence_threshold"]
    sim         = cfg["simulation_mode"]
    model_path  = str(_ROOT / cfg["model_path"])

    logger.info("=" * 54)
    logger.info("OMNIVISION3D READY")
    logger.info("  model    : %s", model_path)
    logger.info("  target   : lat=%.6f  lon=%.6f", lat, lon)
    logger.info("  sim mode : %s", sim)
    logger.info("=" * 54)
    print("\nOMNIVISION3D READY\n")

    predictor = PiPredictor(model_path, cfg["inference_size"])
    tracker   = TargetTracker(
        hold_duration=cfg["lost_target_timeout"],
        simulation=sim,
    )

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(cfg["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])

    if not cap.isOpened():
        logger.error("ERROR: could not open camera / video source")
        sys.exit(1)

    # Graceful Ctrl-C
    running = [True]
    def _stop(sig, frame):
        running[0] = False
    signal.signal(signal.SIGINT, _stop)

    # ---- NAVIGATION ----
    logger.info("NAVIGATING to lat=%.6f  lon=%.6f  alt=%.1f",
                lat, lon, 50.0)
    navigate_to(lat, lon, altitude=50.0, simulation=sim)

    current = get_current_position()
    if sim or is_close_enough(current, (lat, lon)):
        logger.info("APPROACHING TARGET — ACTIVATING CAMERAS")
        print("\nAPPROACHING TARGET — ACTIVATING CAMERAS\n")

    # ---- SCAN / TRACK loop ----
    state          = SCANNING
    lost_since     = 0.0
    fps_disp       = 0.0
    frame_count    = 0
    t_fps          = time.time()

    # CSV log for post-mission analysis
    csv_path = Path(log_dir) / "frames.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "state", "label", "confidence",
                         "dx", "dy", "pitch", "yaw", "fps"])

    try:
        while running[0]:
            ret, frame = cap.read()
            if not ret:
                break

            t0     = time.perf_counter()
            result = predictor.predict_frame(frame)
            _ms    = (time.perf_counter() - t0) * 1000

            frame_count += 1
            if frame_count % 30 == 0:
                fps_disp = 30.0 / max(time.time() - t_fps, 1e-6)
                t_fps    = time.time()

            detected = result.label == "drone" and result.confidence >= conf_thresh

            # State transitions
            if state == SCANNING:
                if detected:
                    state      = TRACKING
                    lost_since = 0.0
                    logger.info("LOCKED ON — conf=%.3f", result.confidence)

            elif state == TRACKING:
                if not detected:
                    if lost_since == 0.0:
                        lost_since = time.monotonic()
                    if time.monotonic() - lost_since >= cfg["lost_target_timeout"]:
                        state = SCANNING
                        logger.info("Target lost — returning to SCAN")
                        print("\nSCANNING...")
                else:
                    lost_since = 0.0

            # Tracker update
            pitch, yaw = tracker.update(result.offset, detected)

            # Console output
            if state == TRACKING:
                dx, dy = result.offset
                print(
                    f"\rLOCKED ON — offset({dx:+.0f},{dy:+.0f})"
                    f"  pitch({pitch:+.3f})  yaw({yaw:+.3f})"
                    f"  {fps_disp:.1f}FPS  {_ms:.0f}ms   ",
                    end="", flush=True,
                )
            else:
                print(
                    f"\rSCANNING...  conf={result.confidence:.1%}"
                    f"  {fps_disp:.1f}FPS  {_ms:.0f}ms   ",
                    end="", flush=True,
                )

            # Log frame
            now = time.time()
            dx, dy = result.offset
            csv_writer.writerow([
                f"{now:.3f}", state, result.label,
                f"{result.confidence:.4f}", f"{dx:.1f}", f"{dy:.1f}",
                f"{pitch:.4f}", f"{yaw:.4f}", f"{fps_disp:.1f}",
            ])

    finally:
        cap.release()
        csv_file.close()
        print("\n\nMISSION ENDED")
        logger.info("MISSION ENDED — frames=%d  log=%s", frame_count, csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVision3D mission")
    parser.add_argument("--lat",    type=float, required=True)
    parser.add_argument("--lon",    type=float, required=True)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--video",  default=None,
                        help="Path to video file instead of live camera")
    args = parser.parse_args()

    run_mission(
        lat=args.lat,
        lon=args.lon,
        config_path=args.config,
        video_path=args.video,
    )
