"""
drone_main.py — Lean flight script for Raspberry Pi.

No display. No overlays. No CLAHE. No UI.
Just: camera -> YOLO -> consecutive counter -> PID -> motors.

Kill switch: create a file named KILL in this directory.
    From laptop over SSH:  touch pi_deploy/KILL
    drone_main.py detects it and stops cleanly.

Usage:
    python drone_main.py
    python drone_main.py --sim
    python drone_main.py --camera 1 --config config.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from vision.camera_stream    import CameraStream
from vision.yolo_detector    import YoloDetector
from control.motor_controller import MotorController

_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(_LOG_DIR / "drone_main.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
log = logging.getLogger("drone_main")

_KILL_FILE = _ROOT / "KILL"

SCANNING   = "SCANNING"
CONFIRMING = "CONFIRMING"
LOCKED     = "LOCKED"


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(open(_ROOT / args.config))
    cfg["_root"] = str(_ROOT)

    sim        = args.sim or cfg.get("simulation_mode", True)
    cam_cfg    = cfg["camera"]
    yolo_cfg   = cfg["yolo"]
    dc_cfg     = cfg.get("drone_classifier", {})
    required   = dc_cfg.get("consecutive_required", 3)

    model_path = str(_ROOT / yolo_cfg["model_path"])

    print("OMNIVISION3D DRONE -- READY")
    print(f"  Camera: {args.camera}")
    print(f"  Model:  {yolo_cfg['model_path']}")
    print(f"  Mode:   {'SIM' if sim else 'LIVE'}")
    print(f"  Lock:   {required} consecutive frames at "
          f"{yolo_cfg.get('confidence_threshold', 0.50):.0%}+")
    print()

    log.info("STARTUP camera=%d model=%s sim=%s",
             args.camera, yolo_cfg["model_path"], sim)

    # ---- Init ----
    camera = CameraStream(
        camera_index=args.camera,
        width=cam_cfg["width"],
        height=cam_cfg["height"],
    ).start()

    yolo = YoloDetector(
        model_path     = model_path,
        conf_threshold = yolo_cfg.get("confidence_threshold", 0.50),
        nms_threshold  = yolo_cfg.get("nms_threshold", 0.45),
        input_size     = yolo_cfg.get("input_size", 320),
    )

    motor_ctrl = MotorController(cfg)

    # ---- State ----
    consecutive:         int   = 0
    last_center               = None
    intercept_announced: bool  = False
    frame_count:         int   = 0
    fps_timer:           float = time.time()
    phase:               str   = SCANNING

    mission_start = time.time()
    log.info("MISSION_START")
    print("Scanning...\n")

    # ---- Main loop ----
    try:
        while True:
            # 1. Kill switch (file-based — no keyboard needed on drone)
            if _KILL_FILE.exists():
                print("KILL signal received -- stopping")
                log.info("KILL_SIGNAL")
                break

            # 2. Get frame (non-blocking — None means no new frame yet)
            frame = camera.read()
            if frame is None:
                continue

            # 3. YOLO detect
            detections = yolo.detect(frame)

            if detections:
                best = max(detections, key=lambda d: d.confidence)
                consecutive += 1
                last_center = best.center
            else:
                consecutive = 0
                last_center = None

            # 4. Determine phase
            if consecutive >= required:
                phase = LOCKED
            elif consecutive > 0:
                phase = CONFIRMING
            else:
                phase = SCANNING

            # 5. Motor control
            if phase == LOCKED:
                if not intercept_announced:
                    motor_ctrl.engage_intercept()
                    print("LOCKED -- INTERCEPT COMMITTED")
                    log.info("INTERCEPT_COMMITTED")
                    intercept_announced = True

                frame_cx = frame.shape[1] // 2
                frame_cy = frame.shape[0] // 2
                dx = last_center[0] - frame_cx
                dy = last_center[1] - frame_cy

                powers = motor_ctrl.calculate_motor_powers(dx, dy)
                motor_ctrl.send_to_pixhawk(powers)

            elif phase == CONFIRMING:
                intercept_announced = False
                motor_ctrl.disengage_intercept()

            else:  # SCANNING
                intercept_announced = False
                motor_ctrl.disengage_intercept()
                motor_ctrl.reset()

            # 6. Console + log every 30 frames (~1s at 30fps)
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30.0 / max(elapsed, 1e-6)
                fps_timer = time.time()
                print(f"{phase} | FPS:{fps:.1f} | consecutive:{consecutive}")
                log.info("phase=%s fps=%.1f consec=%d", phase, fps, consecutive)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt")

    finally:
        motor_ctrl.reset()
        camera.stop()
        mission_secs = time.time() - mission_start
        print(f"\nDRONE STOPPED -- frames:{frame_count}  "
              f"mission:{mission_secs:.1f}s")
        log.info("SHUTDOWN frames=%d mission=%.1fs", frame_count, mission_secs)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVision3D lean flight script")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file (relative to pi_deploy/)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--sim",    action="store_true",
                        help="Simulation mode (no real MAVLink)")
    args = parser.parse_args()
    run(args)
