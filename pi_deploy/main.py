"""
main.py — OmniVision3D mission loop for Raspberry Pi.

Pipeline: YOLO (generic, 30%) → EnemyIdentifier → Phase state machine
Phases: SCANNING → ALERT (unknown) / CONFIRMING (known) → LOCKED

Usage:
    python main.py --lat 24.7136 --lon 46.6753
    python main.py --video tests/test_video.mp4 --sim
    python main.py --sim                           # skip GPS, scan immediately
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

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from vision.pipeline    import VisionPipeline, Phase
from control.tracker    import Tracker, TrackState, TrackingCommand
from navigation.gps_nav import GPSNav

_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(_LOG_DIR / "mission.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
log = logging.getLogger("mission")

_alert_handler = logging.FileHandler(str(_LOG_DIR / "alerts.log"))
_alert_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
alert_log = logging.getLogger("alerts")
alert_log.setLevel(logging.INFO)
alert_log.addHandler(_alert_handler)
alert_log.propagate = False


# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(_ROOT)
    return cfg


def _print_banner(cfg: dict) -> None:
    ycfg  = cfg["yolo"]
    icfg  = cfg["identity_confirmation"]
    tcfg  = cfg["tracking"]
    dccfg = cfg.get("drone_classifier", {})
    print("\n" + "=" * 50)
    print("  OMNIVISION3D — DJI MINI 4 PRO")
    print("=" * 50)
    print(f"  YOLO:              {ycfg['model_path']} (11.6MB)")
    print(f"  Enemy models:      {dccfg.get('enemies_file', 'enemies/enemies.yaml')}")
    print(f"  Generic threshold: {dccfg.get('generic_threshold', 0.30):.0%}")
    print(f"  Alert after:       {dccfg.get('unknown_alert_frames', 3)} consecutive frames")
    print(f"  Frames to lock:    {icfg['consecutive_required']}")
    print(f"  Lost -> Search:    {tcfg['lost_searching_seconds']}s")
    print(f"  Lost -> Navigate:  {tcfg['lost_navigating_seconds']}s")
    print(f"  Lost -> Abort:     abort + return to base")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
def run_mission(args: argparse.Namespace) -> None:
    cfg     = _load_config(args.config)
    sim     = args.sim or cfg.get("simulation_mode", True)
    nav_cfg = cfg["navigation"]
    cam_cfg = cfg["camera"]
    dccfg   = cfg.get("drone_classifier", {})
    alert_frames_required = dccfg.get("unknown_alert_frames", 3)

    _print_banner(cfg)

    # ---- Init ----
    pipeline = VisionPipeline(cfg)
    tracker  = Tracker(cfg)
    nav      = GPSNav(nav_cfg["home_lat"], nav_cfg["home_lon"])

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    if not cap.isOpened():
        print("ERROR: could not open camera / video source")
        sys.exit(1)

    running = [True]
    def _stop(sig, frame): running[0] = False
    signal.signal(signal.SIGINT, _stop)

    # ---- Navigation ----
    if not sim:
        target = (args.lat, args.lon)
        print(f"NAVIGATING to lat={args.lat} lon={args.lon}")
        log.info("NAVIGATING to lat=%.6f lon=%.6f", args.lat, args.lon)
        nav.fly_to(args.lat, args.lon, nav_cfg["altitude"], simulation=False)
        while not nav.is_close_enough(target, nav_cfg["close_enough_meters"]):
            time.sleep(1.0)
    else:
        print("[SIM] Skipping GPS navigation")

    print("TARGET AREA REACHED — CAMERAS ACTIVE\n")
    log.info("CAMERAS ACTIVE")

    # ---- Mission stats ----
    frames_total        = 0
    yolo_hits           = 0
    alerts_logged       = 0
    confirmations       = 0
    tracking_start      = None
    tracking_secs       = 0.0
    end_reason          = "user_exit"
    intercept_announced = False
    alert_announced     = False

    last_print = time.time()

    csv_path = _LOG_DIR / "frames.csv"
    csv_f    = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_f)
    csv_w.writerow(["ts", "phase", "yolo_conf", "id_conf",
                    "consecutive", "pitch", "yaw", "fps"])

    try:
        while running[0]:
            ret, frame = cap.read()
            if not ret:
                end_reason = "video_end"
                break

            result = pipeline.process_frame(frame)

            # ---- Tracker update — skip for ALERT (hold position) ----
            if result.phase == Phase.ALERT:
                # Unknown drone: hold position, do not engage tracker
                # TODO: connect to command center alert API
                tracker.reset()
                cmd = TrackingCommand(state=TrackState.SEARCHING, lost_seconds=0.0)
            else:
                cmd = tracker.update(result)

            overlay = pipeline.draw_overlay(frame, result)

            frames_total += 1
            if result.detection:
                yolo_hits += 1
            if result.phase == Phase.LOCKED:
                confirmations += 1
                if tracking_start is None:
                    tracking_start = time.time()

            # ---- Console output (once per second) ----
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now

                if result.phase == Phase.ALERT:
                    consec = result.identity.consecutive if result.identity else 0
                    conf   = result.classification.confidence if result.classification else 0.0
                    print(f"\rALERT | UNKNOWN DRONE | "
                          f"frames:{consec} conf:{conf:.0%} | "
                          f"FPS:{result.fps:.1f}   ", end="", flush=True)
                    if not alert_announced:
                        print("\n!! UNKNOWN DRONE — ALERTING COMMAND !!")
                        log.info("UNKNOWN DRONE DETECTED — ALERTING")
                        alert_announced = True

                elif cmd.state == TrackState.LOCKED:
                    if not intercept_announced:
                        print("\n!! INTERCEPT COMMITTED !!")
                        log.info("INTERCEPT COMMITTED")
                        intercept_announced = True
                    alert_announced = False
                    print(f"\rLOCKED | dx:{cmd.dx:+d} dy:{cmd.dy:+d} | "
                          f"pitch:{cmd.pitch:+.2f} yaw:{cmd.yaw:+.2f} | "
                          f"FPS:{result.fps:.1f}   ", end="", flush=True)

                elif cmd.state == TrackState.SEARCHING:
                    intercept_announced = False
                    alert_announced     = False
                    print(f"\rSEARCHING | lost:{cmd.lost_seconds:.1f}s | "
                          f"FPS:{result.fps:.1f}   ", end="", flush=True)

                elif cmd.state == TrackState.NAVIGATING:
                    print(f"\rNAVIGATING | lost:{cmd.lost_seconds:.1f}s   ",
                          end="", flush=True)

                elif cmd.state == TrackState.ABORT:
                    print("\nTARGET LOST — RETURNING TO BASE")
                    log.info("ABORT — returning to base")
                    nav.return_to_base(simulation=sim)
                    end_reason = "target_lost_abort"
                    break

                else:
                    intercept_announced = False
                    alert_announced     = False
                    print(f"\rSCANNING... | FPS:{result.fps:.1f}   ",
                          end="", flush=True)

            # ---- Alert log (once, after N consecutive ALERT frames) ----
            if result.phase == Phase.ALERT:
                consec = result.identity.consecutive if result.identity else 0
                if consec >= alert_frames_required and alerts_logged == 0:
                    conf = result.classification.confidence if result.classification else 0.0
                    bbox = result.classification.bbox       if result.classification else None
                    alert_log.info("UNKNOWN_DRONE conf=%.3f bbox=%s", conf, bbox)
                    alerts_logged += 1
            else:
                alerts_logged = 0   # reset so next unknown episode gets logged

            # ---- CSV log ----
            yc = result.detection.confidence if result.detection else 0.0
            ic = result.identity.confidence  if result.identity  else 0.0
            cs = result.identity.consecutive if result.identity  else 0
            csv_w.writerow([
                f"{now:.3f}", result.phase.value,
                f"{yc:.3f}", f"{ic:.3f}", cs,
                f"{cmd.pitch:.3f}", f"{cmd.yaw:.3f}", f"{result.fps:.1f}",
            ])

            # ---- Display ----
            cv2.imshow("OmniVision3D", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                end_reason = "user_exit"
                break

    finally:
        cap.release()
        csv_f.close()
        cv2.destroyAllWindows()

        if tracking_start:
            tracking_secs = time.time() - tracking_start

        print("\n\n" + "=" * 50)
        print("  MISSION SUMMARY")
        print("=" * 50)
        print(f"  Frames processed:  {frames_total}")
        print(f"  YOLO detections:   {yolo_hits}")
        print(f"  Confirmations:     {confirmations}")
        print(f"  Tracking time:     {tracking_secs:.1f}s")
        print(f"  End reason:        {end_reason}")
        print("=" * 50 + "\n")
        log.info("MISSION ENDED frames=%d yolo=%d confirmed=%d track=%.1fs reason=%s",
                 frames_total, yolo_hits, confirmations, tracking_secs, end_reason)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVision3D mission")
    parser.add_argument("--lat",    type=float, default=24.7136)
    parser.add_argument("--lon",    type=float, default=46.6753)
    parser.add_argument("--camera", type=int,   default=0)
    parser.add_argument("--video",  default=None)
    parser.add_argument("--sim",    action="store_true", help="Skip GPS")
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    args = parser.parse_args()
    run_mission(args)
