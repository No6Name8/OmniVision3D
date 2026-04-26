"""
main.py — OmniVision3D mission loop for Raspberry Pi.

Pipeline: YOLO → DroneClassifier → Phase state machine
Phases: SCANNING → CONFIRMING → LOCKED / SEARCHING

Launch flow:
    Laptop monitor sends LAUNCH command over UDP port 5556.
    Without --wait-launch the drone scans immediately (sim default).
    With --wait-launch it holds in STANDBY until LAUNCH is received.

Usage:
    python main.py --lat 24.7136 --lon 46.6753
    python main.py --video tests/test_video.mp4 --sim
    python main.py --sim                           # skip GPS, scan immediately
    python main.py --wait-launch                   # hold until laptop sends LAUNCH
"""

import argparse
import csv
import json
import logging
import signal
import socket
import sys
import threading
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
# UDP launch listener — runs in a background thread, sets mission_active flag
# ---------------------------------------------------------------------------
class _LaunchListener:
    """
    Listens on UDP port 5556 for LAUNCH / ABORT commands from the laptop monitor.
    Thread-safe: read mission_active and abort_requested from the main loop.
    """

    def __init__(self, port: int = 5556) -> None:
        self._port           = port
        self.mission_active  = False
        self.abort_requested = False
        self._sock           = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.5)
        self._sock.bind(("0.0.0.0", port))
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "_LaunchListener":
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[LaunchListener] Listening for commands on UDP :{self._port}")
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                pkt = json.loads(data.decode())
            except json.JSONDecodeError:
                continue

            cmd = pkt.get("command", "")
            if cmd == "LAUNCH":
                self.mission_active = True
                print(f"\n!! LAUNCH COMMAND RECEIVED FROM {addr[0]} !!")
                print(f"   Target confidence: {pkt.get('target_confidence', 0):.0%}")
                print(f"   Heading:           {pkt.get('compass_heading')}°\n")
                log.info("LAUNCH received from %s conf=%.3f",
                         addr[0], pkt.get("target_confidence", 0))
            elif cmd == "ABORT":
                self.abort_requested = True
                print(f"\n!! ABORT COMMAND RECEIVED FROM {addr[0]} !!\n")
                log.info("ABORT received from %s", addr[0])


# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(_ROOT)
    return cfg


def _print_banner(cfg: dict) -> None:
    ycfg  = cfg["yolo"]
    tcfg  = cfg["tracking"]
    dccfg = cfg.get("drone_classifier", {})
    print("\n" + "=" * 50)
    print("  OMNIVISION3D — DJI MINI 4 PRO")
    print("=" * 50)
    print(f"  YOLO:              {ycfg['model_path']}")
    print(f"  Lock threshold:    {dccfg.get('generic_threshold', 0.50):.0%} x "
          f"{dccfg.get('consecutive_required', 3)} frames")
    print(f"  Enemy ID:          every {dccfg.get('enemy_id_every_n_frames', 5)} frames (non-blocking)")
    print(f"  Enemy file:        {dccfg.get('enemies_file', 'enemies/enemies.yaml')}")
    print(f"  Lost -> Search:    {dccfg.get('lost_timeout_seconds', 2.0)}s (pipeline)")
    print(f"  Search -> Nav:     {tcfg['lost_searching_seconds']}s (tracker)")
    print(f"  Nav -> Abort:      {tcfg['lost_navigating_seconds']}s (tracker)")
    lcfg = cfg.get("launch_listener", {})
    if lcfg.get("enabled", False):
        print(f"  Launch port:       :{lcfg.get('port', 5556)} (waiting for laptop LAUNCH command)")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
def run_mission(args: argparse.Namespace) -> None:
    cfg     = _load_config(args.config)
    sim     = args.sim or cfg.get("simulation_mode", True)
    nav_cfg = cfg["navigation"]
    cam_cfg = cfg["camera"]
    dccfg   = cfg.get("drone_classifier", {})

    _print_banner(cfg)

    # ---- Signal handler (must be set early for standby loop) ----
    running = [True]
    def _stop(sig, frame): running[0] = False
    signal.signal(signal.SIGINT, _stop)

    # ---- Init ----
    pipeline = VisionPipeline(cfg)
    tracker  = Tracker(cfg)
    nav      = GPSNav(nav_cfg["home_lat"], nav_cfg["home_lon"])

    # ---- Launch listener (always running; required for --wait-launch) ----
    lcfg            = cfg.get("launch_listener", {})
    launch_port     = lcfg.get("port", 5556)
    wait_for_launch = args.wait_launch or lcfg.get("wait_for_launch", False)
    launch_listener = _LaunchListener(launch_port).start()

    if wait_for_launch:
        print("STANDBY — waiting for LAUNCH command from laptop monitor...")
        log.info("STANDBY — waiting for LAUNCH")
        while not launch_listener.mission_active:
            if not running[0]:
                launch_listener.stop()
                return
            time.sleep(0.1)
        print("LAUNCH RECEIVED — starting mission\n")
        log.info("LAUNCH received — mission start")

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    if not cap.isOpened():
        print("ERROR: could not open camera / video source")
        sys.exit(1)

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
    confirmations       = 0
    tracking_start      = None
    tracking_secs       = 0.0
    end_reason          = "user_exit"
    intercept_announced = False

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

            # ---- Tracker update ----
            # SEARCHING: pipeline already handles the lost timer; pass through
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

                if result.phase == Phase.SEARCHING:
                    intercept_announced = False
                    print(f"\rSEARCHING | lost:{cmd.lost_seconds:.1f}s | "
                          f"FPS:{result.fps:.1f}   ", end="", flush=True)

                elif cmd.state == TrackState.LOCKED:
                    if not intercept_announced:
                        print("\n!! INTERCEPT COMMITTED !!")
                        log.info("INTERCEPT COMMITTED")
                        intercept_announced = True
                    print(f"\rLOCKED | dx:{cmd.dx:+d} dy:{cmd.dy:+d} | "
                          f"pitch:{cmd.pitch:+.2f} yaw:{cmd.yaw:+.2f} | "
                          f"FPS:{result.fps:.1f}   ", end="", flush=True)

                elif cmd.state == TrackState.SEARCHING:
                    intercept_announced = False
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
                    print(f"\rSCANNING... | FPS:{result.fps:.1f}   ",
                          end="", flush=True)

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
        launch_listener.stop()
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
    parser.add_argument("--sim",          action="store_true", help="Skip GPS")
    parser.add_argument("--wait-launch",  action="store_true",
                        help="Hold in STANDBY until laptop sends LAUNCH command")
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    args = parser.parse_args()
    run_mission(args)
