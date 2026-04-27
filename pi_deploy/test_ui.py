"""
test_ui.py — OmniVision3D live test UI.

Window layout: 640x480 camera feed (left) | 320x480 status panel (right)
Total: 960x480

Usage:
    python test_ui.py --camera 0
    python test_ui.py --camera 1
    python test_ui.py --video path/to/file.mp4
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from vision.frame_enhancer import FrameEnhancer
from vision.pipeline       import VisionPipeline, Phase
from control.tracker       import TrackState


# ---------------------------------------------------------------------------
# Phase / state colours (BGR)
# ---------------------------------------------------------------------------
_PHASE_COLOR = {
    Phase.SCANNING:   (255, 255, 255),
    Phase.SEARCHING:  (0,   165, 255),   # orange
    Phase.CONFIRMING: (0,   220, 220),   # cyan
    Phase.LOCKED:     (0,   220,   0),   # green
}
_STATE_COLOR = {
    TrackState.LOCKED:     (0, 255,   0),
    TrackState.SEARCHING:  (0, 165, 255),
    TrackState.NAVIGATING: (0, 200, 255),
    TrackState.ABORT:      (0,   0, 255),
}

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = 0.52
_FONT_MED   = 0.65
_THICK      = 1
_THICK2     = 2


# ---------------------------------------------------------------------------
# Camera overlay drawing
# ---------------------------------------------------------------------------

def _draw_crosshair(img: np.ndarray, color: tuple) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 20, cy), (cx + 20, cy), color, 1)
    cv2.line(img, (cx, cy - 20), (cx, cy + 20), color, 1)


def draw_camera_overlay(frame: np.ndarray, result, track_state, lost_secs: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    phase = result.phase

    # ---- ABORT ----
    if track_state == TrackState.ABORT:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(out, "RETURNING TO BASE", (10, h // 2),
                    _FONT, 0.8, (0, 0, 255), _THICK2, cv2.LINE_AA)
        return out

    # ---- SEARCHING (drone lost after lock) ----
    if phase == Phase.SEARCHING:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
        cv2.putText(out, f"SEARCHING {lost_secs:.1f}s", (10, 30),
                    _FONT, _FONT_MED, (0, 165, 255), _THICK2, cv2.LINE_AA)
        return out

    # ---- SCANNING ----
    if phase == Phase.SCANNING:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 180, 0), 2)
        _draw_crosshair(out, (200, 200, 200))
        return out

    det = result.detection
    if det is None:
        return out

    x1, y1, x2, y2 = det.bbox
    cx_t, cy_t = det.center

    # ---- CONFIRMING ----
    if phase == Phase.CONFIRMING:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 200, 220), 3)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 220), 2)
        n   = result.identity.consecutive if result.identity else 0
        req = result.identity.required    if result.identity else 3
        cv2.putText(out, f"CONFIRMING {n}/{req}", (x1, y1 - 8),
                    _FONT, _FONT_SMALL, (0, 200, 220), _THICK, cv2.LINE_AA)
        return out

    # ---- LOCKED ----
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 3)
    conf = result.identity.confidence if result.identity else 0.0
    name = (result.classification.threat_name
            if result.classification and result.classification.threat_name
            else "DRONE")
    cv2.putText(out, f"LOCKED: {name}  {conf:.0%}", (x1, y1 - 8),
                _FONT, _FONT_SMALL, (0, 220, 0), _THICK2, cv2.LINE_AA)
    frame_cx, frame_cy = w // 2, h // 2
    cv2.line(out, (frame_cx, frame_cy), (cx_t, cy_t), (0, 220, 0), 1)
    _draw_crosshair(out, (0, 220, 0))
    return out


# ---------------------------------------------------------------------------
# Status panel
# ---------------------------------------------------------------------------

def build_status_panel(result, track_state, lost_secs: float,
                       cam_fps: float, log: deque) -> np.ndarray:
    panel = np.zeros((480, 320, 3), dtype=np.uint8)

    def put(text, y, color=(200, 200, 200), scale=_FONT_SMALL, thick=_THICK):
        cv2.putText(panel, text, (10, y), _FONT, scale, color, thick, cv2.LINE_AA)

    def hline(y):
        cv2.line(panel, (5, y), (315, y), (60, 60, 60), 1)

    # Title
    put("OMNIVISION3D", 28, (255, 255, 255), 0.7, _THICK2)
    hline(38)

    # Phase
    phase = result.phase
    phase_color = _PHASE_COLOR.get(phase, (200, 200, 200))
    if track_state in (TrackState.SEARCHING, TrackState.NAVIGATING, TrackState.ABORT):
        if phase not in (Phase.SEARCHING, Phase.CONFIRMING, Phase.LOCKED):
            phase_color = _STATE_COLOR.get(track_state, (200, 200, 200))
            phase_label = track_state.value
        else:
            phase_label = phase.value
    else:
        phase_label = phase.value
    put(f"Phase: {phase_label}", 60, phase_color)

    # FPS
    put(f"FPS:   {cam_fps:.1f}", 82)
    hline(96)

    # Detection stats
    yolo_conf = result.detection.confidence if result.detection else 0.0
    id_conf   = result.identity.confidence  if result.identity  else 0.0
    consec    = result.identity.consecutive if result.identity  else 0
    req       = result.identity.required    if result.identity  else 3
    put(f"YOLO:  {yolo_conf:.0%}", 116)
    put(f"ID:    {id_conf:.0%}",   138)
    put(f"Conf:  {consec}/{req}",  160)

    # Threat name (when known)
    if result.classification and result.classification.threat_name:
        name_color = (0, 220, 0) if phase == Phase.LOCKED else (0, 165, 255)
        put(f"  {result.classification.threat_name}", 178, name_color, 0.42)
        hline(192)
    else:
        hline(174)

    # Recent detections log
    put("RECENT DETECTIONS", 208, (160, 160, 160))
    for i, entry in enumerate(list(log)[-5:]):
        color = (0, 165, 255) if "ALERT" in entry else \
                (0, 220, 0)   if "LOCKED" in entry else \
                (120, 120, 120)
        put(entry, 228 + i * 20, color, 0.42)

    # Lost timer if applicable
    if track_state in (TrackState.SEARCHING, TrackState.NAVIGATING):
        put(f"Lost: {lost_secs:.1f}s", 330, _STATE_COLOR[track_state])

    hline(440)
    put("Q=Quit  E=Enhance  R=Reset  S=Shot", 458, (80, 80, 80), 0.40)

    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _scan_cameras(max_index: int = 6) -> list:
    """Return list of (index, cap) for every camera that opens successfully."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append((i, cap))
                continue
        cap.release()
    return found


def pick_camera() -> int:
    """
    Show a live preview of every detected camera.
    User presses the displayed number key to select one.
    Returns the chosen camera index.
    """
    print("Scanning for cameras...")
    cameras = _scan_cameras()

    if not cameras:
        print("No cameras found — defaulting to index 0")
        return 0

    if len(cameras) == 1:
        idx, cap = cameras[0]
        cap.release()
        print(f"One camera found: index {idx}")
        return idx

    font   = cv2.FONT_HERSHEY_SIMPLEX
    chosen = None

    cv2.namedWindow("Select Camera", cv2.WINDOW_NORMAL)

    while chosen is None:
        tiles = []
        for idx, cap in cameras:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
            tile = cv2.resize(frame, (320, 240))
            label = f"Press {idx}  (Camera {idx})"
            cv2.rectangle(tile, (0, 0), (320, 36), (30, 30, 30), -1)
            cv2.putText(tile, label, (8, 26), font, 0.7, (0, 220, 220), 2, cv2.LINE_AA)
            tiles.append(tile)

        # Lay out tiles in a row
        row = np.hstack(tiles)
        w = row.shape[1]
        info = np.zeros((50, w, 3), dtype=np.uint8)
        cv2.putText(info, "Press the number key to select your camera   |   ESC = use camera 0",
                    (10, 34), font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        combined = np.vstack([row, info])

        cv2.imshow("Select Camera", combined)

        if cv2.getWindowProperty("Select Camera", cv2.WND_PROP_VISIBLE) < 1:
            chosen = cameras[0][0]
            break

        key = cv2.waitKey(30) & 0xFF
        if key == 27:                          # ESC → use first camera
            chosen = cameras[0][0]
        elif chr(key).isdigit():
            num = int(chr(key))
            if any(i == num for i, _ in cameras):
                chosen = num

    for _, cap in cameras:
        cap.release()
    cv2.destroyWindow("Select Camera")
    print(f"Camera {chosen} selected.")
    return chosen


def _startup_banner() -> None:
    lines = [
        "=" * 36,
        "  OMNIVISION3D TEST UI",
        "=" * 36,
        "  USB camera:  --camera 0",
        "  HDMI feed:   --camera 1",
        "  Video file:  --video path.mp4",
        "",
        "  E = toggle enhancement",
        "  R = reset detection",
        "  S = screenshot",
        "  Q = quit",
        "=" * 36,
    ]
    for l in lines:
        print(l)


def run(args: argparse.Namespace) -> None:
    _startup_banner()

    cfg = yaml.safe_load(open(_ROOT / "config.yaml"))
    cfg["_root"] = str(_ROOT)

    enhancer = FrameEnhancer(enable=False)
    pipeline = VisionPipeline(cfg)

    log: deque = deque(maxlen=20)

    # ---- Open video / camera (direct capture — no subprocess spin) ----
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        camera_index = args.camera if args.camera is not None else pick_camera()
        cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.camera}")
        return

    cv2.namedWindow("OmniVision3D", cv2.WINDOW_NORMAL)

    track_state = TrackState.SEARCHING
    lost_secs   = 0.0
    fps_times: deque = deque(maxlen=30)
    t_last = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS counter
        now = time.perf_counter()
        fps_times.append(now - t_last)
        t_last = now
        cam_fps_val = len(fps_times) / max(sum(fps_times), 1e-6)

        enhanced = enhancer.enhance(frame)
        result   = pipeline.process_frame(enhanced)

        # Derive simple track state from phase for display
        if result.phase == Phase.LOCKED:
            track_state = TrackState.LOCKED
            lost_secs   = 0.0
        elif result.phase == Phase.SEARCHING:
            lost_secs += 1.0 / max(cam_fps_val, 1.0)
            if track_state != TrackState.SEARCHING:
                track_state = TrackState.SEARCHING
        elif result.phase == Phase.CONFIRMING:
            track_state = TrackState.LOCKED
            lost_secs   = 0.0
        else:
            track_state = TrackState.SEARCHING
            lost_secs   = 0.0

        # Log notable events
        now_str = time.strftime("%H:%M:%S")
        if result.phase == Phase.LOCKED:
            conf = result.identity.confidence if result.identity else 0.0
            name = (result.classification.threat_name
                    if result.classification and result.classification.threat_name
                    else "DRONE")
            log.append(f"{now_str} LOCKED {name} {conf:.0%}")
        elif result.phase == Phase.SEARCHING:
            log.append(f"{now_str} SEARCHING {lost_secs:.1f}s")
        elif result.phase == Phase.CONFIRMING:
            n = result.identity.consecutive if result.identity else 0
            r = result.identity.required    if result.identity else 3
            log.append(f"{now_str} CONF {n}/{r}")

        overlay = draw_camera_overlay(frame, result, track_state, lost_secs)
        panel   = build_status_panel(result, track_state, lost_secs, cam_fps_val, log)

        combined = np.hstack([overlay, panel])
        cv2.imshow("OmniVision3D", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:   # Q or ESC
            break
        if cv2.getWindowProperty("OmniVision3D", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("e"):
            state = enhancer.toggle()
            print(f"Enhancement: {'ON' if state else 'OFF'}")
        elif key == ord("r"):
            pipeline.reset()
            log.clear()
            track_state = TrackState.SEARCHING
            print("Pipeline reset")
        elif key == ord("s"):
            fname = f"screenshot_{time.time():.0f}.png"
            cv2.imwrite(fname, overlay)
            print(f"Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVision3D test UI")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--video",  default=None)
    args = parser.parse_args()
    run(args)
