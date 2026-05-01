"""
ground_station/main.py — Unified OmniVision3D ground station.

Merges:
  - GPSTest.py          → sensors/gps.py
  - laser777-new.py     → sensors/laser.py + sensors/compass.py + targeting.py
  - laser-button UI     → this file (Tkinter)
  - OmniVision3D YOLO   → detection.py

Threads started here:
  GPS reader     → updates state.lat / state.lon
  Laser reader   → updates state.distance_m
  Compass reader → updates state.compass_raw
  Detection      → updates state.frame / state.phase / state.confidence

UI (main thread, Tkinter):
  Left panel  — live sensor + detection values
  Right panel — camera feed with YOLO overlay
  Bottom      — compass calibration controls + snapshot

Usage:
    python ground_station/main.py \\
        --camera 0 \\
        --model  vision/yolo_dji.onnx \\
        --gps    /dev/serial/by-id/usb-FTDI_...-BG01OJPV-if00-port0 \\
        --laser  /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \\
        --compass /dev/serial/by-id/usb-FTDI_...-AQ045IV6-if00-port0 \\
        --declination 4.0
"""

import argparse
import datetime
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import END
    from PIL import ImageTk, Image
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("ERROR: tkinter or Pillow not installed")
    sys.exit(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from ground_station import shared_state as ss
from ground_station import targeting
from ground_station.sensors  import gps     as gps_sensor
from ground_station.sensors  import laser   as laser_sensor
from ground_station.sensors  import compass as compass_sensor
from ground_station           import detection as det

SNAP_DIR = Path("/home/bravofox/screenshots")

_GREEN  = "#00ff00"
_BLACK  = "black"
_ORANGE = "#ff8800"
_RED    = "#ff3333"
_CYAN   = "#00ffff"
_GREY   = "#888888"

_PHASE_COLOR = {
    "SCANNING":   _GREEN,
    "CONFIRMING": _CYAN,
    "LOCKED":     _RED,
    "SEARCHING":  _ORANGE,
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 6) -> str:
    if val is None:
        return "---"
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)


# ── Main app ──────────────────────────────────────────────────────────────

class GroundStationApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.snap_dir = SNAP_DIR
        self.snap_dir.mkdir(parents=True, exist_ok=True)

        root.title("OmniVision3D Ground Station")
        root.configure(bg=_BLACK)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._start_threads()
        self._poll()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self.root

        # ── Left panel: sensor values ──────────────────────────────────────
        left = tk.Frame(root, bg=_BLACK)
        left.grid(row=0, column=0, padx=12, pady=12, sticky="n")

        tk.Label(left, text="OMNIVISION3D  GROUND STATION",
                 font=("Arial", 16, "bold"), bg=_BLACK, fg=_GREEN
                 ).grid(row=0, column=0, columnspan=2, pady=(0, 10))

        rows = [
            ("time_utc",     "GPS Time"),
            ("lat",          "Latitude"),
            ("lon",          "Longitude"),
            ("compass_raw",  "Compass Raw°"),
            ("compass_used", "Compass Used°"),
            ("distance_m",   "Distance (m)"),
            ("target_lat",   "Target Lat"),
            ("target_lon",   "Target Lon"),
            ("phase",        "AI Phase"),
            ("confidence",   "AI Confidence"),
            ("consecutive",  "Consecutive"),
        ]

        self._entries: dict = {}
        for i, (key, label) in enumerate(rows, start=1):
            tk.Label(left, text=label, font=("Arial", 13),
                     bg=_BLACK, fg=_GREEN, anchor="w", width=16
                     ).grid(row=i, column=0, padx=6, pady=4, sticky="w")
            e = tk.Entry(left, font=("Arial", 13), width=20,
                         bg=_BLACK, fg=_GREEN, insertbackground=_GREEN,
                         state="readonly")
            e.grid(row=i, column=1, padx=6, pady=4, sticky="w")
            self._entries[key] = e

        # ── Compass calibration ────────────────────────────────────────────
        cal = tk.LabelFrame(left, text="Compass Calibration",
                            font=("Arial", 12, "bold"),
                            bg=_BLACK, fg=_GREEN, bd=2, relief="groove")
        cal.grid(row=len(rows)+2, column=0, columnspan=2,
                 padx=4, pady=(16, 4), sticky="w")

        self._offset_var = tk.StringVar(value="0.000000")
        self._cal_status = tk.StringVar(value="")

        tk.Label(cal, text="Offset (°):", font=("Arial", 12),
                 bg=_BLACK, fg=_GREEN).grid(row=0, column=0, padx=8, pady=6)
        tk.Entry(cal, textvariable=self._offset_var, font=("Arial", 12),
                 width=10, bg=_BLACK, fg=_GREEN,
                 insertbackground=_GREEN).grid(row=0, column=1, padx=6)

        btn_cfg = dict(font=("Arial", 11, "bold"), bg="#001a33",
                       fg=_GREEN, width=6)
        tk.Button(cal, text="ZERO",  command=self._cal_zero,
                  **btn_cfg).grid(row=0, column=2, padx=4)
        tk.Button(cal, text="-1.0",  command=lambda: self._cal_step(-1.0),
                  **btn_cfg).grid(row=0, column=3, padx=2)
        tk.Button(cal, text="+1.0",  command=lambda: self._cal_step(+1.0),
                  **btn_cfg).grid(row=0, column=4, padx=2)
        tk.Button(cal, text="-0.1",  command=lambda: self._cal_step(-0.1),
                  **btn_cfg).grid(row=0, column=5, padx=2)
        tk.Button(cal, text="+0.1",  command=lambda: self._cal_step(+0.1),
                  **btn_cfg).grid(row=0, column=6, padx=2)

        tk.Label(cal, textvariable=self._cal_status,
                 font=("Arial", 11), bg=_BLACK, fg=_GREEN
                 ).grid(row=1, column=0, columnspan=7, padx=8, pady=(0, 6),
                        sticky="w")

        # Snapshot button
        tk.Button(left, text="SNAPSHOT",
                  font=("Arial", 14, "bold"),
                  bg="#003300", fg=_GREEN, width=18, height=2,
                  command=self._snapshot
                  ).grid(row=len(rows)+3, column=0, columnspan=2,
                         padx=4, pady=12)

        # ── Right panel: camera ────────────────────────────────────────────
        self._cam_label = tk.Label(root, bg=_BLACK)
        self._cam_label.grid(row=0, column=1, padx=12, pady=12)
        self._cam_photo = None

    # ── Thread startup ─────────────────────────────────────────────────────

    def _start_threads(self) -> None:
        a = self.args

        if a.gps:
            threading.Thread(
                target=gps_sensor.thread_gps,
                args=(a.gps,),
                kwargs={"baud": 9600, "gpsdata_path": "/home/bravofox/gpsdata.txt"},
                daemon=True,
            ).start()
            print(f"[Main] GPS thread started on {a.gps}")
        else:
            print("[Main] No GPS port specified — GPS disabled")

        if a.laser:
            threading.Thread(
                target=laser_sensor.thread_laser,
                args=(a.laser,),
                kwargs={"period": 0.02, "scale": a.laser_scale},
                daemon=True,
            ).start()
            print(f"[Main] Laser thread started on {a.laser}")
        else:
            print("[Main] No laser port — laser disabled")

        if a.compass:
            threading.Thread(
                target=compass_sensor.thread_compass,
                args=(a.compass,),
                daemon=True,
            ).start()
            print(f"[Main] Compass thread started on {a.compass}")
        else:
            print("[Main] No compass port — compass disabled")

        threading.Thread(
            target=det.thread_detection,
            args=(a.camera, str(_ROOT / a.model)),
            kwargs={"conf_threshold": 0.50, "consecutive_required": 3},
            daemon=True,
        ).start()
        print(f"[Main] Detection thread started  camera={a.camera}")

    # ── Polling update (every 100 ms) ──────────────────────────────────────

    def _poll(self) -> None:
        with ss.lock():
            s = ss.state

        # Update target coords from latest sensor readings
        targeting.update_target(self.args.declination)

        # Refresh text entries
        vals = {
            "time_utc":     s.time_utc or "---",
            "lat":          _fmt(s.lat),
            "lon":          _fmt(s.lon),
            "compass_raw":  _fmt(s.compass_raw, 1),
            "compass_used": _fmt(s.compass_used, 1),
            "distance_m":   _fmt(s.distance_m, 1),
            "target_lat":   _fmt(s.target_lat),
            "target_lon":   _fmt(s.target_lon),
            "phase":        s.phase,
            "confidence":   f"{s.confidence:.0%}" if s.confidence else "0%",
            "consecutive":  str(s.consecutive),
        }
        for key, val in vals.items():
            e = self._entries[key]
            color = _PHASE_COLOR.get(val, _GREEN) if key == "phase" else _GREEN
            e.configure(state="normal", fg=color)
            e.delete(0, END)
            e.insert(0, val)
            e.configure(state="readonly")

        # Compass offset display
        self._offset_var.set(f"{targeting.read_offset():.6f}")

        # Camera frame from detection thread
        frame = s.frame
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._cam_label.configure(image=img)
            self._cam_label.image = img   # keep reference

        self.root.after(100, self._poll)

    # ── Calibration callbacks ──────────────────────────────────────────────

    def _cal_zero(self) -> None:
        with ss.lock():
            c_used = ss.state.compass_used
            c_raw  = ss.state.compass_raw

        if c_used is not None:
            new_off = targeting.read_offset() - c_used
            targeting.write_offset(new_off)
            self._cal_status.set("Zero set using compass_used")
        elif c_raw is not None:
            targeting.write_offset(-(c_raw + self.args.declination))
            self._cal_status.set("Zero set using raw compass")
        else:
            self._cal_status.set("No compass data yet")

    def _cal_step(self, delta: float) -> None:
        targeting.write_offset(targeting.read_offset() + delta)
        self._cal_status.set(f"Offset adjusted {delta:+.1f}°")

    # ── Snapshot ──────────────────────────────────────────────────────────

    def _snapshot(self) -> None:
        with ss.lock():
            frame = ss.state.frame
            s     = ss.state

        if frame is None:
            return

        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = self.snap_dir / f"shot_{ts}.jpg"
        cv2.imwrite(str(img_path), frame)

        txt_path = self.snap_dir / f"shot_{ts}.txt"
        txt_path.write_text(
            f"time={s.time_utc}\n"
            f"lat={s.lat}\nlon={s.lon}\n"
            f"compass_raw={s.compass_raw}\ncompass_used={s.compass_used}\n"
            f"distance_m={s.distance_m}\n"
            f"target_lat={s.target_lat}\ntarget_lon={s.target_lon}\n"
            f"phase={s.phase}\nconfidence={s.confidence:.3f}\n"
        )
        print(f"[Snapshot] saved {img_path}")

    # ── Close ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="OmniVision3D Ground Station")
    ap.add_argument("--camera",      type=int,   default=0)
    ap.add_argument("--model",       default="vision/yolo_dji.onnx")
    ap.add_argument("--gps",         default=None,
                    help="GPS serial port (skip to disable)")
    ap.add_argument("--laser",       default=None,
                    help="Laser serial port (skip to disable)")
    ap.add_argument("--compass",     default=None,
                    help="Compass serial port (skip to disable)")
    ap.add_argument("--declination", type=float, default=0.0,
                    help="Magnetic declination to add to compass (deg)")
    ap.add_argument("--laser-scale", type=float, default=1.0,
                    dest="laser_scale",
                    help="Laser distance calibration multiplier")
    args = ap.parse_args()

    root = tk.Tk()
    GroundStationApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
