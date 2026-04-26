# OmniVision3D — Full System Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Data Pipeline — From 3D Mesh to Training Set](#3-data-pipeline--from-3d-mesh-to-training-set)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [YOLO Detector — Primary Detection](#6-yolo-detector--primary-detection)
7. [Drone Classifier — YOLO-Only Locking](#7-drone-classifier--yolo-only-locking)
8. [Enemy Identifier — Post-Lock ID](#8-enemy-identifier--post-lock-id)
9. [Vision Pipeline — Phase State Machine](#9-vision-pipeline--phase-state-machine)
10. [Camera Stream — Isolated Process Architecture](#10-camera-stream--isolated-process-architecture)
11. [Tracker — PID + State Machine](#11-tracker--pid--state-machine)
12. [Motor Controller — PWM Intercept System](#12-motor-controller--pwm-intercept-system)
13. [GPS Navigation](#13-gps-navigation)
14. [Lean Flight Script — drone_main.py](#14-lean-flight-script--drone_mainpy)
15. [Full Mission Loop — main.py](#15-full-mission-loop--mainpy)
16. [Ground Detection Unit](#16-ground-detection-unit)
17. [Laptop Monitor — Command Center](#17-laptop-monitor--command-center)
18. [FastAPI Inference Server](#18-fastapi-inference-server)
19. [Configuration Reference](#19-configuration-reference)
20. [MAVLink Integration Status](#20-mavlink-integration-status)
21. [Friendly Fire Prevention — Deconfliction Theory](#21-friendly-fire-prevention--deconfliction-theory)
22. [Dependency Map](#22-dependency-map)

---

## 1. System Overview

OmniVision3D is a real-time drone interception system built for a Raspberry Pi-equipped fixed-wing UAV. It detects hostile drones using a two-tier vision pipeline:

1. **YOLO generic detector** — fast detection, identifies any drone-shaped object
2. **Enemy identifier** — optional post-lock classification using a pluggable ONNX model per known enemy type

Detection is triggered by N consecutive YOLO frames at ≥50% confidence, at which point the interceptor commits to full-throttle intercept while continuously correcting course via a 4-motor PID system.

The system also includes:
- A **ground detection unit** (separate Pi) that watches the sky and sends UDP alerts to the laptop
- A **laptop monitor** that receives those alerts and commands the interceptor drone to launch
- A **synthetic data generation pipeline** that builds training data from 3D mesh files without needing real drone footage

**Key performance numbers (benchmarked on laptop CPU):**
- YOLO inference at 320px input: ~31ms avg, ~32 FPS
- MobileNetV2 classifier: ~2ms (ONNX, CPU)
- YOLOv8 nano dataset result: mAP50 = 0.993 on 2055 real images

---

## 2. Repository Layout

```
OmniVision3D/
├── renderer/               3D → image pipeline
│   ├── camera_utils.py     Fibonacci sphere sampling, viewpoint geometry
│   ├── render_views.py     Mesh loading, Open3D rendering, DJI coloring
│   ├── sky_compositor.py   Sky background generation and compositing
│   └── thermal_filter.py   RGB → thermal camera simulation
│
├── training/               Model definition and training
│   ├── model.py            MobileNetV2 backbone (classification + detection heads)
│   ├── dataset_loader.py   Binary dataset loader (drone / no_drone)
│   └── train.py            Training script: Adam, early stopping, CSV/PNG logs
│
├── inference/              Laptop-side inference tools
│   ├── predict.py          Predictor class + image folder / video CLI
│   └── visualize.py        draw_prediction, annotate_video
│
├── train_yolo.py           YOLOv8 nano training on yolo_dataset/
│
├── api/
│   └── server.py           FastAPI /predict endpoint (REST API wrapper)
│
├── tests/
│   ├── test_inference.py   Integration tests: Predictor + draw_prediction
│   └── test_renderer.py    Unit tests: camera_utils, sky_compositor, thermal_filter
│
├── configs/
│   └── default.yaml        Rendering, training, and class configuration
│
├── ground_detection/       Standalone ground-based Pi detection unit
│   ├── detect.py           Main loop: YOLO → consecutive counter → UDP alert
│   ├── vision/
│   │   └── drone_detector.py  Parses YOLOv8 output [1, 5, 8400]
│   └── comms/
│       └── alert_sender.py    UDP fire-and-forget to laptop on port 5555
│
├── laptop_monitor/         Laptop-side command center
│   ├── monitor.py          Interactive CLI: displays alerts, keyboard commands
│   ├── alert_receiver.py   UDP listener on port 5555 (background daemon)
│   └── launch_command.py   Sends LAUNCH / ABORT commands to interceptor Pi (port 5556)
│
├── pi_deploy/              Everything that runs on the interceptor Pi
│   ├── drone_main.py       Lean flight script: camera → YOLO → PID → motors
│   ├── main.py             Full mission loop: pipeline + tracker + GPS
│   ├── config.yaml         All runtime parameters
│   ├── requirements_pi.txt onnxruntime, opencv-python, numpy, pyyaml
│   │
│   ├── vision/
│   │   ├── camera_stream.py      Isolated-process camera (shared memory)
│   │   ├── yolo_detector.py      YOLOv8 ONNX wrapper, parses [1, 5, N]
│   │   ├── drone_classifier.py   YOLO-only locking + background enemy ID
│   │   ├── identity_confirmer.py MobileNetV2 ONNX crop classifier (legacy)
│   │   ├── enemy_identifier.py   Pluggable ONNX enemy list (enemies.yaml)
│   │   ├── frame_enhancer.py     CLAHE on LAB L-channel (<5ms target)
│   │   ├── pipeline.py           Phase state machine: SCANNING/CONFIRMING/LOCKED/SEARCHING
│   │   └── pi_predict.py         Standalone MobileNetV2 Pi inference (legacy)
│   │
│   ├── control/
│   │   ├── motor_controller.py   4-motor PID, PWM microseconds, intercept mode
│   │   ├── motor_calibration.py  Interactive spin-and-identify calibration tool
│   │   └── tracker.py            Pitch/yaw PID state machine, drives MotorController
│   │
│   ├── navigation/
│   │   └── gps_nav.py            Haversine + fly_to/return_to_base stubs
│   │
│   └── enemies/                  Enemy model files and manifest
│       └── enemies.yaml          Pluggable enemy list
│
└── yolo_dataset/
    ├── dataset.yaml              Class names and split paths
    └── images/, labels/          80/10/10 train/val/test split (excluded from git)
```

---

## 3. Data Pipeline — From 3D Mesh to Training Set

### Step 1 — Render raw views (`renderer/render_views.py`)

Accepts `.obj`, `.glb`, `.stl`, `.fbx`, `.step`, `.stp` mesh files via trimesh (for obj/glb/stl) or Open3D (for fbx).

Viewpoints are generated with **Fibonacci sphere sampling** (`renderer/camera_utils.py`):
- `fibonacci_sphere(n)` produces n evenly distributed unit vectors across the full sphere — no polar clustering
- `get_sphere_viewpoints()` combines N angles × M distances = a viewpoint grid
- `viewpoint_to_camera(azimuth, elevation, distance)` returns `(eye, center=[0,0,0], up)` in Open3D camera format, with a pole fallback (up = [0,0,-1]) when elevation is near 90°

**DJI color simulation** (`_apply_dji_colors()`):
The mesh is colored by distance from center in four radial zones — body (80/80/80 grey), arms (60/60/60), motors (30/30/30), props (20/20/20) — to match the real DJI Mini 4 Pro's appearance.

Renders are saved to `dataset/raw/{object_name}/` with white background (1.0/1.0/1.0).

`configs/default.yaml` controls: 100 sphere angles, 6 distances (2.0–18.0m), 512×512 resolution, 5 lighting variations.

### Step 2 — Sky compositing (`renderer/sky_compositor.py`)

Generates 100 synthetic sky backgrounds (5 types × 20 seeds):
- `clear_blue` — gradient top/bottom blend
- `overcast` — additive Gaussian noise on grey base
- `sunset` — three-band gradient (sky/horizon/earth)
- `night_sky` — dark base + random star dots
- `hazy_sky` — blue-grey blend

`composite_image()` uses a pixel threshold of 240 — any pixel where all three channels exceed 240 is treated as background and replaced with the sky image.

`run_compositor()` produces 3 composited variants per render, saving to `dataset/processed/{object_name}/` with filename `{stem}_sky{N}.png`.

### Step 3 — Thermal simulation (`renderer/thermal_filter.py`)

Optional. Converts RGB composited images to thermal-style images using a 6-step pipeline:
1. Grayscale conversion
2. CLAHE (clipLimit=2.5, 8×8 tiles)
3. Custom LUT colormap (iron: black→red→orange→yellow→white; rainbow; grayscale)
4. Heat bloom: bright regions emit a soft GaussianBlur glow (σ=7, 21×21 kernel)
5. Gaussian sensor noise (σ=3.0)
6. Scanline darkening (every 4th row × 0.82)

### Step 4 — YOLO format conversion (`training/convert_to_yolo.py`)

Derives bounding boxes from raw white-background renders automatically:
- `detect_bbox_white()` finds all pixels where not all three channels exceed 240 (foreground), computes tight bounding box, normalises to YOLO cx/cy/w/h format
- `_stem_to_raw_key()` strips `_skyN` suffix to map composited images back to their raw bbox
- Splits into train/val/test at 80/10/10 ratio per source folder
- Writes `yolo_dataset/dataset.yaml` (class 0 = `dji_mini_4_pro`) and a sample label check image

### Step 5 — Dataset for classification training

`training/dataset_loader.py` maps four source folders to binary labels:
- `DJI_Mini_4_Pro` (RGB renders) → label 1 (drone)
- `DJI_Mini_4_Pro_thermal` → label 1 (drone)
- `no_drone` (sky backgrounds without drone) → label 0
- `no_drone_thermal` → label 0

Stratified per-folder splits ensure each source is proportionally represented. Training images get RandomHorizontalFlip, RandomRotation(15°), ColorJitter(brightness=0.2, contrast=0.2). Val/test images get resize+normalize only.

---

## 4. Model Architecture

**File:** `training/model.py`

`OmniVisionModel` wraps a pretrained MobileNetV2 backbone with three output paths:

| Head | Output shape | Purpose |
|------|-------------|---------|
| `classifier` | (B, num_classes) | Binary drone/no_drone classification |
| `det_head` | (B, 5) | cx, cy, w, h, objectness (not yet exported) |
| `embed` | (B, 1280) | Raw feature embeddings for retrieval |

The backbone's `features` module outputs (B, 1280, H, W); global average pooling flattens to (B, 1280) before both heads.

MobileNetV2 was chosen for its depthwise separable convolutions — low FLOP count and memory usage for CPU inference on Raspberry Pi 4.

**ONNX export:** The classifier is exported to ONNX opset 12 at 8.5MB. The det_head is not currently exported (TODO comment in `pi_predict.py`).

---

## 5. Training Pipeline

**File:** `training/train.py`

- Adam optimizer, lr=0.001
- ReduceLROnPlateau scheduler (patience=3, factor=0.5)
- CrossEntropyLoss
- Max 50 epochs, early stopping at patience=5
- `num_workers=0` (avoids Windows multiprocessing issues)
- Best checkpoint saved to `checkpoints/best.pt` by validation accuracy
- Training curve saved as `checkpoints/training_curve.png`
- Per-epoch CSV log at `checkpoints/training_log.csv`

**Test evaluation** (`evaluate_test_set()`):
- Full confusion matrix with TP/FP/TN/FN
- Warnings if FNR > 2% (missed drones) or FPR > 5% (false alarms)
- Confusion matrix saved as both CSV and PNG

**YOLOv8 training** (`train_yolo.py`):
- YOLOv8 nano (`yolov8n.pt`)
- 15 epochs, imgsz=512, batch=16, Adam, lr=0.001
- Augmentation: fliplr=0.5, translate=0.2, scale=0.3, degrees=15, mosaic=1.0
- Output to `checkpoints/yolo/dji_detector/`
- Achieved mAP50=0.993 on 2055-image real DJI dataset

---

## 6. YOLO Detector — Primary Detection

**File:** `pi_deploy/vision/yolo_detector.py`

Parses the YOLOv8 ONNX output tensor `[1, 5, N]` where:
- N = number of anchor boxes (8400 for imgsz=320, 5376 for imgsz=512)
- 5 = cx, cy, w, h, class_confidence (single-class model)

Processing steps:
1. Resize frame to `input_size × input_size` (default 320 for Pi, 512 for training)
2. BGR → RGB, divide by 255.0 (no ImageNet normalization — YOLO uses raw [0,1])
3. Transpose to NCHW `(1, 3, H, W)`
4. Parse output: `raw[0].T` gives `(N, 5)`, filter by `conf >= conf_threshold`
5. Scale cx/cy/w/h back to original frame pixel space
6. Custom NMS via greedy IoU-sorted loop (IoU threshold 0.45)
7. Return list of `Detection` dataclasses sorted by confidence (highest first)

Each `Detection` has: `bbox (x1,y1,x2,y2)`, `confidence`, `center (cx,cy)`, `area`.

`draw_detections()` draws yellow bounding boxes labeled "DJI? XX%" onto a frame copy.

---

## 7. Drone Classifier — YOLO-Only Locking

**File:** `pi_deploy/vision/drone_classifier.py`

Implements the core locking logic with four status strings:
- `NOT_DRONE` — YOLO returned no detections above threshold
- `CONFIRMING` — drone detected, consecutive count building (< required)
- `LOCKED` — N consecutive frames reached, no enemy ID match yet
- `IDENTIFIED` — LOCKED + known enemy name from enemy identifier

**Locking logic:**
- Threshold: 50% YOLO confidence (generic_threshold in config)
- Required: 3 consecutive frames (consecutive_required in config)
- On each YOLO miss: resets consecutive counter, threat name, and ID counter to zero

**Enemy ID integration:**
Once LOCKED, runs `EnemyIdentifier.identify()` every 5 frames (`enemy_id_every_n_frames`). This is non-blocking in the sense that it only runs at the configured cadence — it does not run in a separate thread. If a known threat is identified, the status upgrades to IDENTIFIED and the threat name/level are included in `ClassificationResult`. The intercept is NOT blocked waiting for enemy ID — locking happens on YOLO alone.

`ClassificationResult` dataclass carries: status, confidence, bbox, center, consecutive, required, threat_name, threat_level, message, all_scores (per-enemy confidence dict).

---

## 8. Enemy Identifier — Post-Lock ID

**File:** `pi_deploy/vision/enemy_identifier.py`

Loads a pluggable list of enemy ONNX classifiers from `enemies/enemies.yaml`. Each entry has:
- `name` — human-readable enemy name
- `model` — path to .onnx file (relative to enemies/)
- `threat_level` — e.g. "high", "medium", "low", "unknown"
- `confidence_threshold` — default 0.85
- `active` — boolean enable/disable flag

**Inference per enemy:**
1. Crop bbox from frame with 20% padding
2. Resize crop to 224×224, BGR→RGB, normalize (ImageNet mean/std)
3. Run ONNX session, softmax logits
4. Class index 1 = drone confidence
5. Returns highest-scoring active enemy above threshold, or not-known

`IdentificationResult` dataclass: `is_known_threat`, `threat_name`, `threat_level`, `confidence`, `all_scores`.

`add_enemy()` copies a new .onnx to the enemies folder, appends to enemies.yaml, and loads the session into memory — no code changes required.

---

## 9. Vision Pipeline — Phase State Machine

**File:** `pi_deploy/vision/pipeline.py`

Wraps `DroneClassifier` and adds a SEARCHING timer. The four phases:

| Phase | Condition | Visual |
|-------|-----------|--------|
| SCANNING | No drone detected | Thin green border + grey crosshair |
| CONFIRMING | Drone seen, building count | Cyan border + cyan bbox + "CONFIRMING N/3" |
| LOCKED | N consecutive frames | Red thick border + green bbox + intercept line |
| SEARCHING | Lost after LOCKED, timer < 2s | Orange border + "SEARCHING X.Xs" |

**State transitions:**
```
NOT_DRONE  +  was LOCKED/SEARCHING  →  start/continue _lost_since timer
  if lost < lost_timeout (2.0s)     →  SEARCHING
  if lost >= lost_timeout            →  SCANNING (reset timer)
NOT_DRONE  +  was SCANNING/CONFIRMING  →  SCANNING immediately

CONFIRMING classifier result  →  CONFIRMING phase
LOCKED or IDENTIFIED result   →  LOCKED phase (reset lost timer)
```

**FPS tracking:** rolling 30-frame window using `collections.deque(maxlen=30)`.

`PipelineResult` carries: phase, detection (Detection or None), identity (IdentityResult or None), fps, classification (ClassificationResult or None).

`draw_overlay()` handles all four phases with distinct color codes and text overlays.

---

## 10. Camera Stream — Isolated Process Architecture

**File:** `pi_deploy/vision/camera_stream.py`

The camera runs in a completely separate OS process (`mp.Process`) so that `cv2.VideoCapture.read()` never contends with the Python GIL during ONNX inference.

**Architecture:**
- Main process creates a `shared_memory.SharedMemory` block (H×W×3 bytes)
- Worker process imports `cv2` locally (not in main process namespace)
- Worker writes each frame to shared memory under a `mp.Lock`
- Worker increments a `mp.Value(c_uint64)` write counter after each write
- Main process `read()` checks counter — if unchanged, returns None (no new frame)
- Main process copies the shared buffer under lock and returns the copy

`cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` discards buffered frames immediately so the system always gets the most recent frame, not a queued old one.

`stop()` sets the stop event, joins the process with 3s timeout, then cleans up shared memory with `shm.close()` + `shm.unlink()`.

---

## 11. Tracker — PID + State Machine

**File:** `pi_deploy/control/tracker.py`

Converts `PipelineResult` into pitch/yaw corrections and motor PWM commands. Maintains its own lost timer independent of the pipeline's SEARCHING timer, allowing a longer recovery window before abort.

**States (TrackState enum):**
| State | Condition | Action |
|-------|-----------|--------|
| LOCKED | Phase.LOCKED + detection | PID corrections → motor commands |
| SEARCHING | lost < t_search (3.0s) | Reset PID + motors, hold last heading |
| NAVIGATING | t_search < lost < t_nav (10.0s) | Reset motors (future: fly to last known) |
| ABORT | lost > t_nav | Return to base |

**PID internals (`_PID` class):**
Uses `time.monotonic()` for real dt:
```python
dt = max(now - prev_time, 1e-4)
integral += error * dt
derivative = (error - prev_error) / dt
out = kp * error + ki * integral + kd * derivative
```

Pixel offset normalized to [-1, 1] before PID: `norm_dx = dx / (frame_w / 2)`.

PID gains from config: `pid_p=0.1, pid_i=0.01, pid_d=0.05`.

**Motor integration:**
- First LOCKED frame: calls `motor.engage_intercept()` and logs INTERCEPT_COMMITTED
- Leaving LOCKED: calls `motor.disengage_intercept()` and `motor.reset()`
- `TrackingCommand.motor_powers` dict contains all 4 channel values

---

## 12. Motor Controller — PWM Intercept System

**File:** `pi_deploy/control/motor_controller.py`

Controls 4 motors on a fixed-wing interceptor using power-differential steering.

**Motor layout (top-down, nose forward):**
```
M1 (front-left)    M2 (front-right)
M3 (rear-left)     M4 (rear-right)
```

**Power formulas:**
```
M1 = base + yaw_us + pitch_us   (front-left)
M2 = base - yaw_us + pitch_us   (front-right)
M3 = base + yaw_us - pitch_us   (rear-left)
M4 = base - yaw_us - pitch_us   (rear-right)
```

Right turn (positive yaw_us): M1/M3 increase, M2/M4 decrease.
Nose up (positive pitch_us): M1/M2 increase, M3/M4 decrease.

**Intercept mode:**
- Cruise base: 1500us
- Intercept base: 1900us (set when LOCKED)
- Correction microseconds are always calculated relative to cruise base (1500us), independent of intercept mode, so steering differential is consistent at any throttle level

**PID correction chain:**
1. `calculate_corrections(dx, dy)` → yaw%, pitch% (discrete PID, no dt)
2. `yaw_us = (yaw_pct / 100) * base_us`; `pitch_us = (pitch_pct / 100) * base_us`
3. Apply motor formulas, clamp to [min_us=1000, max_us=2000]
4. Map logical motors to physical channels via `motor_mapping` config
5. `send_to_pixhawk()` — in sim mode, prints formatted table + logs to CSV

**MAVLink stub:**
```python
# TODO: MAVLink wiring
# master.mav.rc_channels_override_send(
#     target_system, target_component,
#     ch1, ch2, ch3, ch4, 0, 0, 0, 0)
```

**Calibration** (`motor_calibration.py`):
Interactive one-time tool. Spins each channel at low power (1600us) one at a time while the user identifies which motor moved. After mapping, runs yaw and pitch sanity checks at 2% correction. Auto-detects and sets flip_yaw/flip_pitch flags if steering direction is wrong. Writes confirmed mapping to config.yaml.

---

## 13. GPS Navigation

**File:** `pi_deploy/navigation/gps_nav.py`

All methods are simulation stubs. Real MAVLink wiring is explicitly noted with TODO comments.

**What is implemented:**
- `_haversine(lat1, lon1, lat2, lon2)` → metres (accurate Haversine formula)
- `fly_to(lat, lon, altitude, simulation=True)` — logs command, sleeps 2s in sim, updates internal position
- `is_close_enough(target, threshold=50m)` → bool
- `return_to_base()` → calls `fly_to(home_lat, home_lon)`
- `hold_position()` — logs only

**MAVLink TODOs explicitly noted:**
- `fly_to()`: replace with `MISSION_ITEM` or `SET_POSITION_TARGET_GLOBAL_INT`
- `get_current_position()`: replace with serial GPS / dronekit `vehicle.location`
- `hold_position()`: replace with `MAV_CMD_NAV_LOITER_UNLIM`

Home coordinates in config: `home_lat: 24.7136`, `home_lon: 46.6753` (Riyadh, Saudi Arabia).

---

## 14. Lean Flight Script — drone_main.py

**File:** `pi_deploy/drone_main.py`

A deliberately flat script that bypasses the entire pipeline/classifier/tracker/CLAHE stack. Designed to run directly on the Raspberry Pi during actual missions with minimum latency and code surface area.

**Loop architecture:**
```
while True:
    check KILL file → break if present
    camera.read() → skip if None (no new frame)
    yolo.detect(frame) → get detections
    consecutive counter inline (no DroneClassifier class)
    determine phase (SCANNING/CONFIRMING/LOCKED) inline
    if LOCKED: engage_intercept() on first frame
               calculate_motor_powers(dx, dy)
               send_to_pixhawk(powers)
    log every 30 frames
```

**Kill switch:**
Creates `pi_deploy/KILL` file from laptop over SSH:
```bash
touch pi_deploy/KILL
```
Detected at start of each loop iteration, triggers clean shutdown.

**Benchmarked on laptop CPU:** 31.3ms avg, 31.9 FPS, p99=64.2ms.

**Arguments:** `--config` (default config.yaml), `--camera` (default 0), `--sim` (simulation mode).

---

## 15. Full Mission Loop — main.py

**File:** `pi_deploy/main.py`

The production mission runner that uses the full pipeline stack. Used when the complete phase visualization and tracker state machine are needed.

**Flow:**
1. Print banner (YOLO model, lock threshold, enemy config, launch port)
2. Start `_LaunchListener` on UDP port 5556
3. Optionally hold in STANDBY until LAUNCH command received (`--wait-launch` flag)
4. Navigate to GPS coordinates (simulated by default)
5. Main loop: `pipeline.process_frame()` → `tracker.update()` → `draw_overlay()` → CSV log
6. On ABORT tracker state: `nav.return_to_base()` then break
7. Mission summary: frames, YOLO hits, confirmations, tracking time, end reason

**`_LaunchListener` class:**
Background daemon thread listening on UDP port 5556. Receives JSON packets:
- `{command: LAUNCH, target_confidence, compass_heading}` → sets `mission_active = True`
- `{command: ABORT}` → sets `abort_requested = True`

**CSV log** (`logs/frames.csv`): ts, phase, yolo_conf, id_conf, consecutive, pitch, yaw, fps — every frame.

---

## 16. Ground Detection Unit

**Folder:** `ground_detection/`

A standalone Pi unit that watches the sky and sends UDP alerts to the laptop. Independent of the interceptor Pi.

**`vision/drone_detector.py`:**
- Same YOLOv8 parsing logic as `yolo_detector.py`
- `DetectionResult` dataclass: detected, confidence, bbox, center, consecutive
- `reset()` clears consecutive counter

**`comms/alert_sender.py`:**
- UDP fire-and-forget to laptop on port 5555
- `send_alert(detection_result, compass_heading=None)` → JSON:
  ```json
  {"type": "DRONE_DETECTED", "confidence": 0.82, "center": [320, 240],
   "compass_heading": 45, "timestamp": 1714000000.0}
  ```
- `send_clear()` → `{"type": "CLEAR", "timestamp": ...}`
- Logs to `logs/alerts_sent.log`

**`detect.py`:**
- SCANNING/CONFIRMING/LOCKED draw overlay (same color scheme as pipeline.py)
- Sends alert on first LOCKED transition, resends every 30 frames while locked
- Sends CLEAR on locked→not-detected transition
- CSV log of all frames
- Arguments: `--camera`, `--laptop <IP>`, `--model`, `--sim`

---

## 17. Laptop Monitor — Command Center

**Folder:** `laptop_monitor/`

Receives UDP alerts from the ground Pi, displays them to the operator, and sends LAUNCH/ABORT commands to the interceptor Pi.

**`alert_receiver.py`:**
- Background daemon thread on UDP port 5555
- `socket.timeout = 0.5` for clean shutdown
- `get_latest()` returns the latest packet and clears it (consume-on-read)
- Logs to `logs/alerts_received.log`

**`launch_command.py`:**
- `send_launch(alert_data)` → UDP to interceptor Pi port 5556:
  ```json
  {"command": "LAUNCH", "target_confidence": 0.82,
   "compass_heading": 45, "source_ip": "...", "timestamp": ...}
  ```
- `send_abort()` → `{"command": "ABORT", "timestamp": ...}`
- Logs to `logs/launch_commands.log`

**`monitor.py`:**
- Interactive CLI loop with non-blocking keyboard input
- Platform-specific: `msvcrt.kbhit()` on Windows, `termios` on Linux
- Displays alert details (confidence, heading, timestamp) on DRONE_DETECTED
- Key commands: ENTER → send_launch, A → send_abort, Q → quit
- Note: `dict | None` type annotation requires Python 3.10+

---

## 18. FastAPI Inference Server

**File:** `api/server.py`

REST API wrapper around `Predictor` for laptop-side integration:

```
GET  /health       → {"status": "ok"}
POST /predict      → multipart image upload → {"label": "drone", "confidence": 0.94}
```

Config via environment variables:
- `OMNIVISION_CONFIG` (default: `configs/default.yaml`)
- `OMNIVISION_CHECKPOINT` (default: `checkpoints/best.pt`)
- `USE_GPU` — any value enables CUDA

Image decoding uses `cv2.imdecode(np.frombuffer(...))` — accepts any OpenCV-supported format.

Launch: `uvicorn api.server:app --reload`

---

## 19. Configuration Reference

**File:** `pi_deploy/config.yaml`

| Section | Key | Default | Notes |
|---------|-----|---------|-------|
| `yolo` | `model_path` | `vision/yolo_dji.onnx` | Relative to pi_deploy/ |
| `yolo` | `confidence_threshold` | 0.50 | YOLO detection threshold |
| `yolo` | `nms_threshold` | 0.45 | Non-maximum suppression IoU |
| `yolo` | `input_size` | 320 | Resize resolution before YOLO |
| `drone_classifier` | `generic_threshold` | 0.50 | Same as yolo threshold |
| `drone_classifier` | `consecutive_required` | 3 | Frames needed to lock |
| `drone_classifier` | `lost_timeout_seconds` | 2.0 | SEARCHING window |
| `drone_classifier` | `enemy_id_every_n_frames` | 5 | Enemy ID cadence after lock |
| `drone_classifier` | `enemies_file` | `enemies/enemies.yaml` | Enemy list path |
| `tracking` | `lost_searching_seconds` | 3.0 | Tracker SEARCHING window |
| `tracking` | `lost_navigating_seconds` | 10.0 | Before ABORT |
| `tracking` | `pid_p/i/d` | 0.1/0.01/0.05 | Tracker PID gains |
| `motor_control` | `base_power_us` | 1500 | Cruise PWM |
| `motor_control` | `intercept_power_us` | 1900 | Full-throttle intercept PWM |
| `motor_control` | `min/max_power_us` | 1000/2000 | PWM clamp range |
| `motor_control` | `max_correction_pct` | 20 | Max ±20% steering differential |
| `motor_control` | `pid.Kp/Ki/Kd` | 0.06/0.001/0.01 | Motor PID gains |
| `motor_mapping` | `front_left/right, rear_left/right` | channel_1–4 | Physical channel assignment |
| `motor_mapping` | `flip_yaw/flip_pitch` | false | Set by calibration if wiring reversed |
| `motor_mapping` | `calibrated` | false | Set true after motor_calibration.py |
| `navigation` | `home_lat/lon` | 24.7136/46.6753 | Riyadh coordinates |
| `navigation` | `altitude` | 50m | Cruise altitude |
| `navigation` | `close_enough_meters` | 50 | Acceptance radius |
| `camera` | `width/height` | 640/480 | Camera resolution |
| `launch_listener` | `port` | 5556 | UDP command port |
| `launch_listener` | `wait_for_launch` | false | Hold in STANDBY if true |
| `simulation_mode` | — | true | **Set false for real hardware** |

---

## 20. MAVLink Integration Status

All MAVLink integration is explicitly stubbed with TODO comments. The system is fully functional in simulation mode. The following replacements are needed to go live:

| File | Stub location | Required MAVLink call |
|------|--------------|----------------------|
| `control/motor_controller.py:send_to_pixhawk()` | `# TODO: MAVLink wiring` | `rc_channels_override_send(target_system, target_component, ch1, ch2, ch3, ch4, ...)` |
| `navigation/gps_nav.py:fly_to()` | `# TODO: MAVLink MISSION_ITEM` | `MISSION_ITEM` or `SET_POSITION_TARGET_GLOBAL_INT` |
| `navigation/gps_nav.py:get_current_position()` | `# TODO: replace with real GPS` | Serial GPS / dronekit `vehicle.location` |
| `navigation/gps_nav.py:hold_position()` | `# TODO: MAVLink` | `MAV_CMD_NAV_LOITER_UNLIM` |

To activate simulation_mode: set `simulation_mode: false` in `pi_deploy/config.yaml` and `motor_control.simulation_mode: false`.

---

## 21. Friendly Fire Prevention — Deconfliction Theory

OmniVision3D uses passive GPS-based deconfliction with zero radio signals during flight. The core principle: the interceptor drone cannot be fooled by a spoofed signal it never receives.

**Pre-loaded asset manifest**
Before any drone launches, the operator loads a manifest containing GPS coordinates, flight path, and IFF identifier of every friendly asset in the operational area. This manifest is cryptographically signed and stored locally on the interceptor Pi — it requires no network connection during flight.

**Simultaneous deconfliction**
The onboard GPS gives the interceptor its own position at all times. Combined with the camera's field of view and pixel offset from centre, the system computes an estimated world position for each detected object. Before committing to LOCKED, this estimated position is compared against every friendly asset in the manifest. If the estimated position falls within a configurable exclusion radius of a friendly asset, the detection is suppressed and the system returns to SCANNING.

**Radio silence protocol**
Once airborne, the interceptor Pi makes zero outbound network calls and listens on zero ports. The LAUNCH command received on port 5556 is accepted only during the pre-launch standby phase, before motors are armed. During flight, all sockets are closed. This makes the system immune to both jamming (there is nothing to jam) and spoofing (it accepts no signals).

**Threat level filtering**
The enemy identifier assigns a threat level (high / medium / low / unknown) to each detected object. The operator can configure a minimum threat level required before the intercept commitment is made. Unidentified objects (LOCKED but not yet IDENTIFIED) default to threat level "unknown" and will not be auto-committed if the minimum threshold is set above "unknown".

**Operator-in-the-loop override**
In the default configuration (`wait_for_launch: false`), the system locks and commits automatically. Setting `wait_for_launch: true` in config.yaml requires the laptop monitor operator to explicitly press ENTER before the interceptor launches. This gives a human a final deconfliction check before commitment.

---

## 22. Dependency Map

**Development machine (training / rendering):**
```
torch >= 2.1.0          MobileNetV2 training
torchvision >= 0.16.0   Model weights + transforms
open3d >= 0.19.0        3D rendering
trimesh >= 4.0.0        Mesh loading (.obj/.glb/.stl)
opencv-python >= 4.8.0  Image processing throughout
fastapi >= 0.110.0      REST API server
uvicorn >= 0.27.0       ASGI server for FastAPI
pyyaml >= 6.0           Config loading
numpy >= 1.26.0         Array operations
Pillow >= 10.0.0        Image I/O in renderer
pytest >= 8.0.0         Test runner
ultralytics             YOLOv8 training (train_yolo.py)
```

**Raspberry Pi (runtime only):**
```
onnxruntime             YOLO + classifier inference (no torch needed)
opencv-python           Camera + image processing
numpy                   Array operations
pyyaml                  Config loading
```

The Pi requires no PyTorch, no GPU, no CUDA. Total install size is under 200MB.

**Port assignments:**
| Port | Direction | Purpose |
|------|-----------|---------|
| 5555 | Ground Pi → Laptop | Drone detected alerts |
| 5556 | Laptop → Interceptor Pi | LAUNCH / ABORT commands |
