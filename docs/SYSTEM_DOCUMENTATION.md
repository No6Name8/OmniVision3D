# OmniVision3D — Complete System Documentation

---

## Executive Summary

OmniVision3D is a drone interception system with three cooperating units:

1. **Ground Detection Unit** (`pi_deploy/ground_station/` or `ground_detection/`) — a stationary Raspberry Pi that watches the sky with a camera, GPS, laser rangefinder, and compass. When it detects an enemy drone it computes the drone's GPS coordinates and can alert the laptop.
2. **Laptop Monitor** (`laptop_monitor/`) — an operator workstation that receives alerts from the ground unit and, after operator confirmation, sends a LAUNCH command to the interceptor drone.
3. **Interceptor Drone** (`pi_deploy/`) — a fixed-wing drone running entirely on a Raspberry Pi. It uses a two-stage vision pipeline (YOLOv8 ONNX + consecutive-frame counter) to detect and lock onto the target, then drives four motor channels with a PID controller.

The AI pipeline is built without a single labeled real-world photo. A 3D model of the target (DJI Mini 4 Pro) is rendered from 600 synthetic viewpoints, composited over 100 procedural sky backgrounds (1,800 images), and passed through a thermal filter (another 1,800 images). Both a YOLOv8 nano detector and a MobileNetV2 binary classifier are trained on this purely synthetic data and exported to ONNX for Pi deployment.

Reported results on the synthetic test set: 100% accuracy, 0% false-negative rate, 0% false-positive rate, 0.995 mAP50.

---

## System Architecture Overview

```
+--------------------------------------------------+
|  GROUND DETECTION UNIT (Pi)                      |
|  ground_station/main.py                          |
|  GPS -- Laser -- Compass -- YOLO Camera          |
|      targeting.py -> target GPS coords           |
|      UDP DRONE_DETECTED alert ----------------+  |
+------------------------------------------------+--+
                                                |
                                        UDP :5555
+-----------------------------------------------+--+
|  LAPTOP MONITOR                                   |
|  laptop_monitor/monitor.py                        |
|  alert_receiver -> operator ENTER                 |
|  launch_commander -------------------------+       |
+-------------------------------------------+-------+
                                            |
                                    UDP :5556 LAUNCH
+-------------------------------------------+-------+
|  INTERCEPTOR DRONE (Pi)                           |
|  pi_deploy/main.py (or drone_main.py)             |
|  CameraStream -> VisionPipeline -> Tracker        |
|  -> MotorController -> send_to_pixhawk()          |
|  -> GPSNav (stub)                                 |
+---------------------------------------------------+
```

---

## Part 1 — Ground Detection Unit (pi_deploy/ground_station/)

### Purpose

Stationary Pi positioned to watch the sky. Combines GPS (own position), laser rangefinder (target distance), compass (bearing to target), and YOLO camera (enemy identification) into a single Tkinter UI. When all sensors provide data and an enemy drone is confirmed, it computes the target's GPS coordinates.

### Hardware Components

- Raspberry Pi (any model with USB ports and Python 3.9+)
- USB camera (any OpenCV-compatible webcam)
- Serial GPS module (NMEA, 9600 baud, FTDI USB adapter) — outputs $GPRMC or $GNRMC sentences
- Laser rangefinder (binary protocol, 9600 baud, CH340 USB adapter) — frame format 0xAE 0xA7 ... 0xBC 0xBE
- Compass (text protocol, 9600 baud, FTDI USB adapter) — outputs `$C heading A ... *HEX` sentences

### Software Components (every file)

| File | Purpose |
|------|---------|
| `ground_station/main.py` | Tkinter UI; starts all four sensor/detection threads; polls shared_state every 100 ms; hosts compass calibration controls and snapshot button |
| `ground_station/shared_state.py` | Single `State` dataclass with thread lock; all sensor threads write here; UI reads here |
| `ground_station/targeting.py` | Converts own GPS + compass bearing + laser distance into target GPS coordinates using spherical Earth haversine; manages compass offset file |
| `ground_station/detection.py` | Detection thread; opens camera; runs YoloDetector every frame; maintains SCANNING/CONFIRMING/LOCKED/SEARCHING state; writes annotated frame to shared_state |
| `ground_station/sensors/gps.py` | GPS serial reader thread; parses $GPRMC/$GNRMC NMEA sentences; writes lat/lon/time_utc to shared_state |
| `ground_station/sensors/laser.py` | Laser serial reader thread; sends 0xAE 0xA7 0x04 0x00 0x05 0x09 0xBC 0xBE command every 20 ms; parses binary frame; writes distance_m to shared_state |
| `ground_station/sensors/compass.py` | Compass serial reader thread; parses `$C value A ... *HH` sentences; writes compass_raw to shared_state |
| `ground_station/__init__.py` | Empty package marker |
| `ground_station/sensors/__init__.py` | Empty package marker |

### How It Works Step By Step

1. `main.py` parses CLI arguments for `--camera`, `--model`, `--gps`, `--laser`, `--compass`, `--declination`, `--laser-scale`.
2. GPS thread opens the GPS serial port; on a valid $GPRMC/GNRMC sentence with status `A` (valid fix) it writes lat, lon, time_utc to shared_state.
3. Laser thread sends the poll command every 20 ms, reads the binary response, extracts decimetres from payload bytes 5 and 6 (`dm = (payload[5] << 8) | payload[6]`), converts to metres (`dm / 10.0 * scale`), and writes to shared_state.
4. Compass thread reads serial lines, parses the `$C value` field (or falls back to the first numeric token), writes compass_raw (degrees) to shared_state.
5. Detection thread opens the camera at 640x480, runs YoloDetector (ONNX, 320 px input, 50% threshold) on every frame, maintains consecutive counter, transitions SCANNING/CONFIRMING/LOCKED/SEARCHING, writes annotated frame and detection state to shared_state.
6. Tkinter main loop calls `_poll()` every 100 ms. Reads shared_state, calls `targeting.update_target(declination)`, updates all UI labels and the camera display.
7. `targeting.update_target()` reads lat, lon, compass_raw, distance_m; applies `compass_used = (compass_raw + declination + offset) % 360`; calls `destination_point()` to compute target lat/lon; writes back to shared_state.
8. Compass calibration: ZERO button sets offset so current reading maps to 0. Step buttons adjust offset by +/-0.1 or +/-1.0 degrees. Offset persisted to `/home/bravofox/compass_offset.txt`.
9. Snapshot button saves current annotated frame as JPEG plus text file with all sensor readings to `/home/bravofox/screenshots/`.

### Sensor Data Flow

```
GPS serial    -> thread_gps()       -> shared_state.lat / .lon / .time_utc
Laser serial  -> thread_laser()     -> shared_state.distance_m
Compass serial-> thread_compass()   -> shared_state.compass_raw
Camera        -> thread_detection() -> shared_state.frame / .phase / .confidence / .consecutive / .bbox / .det_center
                                       targeting.update_target()
                                    -> shared_state.compass_used / .target_lat / .target_lon
```

### Shared State Fields (list every field from shared_state.py)

| Field | Type | Source | Meaning |
|-------|------|--------|---------|
| `lat` | `Optional[float]` | GPS thread | Own latitude in decimal degrees |
| `lon` | `Optional[float]` | GPS thread | Own longitude in decimal degrees |
| `time_utc` | `str` | GPS thread | UTC time HHMMSS from NMEA |
| `compass_raw` | `Optional[float]` | Compass thread | Raw heading degrees from sensor |
| `compass_used` | `Optional[float]` | targeting.py | Heading after declination + user offset |
| `distance_m` | `Optional[float]` | Laser thread | Range to target in metres |
| `target_lat` | `Optional[float]` | targeting.py | Computed target latitude |
| `target_lon` | `Optional[float]` | targeting.py | Computed target longitude |
| `phase` | `str` | Detection thread | SCANNING / CONFIRMING / LOCKED / SEARCHING |
| `confidence` | `float` | Detection thread | YOLO confidence of best detection (0.0-1.0) |
| `consecutive` | `int` | Detection thread | Consecutive frames with detection above threshold |
| `bbox` | `Optional[Tuple[int,int,int,int]]` | Detection thread | Bounding box (x1, y1, x2, y2) in pixels |
| `det_center` | `Optional[Tuple[int,int]]` | Detection thread | Bounding box centre (cx, cy) in pixels |
| `frame` | `object` (np.ndarray) | Detection thread | Latest annotated BGR camera frame |

---

## Part 2 — Interceptor Drone (pi_deploy/)

### Purpose

A fixed-wing drone on a Raspberry Pi. Navigates to a GPS coordinate (stub today, MAVLink planned), activates camera, runs the two-stage vision pipeline, and when LOCKED drives four motor PWM channels via a PID controller to steer toward the target.

### Flight Phases: SCANNING -> CONFIRMING -> LOCKED -> SEARCHING

| Phase | Condition | Action |
|-------|-----------|--------|
| SCANNING | No YOLO detection | Motors at cruise (1500 us), log FPS |
| CONFIRMING | YOLO >= 50% but consecutive < 3 | Motors at cruise, accumulating confirmation frames |
| LOCKED | YOLO >= 50% for 3+ consecutive frames | Engage intercept (1900 us), run PID, send motor commands |
| SEARCHING | Target lost after LOCKED, < 3 s elapsed | Hold last heading, reset PID |
| NAVIGATING | Target lost 3-10 s | Reset motors, navigate to last known position |
| ABORT | Target lost > 10 s | Call `nav.return_to_base()`, end mission |

### Files Involved (every file)

| File | Purpose |
|------|---------|
| `main.py` | Full mission loop: navigation, VisionPipeline, Tracker, UDP listener, CSV per-frame log |
| `drone_main.py` | Lean headless flight script: no display, no identity confirmer, file kill-switch support |
| `test_ui.py` | 960x480 laptop test window: 640x480 feed (left) + 320x480 status panel (right); Q/E/R/S keys |
| `config.yaml` | All tunable parameters |
| `requirements_pi.txt` | Pi dependencies: onnxruntime, opencv-python, numpy, pyyaml, pyserial, Pillow |
| `README.md` | Pi deployment guide |
| `vision/yolo_detector.py` | YOLOv8 ONNX wrapper: parses [1,5,N] tensor, NMS, returns Detection list |
| `vision/identity_confirmer.py` | MobileNetV2 ONNX crop classifier: 20% padding, 224x224, ImageNet norm, consecutive counter |
| `vision/pipeline.py` | Four-phase state machine wrapping DroneClassifier; SEARCHING timer; overlay drawing |
| `vision/drone_classifier.py` | YOLO-only locking (50% x 3 frames); runs EnemyIdentifier every 5 frames after lock |
| `vision/enemy_identifier.py` | Loads active enemy ONNX classifiers from enemies.yaml; identifies drone species |
| `vision/camera_stream.py` | Camera in separate OS process with shared memory; non-blocking read(); no GIL contention |
| `vision/frame_enhancer.py` | CLAHE on LAB L-channel; toggleable; self-benchmarks on init |
| `vision/pi_predict.py` | Standalone MobileNetV2-only inference script; display overlay; no YOLO |
| `control/motor_controller.py` | PID: converts dx/dy pixel offsets to 4-channel PWM microsecond commands; MAVLink stub |
| `control/tracker.py` | LOCKED/SEARCHING/NAVIGATING/ABORT state machine; wraps MotorController |
| `control/motor_calibration.py` | Interactive calibration wizard: spins channels, asks operator, saves mapping to config |
| `navigation/gps_nav.py` | GPS navigation with haversine distance check; all fly_to() calls are simulation stubs |
| `enemies/enemies.yaml` | Enemy threat registry; currently `targets: []` (empty) |
| `vision/__init__.py` | Empty package marker |
| `control/__init__.py` | Empty package marker |
| `navigation/__init__.py` | Empty package marker |

### Detection Pipeline

```
Frame (BGR 640x480)
  [optional] FrameEnhancer.enhance() -- CLAHE on L channel
  DroneClassifier.classify()
    YoloDetector.detect()
      resize 320x320 -> RGB -> /255 -> NCHW -> ONNX
      parse [1,5,N] -> filter conf >= 50% -> NMS (IoU 0.45)
      scale boxes to original frame pixels
      return List[Detection]
    if detections: consecutive += 1
    if consecutive < 3: return CONFIRMING
    if consecutive >= 3:
      every 5 frames: EnemyIdentifier.identify(frame, bbox)
      return LOCKED or IDENTIFIED
  VisionPipeline: apply SEARCHING timer (2 s)
  Tracker.update(): LOCKED -> PID -> MotorController
```

### Motor Control

See Motor Control System section below.

### GPS Navigation

See Navigation System section below.

---

## Part 3 — Laptop Monitor (laptop_monitor/)

### Purpose

Operator workstation that bridges the ground detection unit and the interceptor drone. Receives DRONE_DETECTED UDP alerts from the ground unit, displays them to the operator, waits for ENTER key confirmation, then sends a LAUNCH command over UDP to the drone Pi.

### How It Works

1. `AlertReceiver` binds to UDP port 5555 and listens in a background thread. Every received JSON packet is stored as `_latest`; the main loop polls `get_latest()` (consume-once semantics).
2. On `DRONE_DETECTED` packet: prints confidence, compass heading, timestamp, source IP; sets `launch_pending = True`.
3. On `CLEAR` packet: prints "TARGET LOST -- standing down"; clears pending alert.
4. Keyboard: ENTER sends LAUNCH; `A` sends ABORT; `Q` quits.
5. `LaunchCommander` sends `{"command": "LAUNCH", "target_confidence": ..., "compass_heading": ..., "timestamp": ..., "source": "LAPTOP_MONITOR"}` via UDP to drone Pi port 5556.
6. The drone Pi `_LaunchListener` (in `pi_deploy/main.py`) receives LAUNCH -> `mission_active = True`, or ABORT -> `abort_requested = True`.

### Files Involved

| File | Purpose |
|------|---------|
| `monitor.py` | Main operator loop: polls AlertReceiver, handles keyboard, calls LaunchCommander |
| `alert_receiver.py` | UDP listener on port 5555; background thread; stores latest JSON packet |
| `launch_command.py` | UDP sender to drone Pi on port 5556; `send_launch()` and `send_abort()` |
| `config.yaml` | `listen_port: 5555`, `drone_ip: 192.168.1.200`, `drone_port: 5556`, `simulation_mode: true` |
| `requirements.txt` | pyyaml, numpy |

---

## The AI Pipeline

### YOLO Detection (input size, conf threshold, NMS threshold -- from actual code)

| Parameter | Value | Source |
|-----------|-------|--------|
| Input size | 320 px (square) | `pi_deploy/config.yaml` yolo.input_size |
| Confidence threshold | 0.50 | `pi_deploy/config.yaml` yolo.confidence_threshold |
| NMS IoU threshold | 0.45 | `pi_deploy/config.yaml` yolo.nms_threshold |
| ONNX intra_op threads | 2 | `yolo_detector.py` SessionOptions |
| ONNX inter_op threads | 1 | `yolo_detector.py` SessionOptions |
| Output tensor shape | [1, 5, N] where N=5376 at imgsz=320 | `yolo_detector.py` docstring |
| Model file | `vision/yolo_dji.onnx` | `pi_deploy/config.yaml` |
| Model size | 11.6 MB | README |
| Training epochs | 15 | `train_yolo.py` |
| Training imgsz | 512 (train), 320 (export) | `train_yolo.py`, README |
| Training batch | 16 | `train_yolo.py` |
| Training optimizer | Adam, lr=0.001 | `train_yolo.py` |
| Training augmentation | fliplr=0.5, translate=0.2, scale=0.3, degrees=15, mosaic=1.0 | `train_yolo.py` |
| Reported mAP50 | 0.995 | README |

### YoloDetector class (explain every method)

**`__init__(model_path, conf_threshold=0.50, nms_threshold=0.45, input_size=512)`**
Creates ONNX InferenceSession with `intra_op=2`, `inter_op=1`, `ORT_ENABLE_ALL` graph optimization, sequential execution mode, CPU provider. Sets `OMP_NUM_THREADS=4` in environment.

**`detect(frame: np.ndarray) -> List[Detection]`**
Resizes frame to `input_size x input_size` (INTER_LINEAR), converts BGR->RGB, divides by 255.0, transposes to NCHW (1,3,H,W). Runs ONNX session, gets raw output [1,5,N]. Transposes to [N,5] (cx, cy, w, h, confidence). Filters by `conf >= threshold`. Scales cx/cy/w/h from input_size space to original frame pixels. Computes x1/y1/x2/y2, clamps to frame bounds. Calls `_nms()` with IoU 0.45. Returns List[Detection] sorted by confidence descending.

**`draw_detections(frame, detections) -> np.ndarray`**
Draws yellow bounding boxes and `DJI? XX%` labels on a copy of the frame.

**`_nms(boxes, scores, iou_threshold=0.45) -> List[int]` (static)**
Pure NumPy greedy NMS: sort scores descending, iteratively keep highest-confidence box, suppress all remaining boxes with IoU above threshold. Returns list of kept indices.

### FrameEnhancer (what it does, when enabled)

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel of the LAB colour space only. Parameters: `clipLimit=2.0`, `tileGridSize=(8, 8)`.

On init it runs a 100-frame benchmark on a random 480x640 frame; if average > 5 ms it prints a warning.

In `test_ui.py` it starts disabled (`enable=False`) and is toggled by the `E` key. In `drone_main.py` it is intentionally omitted. In `pi_predict.py` it is not used.

### Training Process (what dataset, what script, what results)

**Dataset**: 7,200 synthetic images total.
- 600 raw renders (100 Fibonacci sphere angles x 6 distances at 2.0, 3.5, 5.5, 8.0, 12.0, 18.0 mesh units).
- Each render composited over 3 sky backgrounds -> 1,800 RGB images.
- Each of the 1,800 RGB images through thermal filter -> 1,800 thermal images.
- 1,800 sky-only no_drone backgrounds (18 brightness-jittered variants per background x 100 backgrounds).

**MobileNetV2 training** (`training/train.py`):
- Backbone: MobileNetV2 pretrained on ImageNet; head replaced with Dropout(0.2) + Linear(1280, num_classes).
- Input: 224x224, ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
- Optimizer: Adam lr=0.001; scheduler ReduceLROnPlateau(patience=3, factor=0.5).
- Max epochs: 50; early stopping patience: 5; batch size: 32.
- 80/10/10 train/val/test split (stratified per folder, seed=42).
- Train augmentation: RandomHorizontalFlip(0.5), RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2).

**Actual test results** (`checkpoints/confusion_matrix.csv`):
```
                 no_drone   dji_mini_4_pro
no_drone            360            0
dji_mini_4_pro        0          360
```
100% accuracy, 0 false negatives, 0 false positives on 720-image test set.

**YOLO training** (`train_yolo.py`):
YOLOv8n pretrained, fine-tuned on synthetic renders converted to YOLO bbox format by `training/convert_to_yolo.py`. 1-class model. Exported to ONNX at imgsz=320 with simplification.

**Real-data experiment** (`runs/detect/checkpoints/yolo/real_drone_detector/args.yaml`):
A second YOLO run on real drone images at imgsz=640, 50 epochs, patience=10. Separate from the deployed model.

### Enemy Identification (how enemies.yaml is used)

`pi_deploy/enemies/enemies.yaml` is the threat registry. Current contents: `targets: []` (no active enemies).

Expected entry format:
```yaml
targets:
  - name: "DJI Phantom 4"
    model: "enemies/dji_phantom4.onnx"
    threat_level: "high"
    description: ""
    confidence_threshold: 0.85
    active: true
```

At startup `EnemyIdentifier` loads ONNX sessions for all `active: true` entries with per-enemy thresholds and threat levels. After LOCKED state, `DroneClassifier` calls `EnemyIdentifier.identify(frame, bbox)` every 5 frames (non-blocking). The crop is preprocessed to 224x224 ImageNet-normalized, and each active model scores it. The highest-confidence match above its threshold is returned as `threat_name` and displayed on screen. Enemy identification does NOT gate the intercept -- the drone commits to LOCKED based on YOLO confidence alone.

`add_enemy(name, model_path, threat_level)` copies the model file to `enemies/`, appends to the YAML, and hot-loads the ONNX session without restart.

---

## Motor Control System

### 4-Motor Design

Top-down view (nose forward):
```
M1 (front-left)    M2 (front-right)
M3 (rear-left)     M4 (rear-right)
```

PWM range: 1000-2000 microseconds. Neutral/cruise: 1500 us. Intercept: 1900 us.

Motor formulas (yaw_us and pitch_us are PID-computed corrections):
```
M1 (front-left)  = base + yaw_us + pitch_us
M2 (front-right) = base - yaw_us + pitch_us
M3 (rear-left)   = base + yaw_us - pitch_us
M4 (rear-right)  = base - yaw_us - pitch_us
```

Right turn (positive yaw): M1 and M3 increase, M2 and M4 decrease.
Pitch up (positive pitch): M1 and M2 increase, M3 and M4 decrease.

Physical channels are mapped in `motor_mapping` config (default: FL=CH1, FR=CH2, RL=CH3, RR=CH4). `flip_yaw` and `flip_pitch` boolean flags swap pairs if wiring is reversed.

### PID Logic (explain the actual PID variables and math)

**MotorController** (`control/motor_controller.py`) -- config values from `pi_deploy/config.yaml` motor_control.pid:
- `Kp = 0.06`, `Ki = 0.001`, `Kd = 0.01`
- `max_correction_pct = 20` (corrections clamped to +/-20% of base)

Error convention:
- Yaw error = `dx` (pixels right of centre = positive = need right turn)
- Pitch error = `-dy` (inverted Y: above centre = negative dy = positive pitch error = pitch up)

Discrete PID (no dt normalization in MotorController):
```
yaw_correction   = Kp*dx + integral(Ki*dx) + Kd*(dx - last_error)
pitch_correction = Kp*(-dy) + integral(Ki*(-dy)) + Kd*((-dy) - last_pitch_error)
```

After clamping to +/-max_correction_pct:
```
yaw_us   = (yaw_correction   / 100.0) * base_us   # base_us = 1500
pitch_us = (pitch_correction / 100.0) * base_us
```

**Tracker** (`control/tracker.py`) uses a time-aware `_PID` class -- config values from `pi_deploy/config.yaml` tracking:
- `Kp = 0.1`, `Ki = 0.01`, `Kd = 0.05`
- Inputs normalized: `norm_dx = dx / (frame_width / 2.0)`
- Update: `out = Kp*error + Ki*integral + Kd*(derivative / dt)` with `dt` from `time.monotonic()`
- Output clamped to [-1.0, 1.0]

Note: `main.py` uses the Tracker PID; `drone_main.py` uses MotorController PID directly.

### Calibration

`control/motor_calibration.py` runs interactively before first flight:
1. Spins each ESC channel at 1600 us for 2 s; operator identifies which motor moved.
2. Records motor-to-channel mapping.
3. Commands 2% right-turn correction; if drone turns wrong way, `flip_yaw = True` and swaps left/right.
4. Commands 2% pitch-up correction; if nose goes down, `flip_pitch = True` and swaps front/rear.
5. Saves updated `motor_mapping` to `config.yaml`.

Default mode is simulation (prints instead of sending MAVLink).

### Intercept Mode

On first LOCKED frame: `engage_intercept()` sets `intercept_mode = True`, prints "INTERCEPT ENGAGED -- FULL THRUST". The `active_base` for motor commands switches from 1500 to 1900 us. PID correction magnitudes remain relative to the 1500 us cruise base so differential effort is throttle-independent.

On leaving LOCKED: `disengage_intercept()` returns to cruise.

---

## Navigation System

### GPS implementation

`navigation/gps_nav.py` -- simulation stubs only. No real serial GPS read is implemented.

- `get_current_position()` returns internal `(_current_lat, _current_lon, _current_alt)`. TODO: replace with serial GPS or dronekit.
- `fly_to(lat, lon, altitude, simulation)` -- in simulation: logs, prints, sleeps 2 s, updates internal position.
- `is_close_enough(target, threshold=50.0)` -- calls `_haversine()` and compares to threshold metres.
- `return_to_base(simulation)` -- calls `fly_to(home_lat, home_lon, ...)`.
- `hold_position()` -- logs and prints only. TODO: MAVLink MAV_CMD_NAV_LOITER_UNLIM.

### Dead Reckoning (exists or not)

NOT IMPLEMENTED. No dead reckoning logic exists anywhere in the codebase.

### Targeting (haversine math from targeting.py -- explain the formula)

`destination_point(lat1_deg, lon1_deg, bearing_deg, distance_m)` computes target GPS from own position + bearing + range:

```
phi1  = radians(lat1_deg)
lam1  = radians(lon1_deg)
theta = radians(bearing_deg)
delta = distance_m / 6_371_000.0       # angular distance on spherical Earth

phi2 = asin( sin(phi1)*cos(delta) + cos(phi1)*sin(delta)*cos(theta) )

lam2 = lam1 + atan2( sin(theta)*sin(delta)*cos(phi1),
                     cos(delta) - sin(phi1)*sin(phi2) )

target_lat = degrees(phi2)
target_lon = ((degrees(lam2) + 540.0) % 360.0) - 180.0   # wrap to [-180,180]
```

`gps_nav.py` uses a simpler haversine for distance-only checks:
```
a = sin(dlat/2)^2 + cos(lat1)*cos(lat2)*sin(dlon/2)^2
dist = 2 * 6_371_000 * asin(sqrt(a))
```

### Compass calibration (offset file, ZERO button logic)

Offset stored at `/home/bravofox/compass_offset.txt` (plain float text file).

`read_offset()` reads and returns float; returns 0.0 on any error.
`write_offset(deg)` wraps to [-180,180] via `((deg+180) % 360) - 180` before writing.

**ZERO button** (`_cal_zero()` in `ground_station/main.py`):
- If `compass_used` available: `new_offset = current_offset - compass_used` (forces next compass_used to 0)
- Else if `compass_raw` available: `new_offset = -(compass_raw + declination)`
- Else: prints "No compass data yet"

**Step buttons**: `_cal_step(delta)` adds +/-0.1 or +/-1.0 to current offset and saves.

---

## Ground Station Sensor System

### GPS Reader (protocol, baud rate, NMEA sentences parsed)

- Protocol: NMEA serial 8N1
- Baud rate: 9600 (configurable via `baud` parameter)
- Sentences parsed: `$GPRMC` and `$GNRMC` only
- Validity: field index 2 must be `"A"` (active fix); `"V"` (void) ignored
- Coordinate parsing: DDMM.MMMM -> decimal degrees: `deg = int(value[:dot-2])`, `mins = float(value[dot-2:])`, `decimal = deg + mins/60.0`, negated for S/W
- Writes: `lat`, `lon`, `time_utc` (HHMMSS from field index 1) to shared_state
- Optional legacy file write to `/home/bravofox/gpsdata.txt`
- Error handling: prints error, retries after 3 s

### Laser Rangefinder (binary protocol, frame format, distance formula)

- Baud rate: 9600, timeout 0.02 s
- Poll command (8 bytes): `0xAE 0xA7 0x04 0x00 0x05 0x09 0xBC 0xBE` sent every 20 ms
- Frame format: `0xAE 0xA7 <payload> 0xBC 0xBE`
- Distance formula: `dm = (payload[5] << 8) | payload[6]` (decimetres); `distance_m = (dm / 10.0) * scale`
- Validity: `dm <= 0` or `dm > 50000` rejected
- Scale factor from `--laser-scale` CLI argument (default 1.0)
- Error handling: prints error, retries after 3 s

### Compass (text protocol, field parsing, fallback)

- Baud rate: 9600, timeout 0.05 s
- Primary: searches for `$<body>*<checksum>` pattern, extracts field `C value` using `([A-Z])\s*(-?\d+(?:\.\d+)?)` regex
- Fallback: if no standard sentence found, extracts first numeric value from raw line
- Writes: `compass_raw` (degrees, no declination) to shared_state
- Error handling: prints error, retries after 3 s

### Targeting calculation (full formula: raw + declination + offset -> bearing -> haversine -> target GPS)

```
1. own_lat, own_lon = shared_state.lat, shared_state.lon
2. raw_bearing      = shared_state.compass_raw
3. offset           = read_offset()    # from /home/bravofox/compass_offset.txt
4. compass_used     = (raw_bearing + declination_deg + offset) % 360.0
5. distance_m       = shared_state.distance_m
6. delta            = distance_m / 6_371_000.0
7. phi2             = asin(sin(lat)*cos(delta) + cos(lat)*sin(delta)*cos(compass_used))
8. lam2             = lon + atan2(sin(compass_used)*sin(delta)*cos(lat),
                                   cos(delta) - sin(lat)*sin(phi2))
9. target_lat       = degrees(phi2)
10. target_lon      = ((degrees(lam2) + 540) % 360) - 180
11. shared_state.compass_used = compass_used
12. shared_state.target_lat   = target_lat
13. shared_state.target_lon   = target_lon
```

---

## Communication Flow

### Ground Unit -> Laptop (UDP if exists)

**Sender**: `ground_detection/comms/alert_sender.py` (`AlertSender`)
**Protocol**: UDP fire-and-forget, default port 5555
**Packets**:
- `DRONE_DETECTED`: `{"type": "DRONE_DETECTED", "confidence": float, "consecutive": int, "compass_heading": float|null, "timestamp": float, "source": "GROUND_UNIT_001"}`
- `CLEAR`: `{"type": "CLEAR", "timestamp": float, "source": "GROUND_UNIT_001"}`

Alert sent on first lock (consecutive >= 3) and every 30 consecutive frames (~1 s). CLEAR sent when detection is lost.

Note: The integrated ground station (`pi_deploy/ground_station/`) does not currently include a UDP sender -- it writes to shared_state for Tkinter display only. The `ground_detection/` module has the full sender.

### Laptop -> Interceptor (if exists)

**Sender**: `laptop_monitor/launch_command.py` (`LaunchCommander`)
**Protocol**: UDP fire-and-forget, default port 5556
**Packets**:
- `LAUNCH`: `{"command": "LAUNCH", "target_confidence": float, "compass_heading": float|null, "timestamp": float, "source": "LAPTOP_MONITOR"}`
- `ABORT`: `{"command": "ABORT", "timestamp": float, "source": "LAPTOP_MONITOR"}`

**Receiver**: `pi_deploy/main.py` `_LaunchListener` binds to `0.0.0.0:5556`. LAUNCH -> `mission_active = True`; ABORT -> `abort_requested = True`.

---

## Configuration Files

### pi_deploy/config.yaml (every field)

| Key | Default | Meaning |
|-----|---------|---------|
| `yolo.model_path` | `vision/yolo_dji.onnx` | YOLO ONNX model (relative to pi_deploy/) |
| `yolo.confidence_threshold` | 0.50 | Minimum detection confidence |
| `yolo.nms_threshold` | 0.45 | IoU threshold for NMS |
| `yolo.input_size` | 320 | Inference image size in pixels |
| `identity_confirmation.model_path` | `vision/omnivision3d.onnx` | MobileNetV2 ONNX model |
| `identity_confirmation.confidence_threshold` | 0.85 | Min confidence to increment consecutive counter |
| `identity_confirmation.consecutive_required` | 3 | Frames needed before identity confirmed |
| `identity_confirmation.reset_threshold` | 0.50 | Confidence below this resets counter to 0 |
| `identity_confirmation.crop_padding` | 0.20 | Fractional padding added to YOLO bbox |
| `tracking.lost_searching_seconds` | 3.0 | Hold last heading for this many seconds |
| `tracking.lost_navigating_seconds` | 10.0 | Navigate to last position before ABORT |
| `tracking.pid_p` | 0.1 | Tracker PID P gain |
| `tracking.pid_i` | 0.01 | Tracker PID I gain |
| `tracking.pid_d` | 0.05 | Tracker PID D gain |
| `navigation.home_lat` | 24.7136 | Home base latitude |
| `navigation.home_lon` | 46.6753 | Home base longitude |
| `navigation.altitude` | 50 | Target flight altitude (metres) |
| `navigation.close_enough_meters` | 50 | Acceptance radius for GPS waypoint |
| `camera.index` | 0 | Camera device index |
| `camera.width` | 640 | Capture width in pixels |
| `camera.height` | 480 | Capture height in pixels |
| `drone_classifier.generic_threshold` | 0.50 | YOLO confidence to count as detection hit |
| `drone_classifier.consecutive_required` | 3 | Consecutive hits to enter LOCKED |
| `drone_classifier.lost_timeout_seconds` | 2.0 | SEARCHING phase duration in seconds |
| `drone_classifier.enemy_id_every_n_frames` | 5 | Enemy identification frequency |
| `drone_classifier.enemies_file` | `enemies/enemies.yaml` | Path to threat registry |
| `motor_control.cruise_power_us` | 1500 | Cruise PWM (microseconds) |
| `motor_control.base_power_us` | 1500 | Base for PID correction math |
| `motor_control.intercept_power_us` | 1900 | Throttle when LOCKED |
| `motor_control.min_power_us` | 1000 | PWM floor |
| `motor_control.max_power_us` | 2000 | PWM ceiling |
| `motor_control.max_correction_pct` | 20 | Maximum PID correction as % of base |
| `motor_control.max_pixel_offset` | 320 | Reference pixel offset (frame half-width) |
| `motor_control.simulation_mode` | true | Print commands instead of MAVLink |
| `motor_control.pid.Kp` | 0.06 | Motor controller PID P gain |
| `motor_control.pid.Ki` | 0.001 | Motor controller PID I gain |
| `motor_control.pid.Kd` | 0.01 | Motor controller PID D gain |
| `motor_mapping.front_left` | channel_1 | ESC channel for front-left motor |
| `motor_mapping.front_right` | channel_2 | ESC channel for front-right motor |
| `motor_mapping.rear_left` | channel_3 | ESC channel for rear-left motor |
| `motor_mapping.rear_right` | channel_4 | ESC channel for rear-right motor |
| `motor_mapping.flip_yaw` | false | Swap left/right if yaw direction wrong |
| `motor_mapping.flip_pitch` | false | Swap front/rear if pitch direction wrong |
| `motor_mapping.calibrated` | false | Set true after running motor_calibration.py |
| `launch_listener.enabled` | true | Start UDP launch listener on boot |
| `launch_listener.port` | 5556 | UDP port for LAUNCH/ABORT commands |
| `launch_listener.wait_for_launch` | false | Hold in STANDBY until LAUNCH received |
| `simulation_mode` | true | Global simulation flag |

### configs/default.yaml (every field)

| Key | Value | Meaning |
|-----|-------|---------|
| `rendering.sphere_angles` | 100 | Fibonacci sphere viewpoints |
| `rendering.distances` | [2.0, 3.5, 5.5, 8.0, 12.0, 18.0] | Camera distances in mesh units |
| `rendering.lighting_variations` | 5 [x,y,z] positions | Point light positions cycled across batches |
| `rendering.image_size` | 512 | Render resolution (square pixels) |
| `rendering.output_path` | `dataset` | Root for raw/ and processed/ sub-folders |
| `rendering.supported_formats` | .obj .glb .stl .fbx .step .stp | Mesh formats accepted |
| `training.epochs` | 30 | Max training epochs |
| `training.batch_size` | 32 | Training batch size |
| `training.learning_rate` | 0.001 | Initial learning rate |
| `training.checkpoint_dir` | `checkpoints` | Where to save best.pt |
| `thermal` | false | Whether thermal mode is active by default |
| `thermal_palettes` | [iron, rainbow, grayscale] | Available thermal colormaps |
| `classes` | [no_drone, dji_mini_4_pro] | Class names |

### laptop_monitor/config.yaml (every field)

| Key | Value | Meaning |
|-----|-------|---------|
| `comms.listen_port` | 5555 | UDP port for ground unit alerts |
| `comms.drone_ip` | 192.168.1.200 | Interceptor drone Pi IP address |
| `comms.drone_port` | 5556 | UDP port on drone Pi for commands |
| `display.show_all_alerts` | true | Display every alert |
| `display.log_to_file` | true | Log alerts to file |
| `simulation_mode` | true | LAUNCH commands go to localhost |

### ground_detection/config.yaml (every field)

| Key | Value | Meaning |
|-----|-------|---------|
| `camera.index` | 0 | Camera device index |
| `camera.width` | 640 | Frame width |
| `camera.height` | 480 | Frame height |
| `detection.model_path` | `../pi_deploy/vision/yolo_dji.onnx` | YOLO model path |
| `detection.conf_threshold` | 0.50 | Detection confidence threshold |
| `detection.consecutive_required` | 3 | Frames for lock |
| `detection.input_size` | 320 | YOLO input size |
| `comms.laptop_ip` | 192.168.1.100 | Laptop alert target |
| `comms.alert_port` | 5555 | UDP alert port |
| `comms.source_id` | GROUND_UNIT_001 | Packet source identifier |
| `simulation_mode` | true | Alerts go to localhost |

---

## What Is Built and Working

- Full synthetic dataset generation pipeline (render -> composite -> thermal -> YOLO convert)
- MobileNetV2 training with early stopping, confusion matrix, training curve
- YOLOv8n training script and ONNX export
- ONNX inference pipeline (YoloDetector + IdentityConfirmer + DroneClassifier + EnemyIdentifier)
- Four-phase vision state machine (SCANNING -> CONFIRMING -> LOCKED -> SEARCHING)
- CameraStream with OS-level process isolation and shared memory
- FrameEnhancer (CLAHE)
- Tracker with time-based state transitions (LOCKED -> SEARCHING -> NAVIGATING -> ABORT)
- MotorController with dual-axis PID, 4-channel PWM math, intercept mode
- Motor calibration wizard
- Ground station Tkinter UI with all four sensor threads
- GPS NMEA parser, laser binary frame parser, compass text parser
- Haversine targeting formula
- Compass offset file with ZERO calibration button
- Laptop monitor (AlertReceiver + LaunchCommander)
- UDP LAUNCH/ABORT command flow (ground -> laptop -> drone)
- `_LaunchListener` in drone Pi for LAUNCH/ABORT commands
- Kill-switch file (`pi_deploy/KILL`) for headless drone stop
- Test UI (960x480 laptop window) with E/R/S/Q keyboard controls
- FastAPI inference server (`api/server.py`)
- 19 unit and integration tests (renderer + inference)

### What Needs Hardware to Test

- GPS serial read (requires NMEA GPS module)
- Laser rangefinder serial read (requires the specific binary-protocol device)
- Compass serial read (requires compass with `$C` sentence format)
- Motor PWM output (requires Pixhawk or ESC wiring + pymavlink)
- Full intercept flight (requires assembled fixed-wing airframe)

### Known Limitations

- `gps_nav.py` `get_current_position()` returns the last commanded simulated position, not a real GPS fix.
- `motor_controller.py` `send_to_pixhawk()` only logs to CSV and prints -- the MAVLink RC_CHANNELS_OVERRIDE call is commented out.
- `motor_calibration.py` in live mode calls `_spin_channel()` which has TODO but sends nothing.
- `gps_nav.py` `hold_position()` logs only -- TODO MAVLink MAV_CMD_NAV_LOITER_UNLIM.
- `pi_predict.py` `predict_frame()` always returns `offset = (0.0, 0.0)` -- TODO replace with det_head bbox centre.
- The integrated ground station (`pi_deploy/ground_station/`) has no UDP alert sender. The standalone `ground_detection/` module has that functionality.
- `enemies/enemies.yaml` is empty -- enemy identification never matches a specific threat in the current deployment.
- `simulation_mode: true` is the default in all three configs -- no hardware touched without explicitly setting false.

### TODO comments found in code (list them)

| File | Line | TODO text |
|------|------|-----------|
| `pi_deploy/vision/pi_predict.py` | 116 | `# TODO: replace with det_head bbox centre` |
| `pi_deploy/control/motor_controller.py` | 251 | `# TODO: MAVLink wiring` (with commented MAVLink code below) |
| `pi_deploy/control/motor_calibration.py` | 34 | `# TODO: MAVLink RC_CHANNELS_OVERRIDE pulse on ch` |
| `pi_deploy/control/motor_calibration.py` | 69 | `# TODO: send via MAVLink` |
| `pi_deploy/navigation/gps_nav.py` | 10 | `TODO (MAVLink wiring): Replace fly_to() stub with MISSION_ITEM...` |
| `pi_deploy/navigation/gps_nav.py` | 53 | `TODO: replace with real GPS read.` |
| `pi_deploy/navigation/gps_nav.py` | 73 | `# TODO: MAVLink MISSION_ITEM command` |
| `pi_deploy/navigation/gps_nav.py` | 97 | `# TODO: MAVLink MAV_CMD_NAV_LOITER_UNLIM` |

---

## Deployment

### Install dependencies (from requirements files)

**Development machine (training + rendering):**
```bash
pip install torch>=2.1.0 torchvision>=0.16.0 open3d>=0.19.0 trimesh>=4.0.0 \
    opencv-python>=4.8.0 fastapi>=0.110.0 "uvicorn[standard]>=0.27.0" \
    pyyaml>=6.0 numpy>=1.26.0 Pillow>=10.0.0 pytest>=8.0.0
```

**Raspberry Pi (drone):**
```bash
pip install onnxruntime opencv-python numpy pyyaml pyserial Pillow
```

**Laptop monitor:**
```bash
pip install pyyaml numpy
```

**Standalone ground detection unit:**
```bash
pip install onnxruntime opencv-python numpy pyyaml
```

### Run on Raspberry Pi (exact command)

```bash
# Copy to Pi
scp -r pi_deploy/ pi@<PI_IP>:~/omnivision3d/
ssh pi@<PI_IP>
cd ~/omnivision3d
pip install -r requirements_pi.txt

# Full mission loop
python main.py --lat 24.7136 --lon 46.6753

# Simulation mode (no GPS required)
python main.py --sim

# Wait for LAUNCH command before scanning
python main.py --wait-launch

# Against a recorded video
python main.py --video tests/test_video.mp4 --sim

# Lean headless script (fastest, minimal dependencies)
python drone_main.py --sim

# Ground station (all sensors)
python ground_station/main.py \
    --camera 0 \
    --model vision/yolo_dji.onnx \
    --gps /dev/serial/by-id/usb-FTDI_...-port0 \
    --laser /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
    --compass /dev/serial/by-id/usb-FTDI_...-AQ045IV6-if00-port0 \
    --declination 4.0

# Kill the drone remotely (over SSH)
touch ~/omnivision3d/KILL
```

### Run on Laptop (exact command)

```bash
# Test UI (live camera)
cd pi_deploy
python test_ui.py --camera 0

# Test UI (video file)
python test_ui.py --video path/to/video.mp4

# Laptop monitor
cd laptop_monitor
python monitor.py --drone 192.168.1.200
python monitor.py --sim    # simulation mode

# FastAPI inference server
uvicorn api.server:app --reload

# Inference on image folder
python inference/predict.py --model checkpoints/best.pt \
    --input tests/test_images/ --output tests/test_results/

# Standalone ground detection unit
cd ground_detection
python detect.py --laptop 192.168.1.100
python detect.py --sim
```

### Run Tests

```bash
python -m pytest tests/ -v
```

19 tests covering renderer (camera_utils, sky_compositor, thermal_filter) and inference (predict, visualize). All passing.

---

## Performance Numbers

### From actual code: YOLO input size, conf threshold, thread count, FPS target

| Metric | Value | Source |
|--------|-------|--------|
| YOLO input size | 320 px | `pi_deploy/config.yaml` |
| YOLO confidence threshold | 0.50 | `pi_deploy/config.yaml` |
| YOLO NMS IoU threshold | 0.45 | `pi_deploy/config.yaml` |
| OMP_NUM_THREADS | 4 | `yolo_detector.py`, `identity_confirmer.py`, `pi_predict.py` |
| ONNX intra_op threads (YOLO) | 2 | `yolo_detector.py` |
| ONNX intra_op threads (MobileNetV2) | max(4, cpu_count) | `identity_confirmer.py` |
| YOLO inference time | ~18 ms | README |
| MobileNetV2 inference time | ~7 ms | README |
| Scanning FPS (laptop) | 37 FPS | README |
| Locked FPS (laptop) | 61 FPS | README |
| Pi 4 expected FPS | 15-20 FPS | `pi_deploy/README.md` |
| Consecutive frames to lock | 3 | `pi_deploy/config.yaml` |
| SEARCHING timeout | 2.0 s | `pi_deploy/config.yaml` |
| CLAHE benchmark target | < 5 ms per frame | `frame_enhancer.py` |

### From any test results found

| Metric | Value | Source |
|--------|-------|--------|
| Synthetic training images | 7,200 | README |
| Test accuracy | 100% | `checkpoints/confusion_matrix.csv` |
| False negative rate | 0% | `checkpoints/confusion_matrix.csv` |
| False positive rate | 0% | `checkpoints/confusion_matrix.csv` |
| Test set size | 720 images (360 drone + 360 no_drone) | `checkpoints/confusion_matrix.csv` |
| YOLO mAP50 at imgsz=320 | 0.995 | README |
| MobileNetV2 model size | 8.5 MB | README |
| YOLOv8n model size | 11.6 MB | README |

---

## Complete File Reference

Alphabetical list of every file with its full path and one-sentence purpose.

| File | Full Path | Purpose |
|------|-----------|---------|
| `.gitignore` | `/OmniVision3D/.gitignore` | Git ignore rules for datasets, model weights, venvs, kill-switch file |
| `api/__init__.py` | `/OmniVision3D/api/__init__.py` | Empty package marker |
| `api/server.py` | `/OmniVision3D/api/server.py` | FastAPI server with `/health` and `/predict` (image upload) endpoints |
| `checkpoints/confusion_matrix.csv` | `/OmniVision3D/checkpoints/confusion_matrix.csv` | MobileNetV2 test-set result: 360/360 drone correct, 360/360 no_drone correct |
| `configs/default.yaml` | `/OmniVision3D/configs/default.yaml` | Rendering and MobileNetV2 training configuration |
| `docs/QUICK_START.md` | `/OmniVision3D/docs/QUICK_START.md` | Quick command reference card |
| `docs/SYSTEM_DOCUMENTATION.md` | `/OmniVision3D/docs/SYSTEM_DOCUMENTATION.md` | This file -- complete system documentation |
| `ground_detection/comms/__init__.py` | `/OmniVision3D/ground_detection/comms/__init__.py` | Empty package marker |
| `ground_detection/comms/alert_sender.py` | `/OmniVision3D/ground_detection/comms/alert_sender.py` | UDP DRONE_DETECTED/CLEAR sender from ground unit to laptop |
| `ground_detection/config.yaml` | `/OmniVision3D/ground_detection/config.yaml` | Standalone ground detection unit configuration |
| `ground_detection/detect.py` | `/OmniVision3D/ground_detection/detect.py` | Standalone ground detection main loop: camera, YOLO, alert sender, live display |
| `ground_detection/logs/__init__.py` | `/OmniVision3D/ground_detection/logs/__init__.py` | Empty package marker |
| `ground_detection/requirements.txt` | `/OmniVision3D/ground_detection/requirements.txt` | onnxruntime, opencv-python, numpy, pyyaml |
| `ground_detection/vision/__init__.py` | `/OmniVision3D/ground_detection/vision/__init__.py` | Empty package marker |
| `ground_detection/vision/drone_detector.py` | `/OmniVision3D/ground_detection/vision/drone_detector.py` | YOLO ONNX wrapper for standalone ground detection unit with consecutive counter |
| `inference/__init__.py` | `/OmniVision3D/inference/__init__.py` | Empty package marker |
| `inference/predict.py` | `/OmniVision3D/inference/predict.py` | PyTorch Predictor class plus CLI for image folder and video inference |
| `inference/visualize.py` | `/OmniVision3D/inference/visualize.py` | draw_prediction() and annotate_video() overlay utilities |
| `laptop_monitor/alert_receiver.py` | `/OmniVision3D/laptop_monitor/alert_receiver.py` | UDP listener on port 5555; background thread; consume-once packet storage |
| `laptop_monitor/config.yaml` | `/OmniVision3D/laptop_monitor/config.yaml` | Laptop monitor network configuration |
| `laptop_monitor/launch_command.py` | `/OmniVision3D/laptop_monitor/launch_command.py` | UDP LAUNCH/ABORT sender to drone Pi on port 5556 |
| `laptop_monitor/monitor.py` | `/OmniVision3D/laptop_monitor/monitor.py` | Operator terminal UI: receives alerts, waits for ENTER, sends LAUNCH |
| `laptop_monitor/requirements.txt` | `/OmniVision3D/laptop_monitor/requirements.txt` | pyyaml, numpy |
| `pi_deploy/config.yaml` | `/OmniVision3D/pi_deploy/config.yaml` | Complete drone mission configuration -- all thresholds, PID gains, motor mapping |
| `pi_deploy/control/__init__.py` | `/OmniVision3D/pi_deploy/control/__init__.py` | Empty package marker |
| `pi_deploy/control/motor_calibration.py` | `/OmniVision3D/pi_deploy/control/motor_calibration.py` | Interactive motor calibration wizard; identifies channel-to-motor mapping; saves to config |
| `pi_deploy/control/motor_controller.py` | `/OmniVision3D/pi_deploy/control/motor_controller.py` | PID controller: converts pixel dx/dy to 4-channel PWM commands; MAVLink stub |
| `pi_deploy/control/tracker.py` | `/OmniVision3D/pi_deploy/control/tracker.py` | LOCKED/SEARCHING/NAVIGATING/ABORT state machine wrapping MotorController |
| `pi_deploy/drone_main.py` | `/OmniVision3D/pi_deploy/drone_main.py` | Lean headless flight script: no display, no confirmer, file kill-switch |
| `pi_deploy/enemies/enemies.yaml` | `/OmniVision3D/pi_deploy/enemies/enemies.yaml` | Enemy threat registry (currently empty: `targets: []`) |
| `pi_deploy/ground_station/__init__.py` | `/OmniVision3D/pi_deploy/ground_station/__init__.py` | Empty package marker |
| `pi_deploy/ground_station/detection.py` | `/OmniVision3D/pi_deploy/ground_station/detection.py` | YOLO detection thread for ground station; writes phase/frame/detection to shared_state |
| `pi_deploy/ground_station/main.py` | `/OmniVision3D/pi_deploy/ground_station/main.py` | Unified ground station Tkinter UI; starts GPS/laser/compass/detection threads |
| `pi_deploy/ground_station/sensors/__init__.py` | `/OmniVision3D/pi_deploy/ground_station/sensors/__init__.py` | Empty package marker |
| `pi_deploy/ground_station/sensors/compass.py` | `/OmniVision3D/pi_deploy/ground_station/sensors/compass.py` | Compass serial reader; parses `$C value` NMEA-like sentences at 9600 baud |
| `pi_deploy/ground_station/sensors/gps.py` | `/OmniVision3D/pi_deploy/ground_station/sensors/gps.py` | GPS serial reader; parses $GPRMC/$GNRMC at 9600 baud |
| `pi_deploy/ground_station/sensors/laser.py` | `/OmniVision3D/pi_deploy/ground_station/sensors/laser.py` | Laser serial reader; binary 0xAE 0xA7 frame protocol; 20 ms poll period |
| `pi_deploy/ground_station/shared_state.py` | `/OmniVision3D/pi_deploy/ground_station/shared_state.py` | Thread-safe State dataclass with Lock; single source of truth for all ground sensors |
| `pi_deploy/ground_station/targeting.py` | `/OmniVision3D/pi_deploy/ground_station/targeting.py` | Haversine destination-point formula; compass offset file read/write/calibration |
| `pi_deploy/main.py` | `/OmniVision3D/pi_deploy/main.py` | Full mission loop: navigation, VisionPipeline, Tracker, UDP listener, CSV frame logging |
| `pi_deploy/navigation/__init__.py` | `/OmniVision3D/pi_deploy/navigation/__init__.py` | Empty package marker |
| `pi_deploy/navigation/gps_nav.py` | `/OmniVision3D/pi_deploy/navigation/gps_nav.py` | GPS navigation stubs: fly_to/return_to_base/hold_position all TODO MAVLink |
| `pi_deploy/README.md` | `/OmniVision3D/pi_deploy/README.md` | Pi deployment guide: install, run, config reference, future upgrades |
| `pi_deploy/requirements_pi.txt` | `/OmniVision3D/pi_deploy/requirements_pi.txt` | Pi dependencies: onnxruntime, opencv-python, numpy, pyyaml, pyserial, Pillow |
| `pi_deploy/test_ui.py` | `/OmniVision3D/pi_deploy/test_ui.py` | 960x480 laptop test window: camera feed + status panel; Q/E/R/S keyboard controls |
| `pi_deploy/vision/__init__.py` | `/OmniVision3D/pi_deploy/vision/__init__.py` | Empty package marker |
| `pi_deploy/vision/camera_stream.py` | `/OmniVision3D/pi_deploy/vision/camera_stream.py` | Camera in separate OS process with shared memory; non-blocking read(); no GIL |
| `pi_deploy/vision/drone_classifier.py` | `/OmniVision3D/pi_deploy/vision/drone_classifier.py` | YOLO-only locking plus background EnemyIdentifier; returns ClassificationResult |
| `pi_deploy/vision/enemy_identifier.py` | `/OmniVision3D/pi_deploy/vision/enemy_identifier.py` | Loads and runs all active enemy ONNX classifiers; hot-loadable via add_enemy() |
| `pi_deploy/vision/frame_enhancer.py` | `/OmniVision3D/pi_deploy/vision/frame_enhancer.py` | Toggleable CLAHE on LAB L-channel with self-benchmark and 5 ms warning |
| `pi_deploy/vision/identity_confirmer.py` | `/OmniVision3D/pi_deploy/vision/identity_confirmer.py` | MobileNetV2 ONNX crop classifier with hysteresis consecutive-frame counter |
| `pi_deploy/vision/pi_predict.py` | `/OmniVision3D/pi_deploy/vision/pi_predict.py` | Standalone MobileNetV2-only Pi inference script with display overlay |
| `pi_deploy/vision/pipeline.py` | `/OmniVision3D/pi_deploy/vision/pipeline.py` | Four-phase state machine (SCANNING/CONFIRMING/LOCKED/SEARCHING) with timer and overlay |
| `pi_deploy/vision/yolo_detector.py` | `/OmniVision3D/pi_deploy/vision/yolo_detector.py` | YOLOv8 ONNX wrapper: parses [1,5,N] output, NMS, returns sorted Detection list |
| `README.md` | `/OmniVision3D/README.md` | Project overview: results table, full pipeline walkthrough, test UI guide |
| `renderer/__init__.py` | `/OmniVision3D/renderer/__init__.py` | Empty package marker |
| `renderer/camera_utils.py` | `/OmniVision3D/renderer/camera_utils.py` | Fibonacci sphere viewpoint sampling and spherical-to-Cartesian camera math |
| `renderer/render_views.py` | `/OmniVision3D/renderer/render_views.py` | Mesh loading, normalization, DJI colour painting, Open3D offscreen rendering |
| `renderer/sky_compositor.py` | `/OmniVision3D/renderer/sky_compositor.py` | 100 procedural sky backgrounds + white-mask-replacement compositor |
| `renderer/thermal_filter.py` | `/OmniVision3D/renderer/thermal_filter.py` | Thermal simulation: CLAHE + iron/rainbow/grayscale LUT + bloom + noise + scanlines |
| `requirements.txt` | `/OmniVision3D/requirements.txt` | Full dev dependencies: torch, torchvision, open3d, trimesh, fastapi, pytest |
| `runs/detect/checkpoints/yolo/real_drone_detector/args.yaml` | `/OmniVision3D/runs/detect/checkpoints/yolo/real_drone_detector/args.yaml` | YOLO training args for the real-data experiment (imgsz=640, 50 epochs, patience=10) |
| `tests/__init__.py` | `/OmniVision3D/tests/__init__.py` | Empty package marker |
| `tests/test_inference.py` | `/OmniVision3D/tests/test_inference.py` | 4 pytest tests: Predictor class validity and draw_prediction shape/copy behaviour |
| `tests/test_renderer.py` | `/OmniVision3D/tests/test_renderer.py` | 15 pytest tests: camera_utils, sky_compositor, thermal_filter |
| `train_yolo.py` | `/OmniVision3D/train_yolo.py` | One-shot YOLOv8n training: 15 epochs, imgsz=512, Adam lr=0.001, data augmentation |
| `training/__init__.py` | `/OmniVision3D/training/__init__.py` | Empty package marker |
| `training/convert_to_yolo.py` | `/OmniVision3D/training/convert_to_yolo.py` | Converts synthetic renders to YOLOv8 format; auto-detects bbox from white-background renders |
| `training/dataset_loader.py` | `/OmniVision3D/training/dataset_loader.py` | Maps four dataset folders to binary labels; returns stratified 80/10/10 splits |
| `training/model.py` | `/OmniVision3D/training/model.py` | OmniVisionModel: MobileNetV2 + classifier head + detection head + embed() |
| `training/train.py` | `/OmniVision3D/training/train.py` | MobileNetV2 training: Adam, ReduceLROnPlateau, early stopping, confusion matrix, training curve |
