# Zarqaa Al-Yamama & Sahm
### Advanced Counter-UAS Intercept System

![Status](https://img.shields.io/badge/Status-Active%20Development-cyan)
![License](https://img.shields.io/badge/License-Proprietary-red)
![Language](https://img.shields.io/badge/Python-3.10-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20Linux%20%7C%20Windows-lightgrey)
![AI](https://img.shields.io/badge/AI-YOLOv8%20ONNX-green)

> **A sovereign Saudi solution to a sovereign Saudi problem.**
> An autonomous, unjammable, low-cost counter-drone system engineered
> to reverse the economic asymmetry of modern drone warfare.

---

## The Problem — Brutal Math

Traditional air defense is losing the economic war against low-cost kamikaze drones.

| Asset | Unit Cost |
|-------|-----------|
| Shahed-136 kamikaze drone | ~$20,000 |
| Patriot PAC-3 interceptor missile | ~$3,000,000 |
| **Kill ratio disadvantage** | **150 : 1** |

Modern commercial and military drones compound this with three further failures of legacy systems:

- **Radar blind spot:** Small plastic airframes have near-zero radar cross-section. Standard air-defense radar cannot reliably track them below 50m AGL.
- **Swarm saturation:** A salvo of 10 drones overwhelms a single launcher. Reload time is measured in minutes; swarm replacement in seconds.
- **No sovereign kill chain:** Dependence on foreign interceptor munitions means every kill decision carries a geopolitical cost.

This system addresses all three.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│  ZARQAA AL-YAMAMA — Ground Detection Hub            │
│  Raspberry Pi · Camera · GPS · Laser · Compass      │
│                                                     │
│  YOLO detection → haversine targeting               │
│  → target GPS coords via UDP :5555 alert ──────┐   │
└────────────────────────────────────────────────+───┘
                                                 │ UDP
┌────────────────────────────────────────────────+───┐
│  LAPTOP MONITOR — Operator Station                  │
│  Receives alert → operator confirms ENTER           │
│  → LAUNCH command via UDP :5556 ───────────────┐   │
└────────────────────────────────────────────────+───┘
                                                 │ UDP / Fiber
┌────────────────────────────────────────────────+───┐
│  SAHM — Interceptor Drone                          │
│  Raspberry Pi · Fixed-wing airframe                │
│  Camera → YOLO → PID → 4-motor intercept           │
└────────────────────────────────────────────────────┘
```

---

## Component I — Zarqaa Al-Yamama (Detection Hub)

A stationary ground unit that bypasses radar's limitations using three fused detection modalities.

### Detection Pillars

| Pillar | Technology | Notes |
|--------|-----------|-------|
| **EO/IR Vision** | YOLOv8 nano ONNX · 320 px · 50% threshold | Runs on CPU, no GPU required |
| **Kinematic Targeting** | GPS + compass + laser rangefinder | Haversine destination_point → target GPS |
| **Acoustic (planned)** | Directional mic array · 3–8 kHz motor signature | Architecture designed, hardware TBD |

### Sensor Stack

| Sensor | Protocol | Baud | Data written |
|--------|----------|------|-------------|
| GPS (FTDI) | NMEA $GPRMC / $GNRMC | 9600 | `lat`, `lon`, `time_utc` |
| Laser rangefinder (CH340) | Binary `0xAE 0xA7 … 0xBC 0xBE` | 9600 | `distance_m` |
| Compass (FTDI) | Text `$C heading A … *HH` | 9600 | `compass_raw` |
| USB camera | OpenCV V4L2 | — | `frame`, detection state |

### Targeting Formula

```
compass_used = (compass_raw + declination + user_offset) % 360
target_lat, target_lon = haversine_destination(own_lat, own_lon,
                                               compass_used,
                                               distance_m)
```

Accuracy: spherical Earth model, <0.1% error at all relevant intercept ranges.

---

## Component II — Sahm (Interceptor Drone)

A fixed-wing intercept drone running entirely on a Raspberry Pi.
All heavy computation stays on the ground. The drone carries only what is needed for the kill.

### Key Design Principles

**Radio silence after launch.**
Once airborne, Sahm transmits nothing. No RF emissions means no jamming surface,
no GPS spoofing vector, and no direction-finding by the adversary.

**Vision-only terminal guidance.**
A 37 FPS camera-based tracker drives four motor channels via PID.
The target is centred in-frame. The drone becomes the projectile.

**Kinetic warhead — no explosives.**
The airframe nose is the weapon. No fuze, no payload, no hazmat classification.
Unit cost target: under 10,000 SAR.

**Second-attempt capability.**
On intercept failure the drone executes a 180° return pass and re-engages.
Missiles cannot do this.

**Edge AI on ground — dumb body, smart base.**
YOLOv8 training, MobileNetV2 training, and all heavy inference configuration
runs on the laptop. The Pi runs only the pre-exported ONNX models.

### Flight Phase State Machine

| Phase | Trigger | Motor Command |
|-------|---------|--------------|
| `SCANNING` | No detection | Cruise (1500 µs) |
| `CONFIRMING` | YOLO ≥ 50%, consecutive < 3 | Cruise, accumulating |
| `LOCKED` | YOLO ≥ 50% × 3 frames | Intercept (1900 µs) + PID |
| `SEARCHING` | Target lost, < 3 s | Hold heading, reset PID |
| `NAVIGATING` | Target lost 3–10 s | Navigate to last known GPS |
| `ABORT` | Target lost > 10 s | Return to base |

### PID Motor Controller

Four independent channels (front-left, front-right, rear-left, rear-right)
driven by pixel-offset error signals from the detection bounding box centre:

```
dx = target_cx - frame_cx        # horizontal error
dy = target_cy - frame_cy        # vertical error

pitch_cmd = Kp·dy + Ki·∫dy + Kd·Δdy    # PWM µs
yaw_cmd   = Kp·dx + Ki·∫dx + Kd·Δdx
```

Default gains: `Kp=0.5  Ki=0.01  Kd=0.1`
Motor range: 1000–2000 µs PWM. MAVLink RC_CHANNELS_OVERRIDE wiring: next milestone.

---

## The AI Pipeline

### Training Data — Fully Synthetic

No labeled photographs. Zero real-world annotation effort.

```
1 × 3D model (DJI Mini 4 Pro .STL)
    │
    ▼ render_views.py
600 renders (100 Fibonacci-sphere viewpoints × 6 distances)
    │
    ▼ sky_compositor.py
1,800 composited images (5 sky types × 20 seeds × 3 per render)
    │
    ▼ thermal_filter.py
1,800 thermal variants (iron LUT + CLAHE + heat bloom + sensor noise)
    │
    ▼  train.py + train_yolo.py
    ├── YOLOv8 nano → yolo_dji.onnx (11.6 MB)
    └── MobileNetV2 → omnivision3d.onnx (8.5 MB)
```

### Reported Results (synthetic test set)

| Metric | Value |
|--------|-------|
| Test accuracy | 100% |
| False negative rate | 0% |
| False positive rate | 0% |
| YOLO mAP50 (imgsz=320) | 0.995 |
| ONNX model — YOLOv8n | 11.6 MB |
| ONNX model — MobileNetV2 | 8.5 MB |
| FPS — laptop CPU (scanning) | 37 FPS |
| FPS — laptop CPU (locked) | 61 FPS |
| YOLO inference time | ~18 ms |
| MobileNetV2 inference time | ~7 ms |

### Two-Stage Pipeline

```
Frame (BGR 640×480)
  └─ YoloDetector.detect()     [18 ms, imgsz=320]
       └─ Detection found?
            ├── No  → SCANNING
            └── Yes → consecutive++
                       ├── < 3 frames → CONFIRMING
                       └── ≥ 3 frames → LOCKED
                            └─ IdentityConfirmer.confirm()  [7 ms, crop 224×224]
                                 └─ EnemyIdentifier (every 5 frames)
```

When LOCKED, YOLO is skipped every other frame (confirmer only) → 61 FPS.

---

## Friendly Fire Prevention — Deconfliction

When multiple Sahm drones launch simultaneously each runs OmniVision3D
independently. Without deconfliction, friendly drones could detect and
target each other.

**Our approach: pre-loaded GPS manifest, zero in-flight comms.**

Before launch the command center loads each drone with a deconfliction
manifest — the expected GPS positions and flight paths of all other friendly
units at every second of the mission. During flight, before committing to
LOCKED, the drone calculates the estimated GPS of the detected object and
compares it against the manifest. A match within 20 m → skip, return to SCANNING.

**Why this cannot be jammed:** The manifest is a local file. Nothing is
broadcast during flight. There is no signal to intercept or disrupt.

**Why this cannot be spoofed:** The drone accepts no incoming signals after
launch. A fake IFF broadcast is silently ignored.

**Current status:** Deconfliction logic is designed and documented.
Full implementation requires live MAVLink GPS integration — next milestone.

---

## Installation

### Laptop / Development Machine

```bash
git clone https://github.com/No6Name8/OmniVision3D.git
cd OmniVision3D
pip install onnxruntime opencv-python numpy pillow pyyaml open3d trimesh torch torchvision ultralytics
```

### Raspberry Pi (Detection Hub or Interceptor)

```bash
pip install -r pi_deploy/requirements_pi.txt --break-system-packages
```

---

## Quick Start — Live Detection Test

No hardware required. Laptop camera only.

```bash
python pi_deploy/test_ui.py --camera 0
```

A 960×480 window opens. Hold a DJI Mini 4 Pro (or a printed photo of one)
in front of the camera at 3–8 metres. The system locks in under 3 frames.

| Border color | State | Meaning |
|-------------|-------|---------|
| Green | SCANNING | Watching, nothing detected |
| Yellow | CONFIRMING | Possible target, accumulating frames |
| Red | LOCKED | Identity confirmed, intercept committed |
| Orange | SEARCHING | Target lost, holding < 3 s |

---

## Running the Ground Station

```bash
python pi_deploy/ground_station/main.py \
  --camera 0 \
  --model  vision/yolo_dji.onnx \
  --gps    /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG01OJPV-if00-port0 \
  --laser  /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --compass /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AQ045IV6-if00-port0 \
  --declination 4.0
```

All sensor arguments are optional. Omit any port to disable that sensor.

---

## Running the Minimal Fullscreen View (Pi display)

```bash
python pi_deploy/pi_view.py \
  --camera 0 \
  --model  vision/yolo_dji.onnx \
  --gps    /dev/serial/by-id/... \
  --laser  /dev/serial/by-id/... \
  --compass /dev/serial/by-id/... \
  --declination 4.0
```

Fullscreen camera feed. Shows only: green bounding box, DRONE DETECTED,
laser distance, and target GPS. Nothing else on screen.

---

## Running the Laptop Monitor (Operator Station)

```bash
python laptop_monitor/monitor.py --drone 192.168.1.200
```

Listens on UDP :5555 for DRONE_DETECTED alerts from the ground unit.
Press ENTER to launch, A to abort, Q to quit.

---

## Training From Scratch

```bash
# 1. Render synthetic views
python -m renderer.render_views --config configs/default.yaml --obj models/dji-mini-4-pro.stl

# 2. Composite over sky backgrounds
python -m renderer.sky_compositor \
  --input  dataset/raw/dji-mini-4-pro/ \
  --output dataset/processed/DJI_Mini_4_Pro/ \
  --backgrounds dataset/backgrounds/

# 3. Thermal variants
python -m renderer.thermal_filter \
  --input   dataset/processed/DJI_Mini_4_Pro/ \
  --output  dataset/processed/DJI_Mini_4_Pro_thermal/ \
  --palette iron

# 4. Train MobileNetV2 classifier
python -m training.train --config configs/default.yaml

# 5. Convert to YOLO format and train YOLOv8
python training/convert_to_yolo.py
# then train with ultralytics — see docs/QUICK_START.md

# 6. Export both models to ONNX
# see docs/QUICK_START.md for export commands
```

---

## Tests

```bash
python -m pytest tests/ -v
```

19 tests covering: renderer camera utils, sky compositor, thermal filter,
YOLO predictor, and visualizer. All passing.

---

## Project Structure

```
OmniVision3D/
├── models/                    3D input files (.stl, .obj, .glb, .fbx, .step)
├── renderer/                  Synthetic data generation pipeline
│   ├── render_views.py        Fibonacci sphere sampling + vertex color painting
│   ├── sky_compositor.py      Procedural sky generation and compositing
│   ├── thermal_filter.py      Iron LUT + CLAHE + bloom + sensor noise simulation
│   └── camera_utils.py        Viewpoint math utilities
├── training/                  Model training
│   ├── train.py               MobileNetV2 training loop
│   ├── model.py               OmniVisionModel (MobileNetV2 binary classifier)
│   ├── dataset_loader.py      Folder-to-label mapping, train/val/test split
│   └── convert_to_yolo.py     Renders → YOLO bbox labels
├── pi_deploy/                 Self-contained Raspberry Pi package
│   ├── ground_station/        Detection hub (GPS + laser + compass + YOLO)
│   │   ├── main.py            Tkinter UI, thread orchestration
│   │   ├── shared_state.py    Thread-safe state dataclass
│   │   ├── targeting.py       Haversine GPS targeting + compass offset
│   │   ├── detection.py       YOLO detection thread
│   │   └── sensors/           GPS, laser, compass serial readers
│   ├── vision/                AI inference modules
│   │   ├── yolo_detector.py   YOLOv8 ONNX wrapper
│   │   ├── identity_confirmer.py  MobileNetV2 crop classifier
│   │   ├── pipeline.py        Four-phase state machine
│   │   └── camera_stream.py   Camera in separate OS process
│   ├── control/               Motor control
│   │   ├── motor_controller.py    PID pitch/yaw → PWM (MAVLink stub)
│   │   └── tracker.py         Intercept state machine
│   ├── navigation/gps_nav.py  GPS navigation (MAVLink stub)
│   ├── pi_view.py             Minimal fullscreen detection display
│   ├── test_ui.py             960×480 laptop test window
│   ├── main.py                Full mission loop
│   └── config.yaml            All tunable parameters
├── laptop_monitor/            Operator workstation
│   ├── monitor.py             Alert receiver + launch commander
│   ├── alert_receiver.py      UDP :5555 listener
│   └── launch_command.py      UDP :5556 launch/abort sender
├── website/                   Marketing landing page (Arabic RTL)
├── docs/                      Full technical documentation
│   ├── SYSTEM_DOCUMENTATION.md
│   └── QUICK_START.md
├── configs/default.yaml       Rendering and training configuration
└── tests/                     19 unit and integration tests
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| AI Detection | YOLOv8 nano (Ultralytics) → ONNX |
| AI Classification | MobileNetV2 (PyTorch/torchvision) → ONNX |
| Inference (Pi) | onnxruntime CPU — no PyTorch, no CUDA |
| 3D Rendering | Open3D + Trimesh |
| Computer Vision | OpenCV |
| GPS Math | Haversine spherical Earth formula |
| Serial Sensors | pyserial |
| UI (ground station) | Tkinter + Pillow |
| UI (laptop test) | OpenCV window |
| Communication | UDP JSON packets |
| Configuration | PyYAML |
| Language | Python 3.10 |
| Hardware | Raspberry Pi (any model with USB) |

---

## Roadmap

- **MAVLink RC_CHANNELS_OVERRIDE** — replace PID stub with live motor commands
- **u-blox GPS integration** — replace navigation stubs with real serial reads
- **Thermal camera channel** — second inference pass on IR input (model already trained)
- **Acoustic array** — directional mic detection layer for passive sensing
- **Multi-target tracking** — simultaneous track of swarm members
- **Deconfliction live test** — GPS manifest validation with two Pi units

---

## Strategic Scope

| Application | Description |
|-------------|-------------|
| Base & facility defense | Perimeter surveillance against kamikaze UAS |
| Critical infrastructure | Refineries, ports, power plants, water infrastructure |
| Frontline counter-armor | Adaptation for anti-armor drone swarm defense |
| Naval defense | Ship-based variant for low-altitude maritime threats |

---

*Built in Saudi Arabia. Designed for Saudi skies.*
