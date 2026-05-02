# OmniVision3D — Quick Start Guide

## Prerequisites

- Python 3.10+ on development machine
- Raspberry Pi 4 (2 GB RAM minimum) for deployment
- A DJI Mini 4 Pro 3D mesh file (.obj, .glb, .stl, or .fbx)

---

## A. Development Machine Setup

```bash
git clone <repo>
cd OmniVision3D
pip install -r requirements.txt
```

---

## B. Generate Training Data

### 1. Render 3D views

```bash
python -m renderer.render_views --model path/to/dji_mini_4_pro.obj
```

Output: `dataset/raw/dji-mini-4-pro/` (600 white-background renders)

### 2. Composite sky backgrounds

```bash
python -m renderer.sky_compositor
```

Output: `dataset/processed/DJI_Mini_4_Pro/` (1800 composited images)

### 3. (Optional) Apply thermal filter

```bash
python -m renderer.thermal_filter \
    --input  dataset/processed/DJI_Mini_4_Pro/ \
    --output dataset/processed/DJI_Mini_4_Pro_thermal/ \
    --palette iron
```

### 4. Convert to YOLO format

```bash
python training/convert_to_yolo.py
```

Output: `yolo_dataset/` with 80/10/10 train/val/test split and `dataset.yaml`.

---

## C. Train Models

### Train YOLO detector (primary model — use this on Pi)

```bash
python train_yolo.py
```

Output: `checkpoints/yolo/dji_detector/weights/best.pt`

Export to ONNX:
```bash
yolo export model=checkpoints/yolo/dji_detector/weights/best.pt format=onnx imgsz=320
```

Copy the exported `.onnx` to `pi_deploy/vision/yolo_dji.onnx`.

### Train MobileNetV2 classifier (optional — for enemy ID)

```bash
python -m training.train --config configs/default.yaml
```

Output: `checkpoints/best.pt`, `checkpoints/training_curve.png`

Export to ONNX:
```bash
python -c "
import torch
from training.model import load_model
model = load_model('checkpoints/best.pt', num_classes=2, device=torch.device('cpu'))
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'checkpoints/omnivision3d.onnx', opset_version=12,
    input_names=['input'], output_names=['output'])
print('Exported.')
"
```

---

## D. Run Tests

```bash
pytest tests/ -v
```

Tests cover: Predictor forward pass, draw_prediction shape/copy, camera_utils geometry, sky_compositor output count/dimensions, thermal_filter palettes.

---

## E. Deploy to Raspberry Pi

### 1. Copy files to Pi

```bash
scp -r pi_deploy/ pi@<PI_IP>:~/omnivision3d/
```

The minimum set of files the Pi needs:

```
pi_deploy/
├── drone_main.py          ← lean flight script (recommended for actual flight)
├── main.py                ← full pipeline (use for testing with display)
├── config.yaml            ← all parameters
├── requirements_pi.txt
├── vision/
│   ├── camera_stream.py
│   ├── yolo_detector.py
│   ├── drone_classifier.py
│   ├── enemy_identifier.py
│   ├── identity_confirmer.py
│   ├── pipeline.py
│   └── yolo_dji.onnx      ← YOLO model (copy from checkpoints/)
├── control/
│   ├── motor_controller.py
│   ├── motor_calibration.py
│   └── tracker.py
├── navigation/
│   └── gps_nav.py
└── enemies/
    └── enemies.yaml       ← enemy model list (can be empty initially)
```

### 2. Install on Pi

```bash
pip install -r requirements_pi.txt
```

### 3. Configure

Edit `config.yaml`:
- Set `simulation_mode: false` (and `motor_control.simulation_mode: false`) when hardware is wired
- Set `camera.index` to the correct `/dev/videoN` index
- Set `navigation.home_lat` and `home_lon` to your base coordinates
- Set `launch_listener.wait_for_launch: true` if you want operator confirmation before intercept

### 4. (One-time) Calibrate motors

```bash
cd ~/omnivision3d
python control/motor_calibration.py
```

Follow the prompts — it spins each motor in turn and asks which one moved. Writes confirmed mapping and flip flags to config.yaml.

---

## F. Flight Operations

### Option 1 — Lean flight script (recommended for real missions)

```bash
python drone_main.py --sim       # simulation mode
python drone_main.py             # live mode (config.yaml controls simulation_mode)
```

Kill switch (from laptop over SSH while mission is running):
```bash
ssh pi@<PI_IP> touch ~/omnivision3d/KILL
```

### Option 2 — Full pipeline (includes display, tracker, GPS)

```bash
python main.py --sim
python main.py --lat 24.7136 --lon 46.6753   # with GPS navigation
python main.py --wait-launch                  # hold until laptop sends LAUNCH
```

---

## G. Three-Unit Deployment (Ground Pi + Laptop + Interceptor Pi)

### Unit 1 — Ground detection Pi

```bash
cd ~/omnivision3d/pi_deploy
python ground_station/main.py \
    --camera 0 \
    --model  vision/yolo_dji.onnx \
    --gps    /dev/serial/by-id/usb-FTDI_...-BG01OJPV-if00-port0 \
    --laser  /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
    --compass /dev/serial/by-id/usb-FTDI_...-AQ045IV6-if00-port0 \
    --declination 4.0
```

Opens Tkinter UI showing live sensor values and YOLO camera feed. Sends UDP alert to laptop (port 5555) when drone is detected for 3+ consecutive frames. Omit `--gps`, `--laser`, or `--compass` to disable individual sensors.

### Unit 2 — Laptop monitor

```bash
cd laptop_monitor/
python monitor.py --drone <INTERCEPTOR_PI_IP>
python monitor.py --sim                         # simulation (commands to localhost)
```

Receives alerts from ground Pi on port 5555. Displays confidence and compass heading.
- **ENTER** — send LAUNCH to interceptor Pi (port 5556)
- **A** — send ABORT
- **Q** — quit

### Unit 3 — Interceptor Pi

```bash
cd ~/omnivision3d
python drone_main.py            # lean flight (reads config.yaml for simulation_mode)
python drone_main.py --sim      # force simulation mode
python main.py --wait-launch    # full pipeline — holds until laptop sends LAUNCH
```

Waits for LAUNCH command on port 5556, then begins scanning and intercept loop.

---

## H. REST API (Laptop Development)

```bash
uvicorn api.server:app --reload
```

```bash
# Health check
curl http://localhost:8000/health

# Predict from image file
curl -X POST http://localhost:8000/predict \
  -F "file=@tests/test_images/drone_001.png"
# → {"label": "dji_mini_4_pro", "confidence": 0.97}
```

---

## I. Performance Reference

| Platform | Model | Input | FPS |
|----------|-------|-------|-----|
| Laptop CPU (benchmark) | YOLO nano | 320px | ~32 FPS |
| Raspberry Pi 4 (expected) | YOLO nano | 320px | ~10–15 FPS |
| Raspberry Pi 4 (expected) | MobileNetV2 | 224px | ~15–20 FPS |

YOLO at 320px input is the recommended configuration for Pi. Reduce to 224px if FPS is insufficient.

---

## J. Troubleshooting

**Camera not opening:**
```bash
ls /dev/video*
python drone_main.py --camera 1   # try index 1 if 0 fails
```

**ONNX model not found:**
- Verify `config.yaml` `yolo.model_path` is correct relative to `pi_deploy/`
- Check file exists: `ls pi_deploy/vision/yolo_dji.onnx`

**Low FPS on Pi:**
- Reduce `yolo.input_size` from 320 to 224 in config.yaml
- Reduce `camera.width/height` to 320×240

**Motor not responding:**
1. Check `simulation_mode: false` in both top-level and `motor_control` sections
2. Run `python control/motor_calibration.py` to verify channel mapping
3. Check `motor_mapping.calibrated: true` after calibration

**UDP alerts not received:**
- Verify no firewall blocking port 5555 (ground → laptop) or 5556 (laptop → interceptor)
- Run `python laptop_monitor/monitor.py` with `--verbose` to see raw packets
