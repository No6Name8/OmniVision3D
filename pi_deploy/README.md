# OmniVision3D — Pi Deploy

Onboard vision system for a fixed-wing response drone. Detects Shahed 136 UAVs in
real-time using a MobileNetV2 classifier exported to ONNX and running on
Raspberry Pi 4 via onnxruntime.

## What this folder does

```
pi_deploy/
├── vision/           Camera feed + ONNX inference
├── control/          PID tracker — converts pixel offset to pitch/yaw
├── navigation/       GPS navigation stubs (simulation today, MAVLink later)
├── logs/             Per-mission CSV and text logs
├── main.py           Master mission loop
└── config.yaml       All tunable parameters
```

Mission flow: receive GPS coordinates → navigate to area → activate camera →
scan frames → lock on when confidence > 85% → send pitch/yaw corrections to
flight controller.

## Requirements

- Raspberry Pi 4 (2 GB RAM minimum, 4 GB recommended)
- Python 3.9+
- Camera module or USB webcam

## Install

```bash
# On the Raspberry Pi:
pip install -r requirements_pi.txt
```

Total install size is under 200 MB. No GPU or CUDA required.

## Copy to Raspberry Pi

```bash
scp -r pi_deploy/ pi@<PI_IP>:~/omnivision3d/
```

## Run

```bash
cd ~/omnivision3d

# Simulation mode (no hardware needed, uses config.yaml defaults)
python main.py --lat 24.7136 --lon 46.6753

# Test with a video file instead of live camera
python main.py --lat 24.7136 --lon 46.6753 --video path/to/test.mp4

# Vision module standalone (live camera, display window)
python vision/pi_predict.py --show

# Vision module standalone (video file)
python vision/pi_predict.py --video path/to/test.mp4 --show
```

Press `Ctrl-C` to end the mission cleanly. All logs are written to `logs/`.

## Configuration

Edit `config.yaml` before deploying:

| Key | Default | Notes |
|-----|---------|-------|
| `confidence_threshold` | 0.85 | Raise to reduce false positives |
| `camera_index` | 0 | Change if USB camera is not /dev/video0 |
| `frame_width/height` | 640×480 | Lower for more FPS on Pi Zero |
| `lost_target_timeout` | 2.0 s | Hold last correction before returning to scan |
| `simulation_mode` | true | Set false when hardware is wired |

## Performance

Benchmarked on a desktop CPU simulating Pi 4 with 4 threads:

| Platform | Avg inference | FPS |
|----------|--------------|-----|
| Desktop (benchmark) | 2.1 ms | 476 |
| Raspberry Pi 4 (expected) | ~55–65 ms | ~15–20 |

Model size: **8.5 MB** (MobileNetV2, ONNX opset 12).

## Logs

Every mission writes three log files:

- `logs/mission.log` — state transitions, lock-on events, errors
- `logs/detections.log` — every confident drone detection with confidence and offset
- `logs/frames.csv` — per-frame label, confidence, pitch/yaw, FPS (for post-analysis)

## Current status: simulation mode

All hardware calls are stubbed. The system prints what it would send to the
flight controller without touching any hardware. Set `simulation_mode: false`
in `config.yaml` when wiring is complete.

## Future upgrades

**GPS / compass wiring**
Replace `navigation/gps_nav.py`'s stubs with serial reads from a u-blox module
or dronekit `vehicle.location`. The `haversine()` function and acceptance-radius
logic are already in place.

**Flight controller**
Replace `control/tracker.py`'s `_send_correction()` stub with MAVLink
`RC_CHANNELS_OVERRIDE` commands via pymavlink. PID gains (P=0.1, I=0.01, D=0.05)
are a starting point — tune against actual airframe response.

**Thermal camera**
The model was trained on both RGB and thermal synthetic data. A second
`VideoCapture` index for a thermal camera can be added to `pi_predict.py`
with a separate inference pass.

**Bounding box**
The backbone has a detection head (`det_head`) exporting
`(cx, cy, w, h, objectness)`. Export it alongside the classifier and wire
the output into `DetectionResult.offset` for precise pixel targeting rather
than the current frame-centre fallback.

**Offline maps**
For autonomous area-search patterns, integrate a lightweight tile server
or pre-downloaded GeoTIFF for waypoint generation without internet.
