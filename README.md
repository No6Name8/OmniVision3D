# OmniVision3D

OmniVision3D is the vision module of a drone-based detection and response system.
It runs onboard a fixed-wing response drone on a Raspberry Pi and detects enemy
drones in real-time — both regular and thermal camera input.

The core idea: most systems need hundreds of labeled photos. OmniVision3D takes
a different approach — give it one 3D model of the target, and it builds its own
training dataset from scratch. Render the model from hundreds of synthetic
viewpoints, train on the results, deploy to the Pi.

**Current target: DJI Mini 4 Pro**

---

## Results

| Metric | Value |
|--------|-------|
| Synthetic training images | 7,200 |
| Test accuracy | 100% |
| False negative rate | 0% — no drone frames missed |
| False positive rate | 0% — no sky-only false alarms |
| ONNX model size | 8.5 MB (MobileNetV2) + 11.6 MB (YOLOv8n) |
| Dev machine FPS | 37 FPS scanning · 61 FPS locked |
| YOLO mAP50 at imgsz=320 | 0.995 (identical to imgsz=512) |

---

## How it works

**1. Input a 3D model**
Drop in a `.stl`, `.obj`, `.glb`, `.fbx`, or `.step` file of the target drone.

**2. Synthetic view generation**
Open3D renders the model from 100 Fibonacci-sphere viewpoints × 6 distances
= 600 renders. Fibonacci sampling gives uniform coverage with no pole clustering.
Vertex colors are applied to match the real DJI Mini 4 Pro (dark grey body,
black arms and motors).

**3. Sky compositing**
Each render is composited over one of 100 synthetic sky backgrounds (5 procedural
types × 20 seeds). Three sky variants per render = 1,800 composited drone images.

**4. Thermal simulation**
The same 1,800 images are passed through a thermal filter (iron LUT, CLAHE,
heat bloom, sensor noise, scanlines) to produce 1,800 thermal equivalents.

**5. Training — two models**
- **YOLOv8 nano** — fast 1-class detector, finds the drone in each frame, exports
  to ONNX at imgsz=320 for speed.
- **MobileNetV2** — binary classifier (drone / no_drone), confirms identity from
  the YOLO crop with 85% threshold and 3 consecutive frames before locking.

**6. Two-stage pipeline**
```
Frame → YOLO (18ms, imgsz=320) → crop → MobileNetV2 (7ms, 224×224 crop)
         ↓ nothing                        ↓ <85% or <3 frames
      SCANNING                         CONFIRMING
                                        ↓ ≥85% × 3 frames
                                      LOCKED → PID pitch/yaw → flight controller
```
When LOCKED the YOLO step is skipped every other frame (confirmer only, 7ms)
giving 61 FPS on a laptop CPU.

**7. Deployment**
ONNX models are copied to `pi_deploy/` and run via `onnxruntime` — no PyTorch on
the Pi. The camera runs in its own OS process with shared memory so inference
never blocks capture.

---

## Live camera test — laptop

This is the fastest way to see the pipeline working before touching any hardware.
You only need a laptop and a USB webcam (or the built-in camera).

### What you need

- Windows 10/11 or Linux laptop
- Any USB webcam, or the built-in laptop camera
- Python 3.10
- The DJI Mini 4 Pro (to fly in front of the camera)

---

### Step 1 — Install dependencies

Open a terminal in the project root and run:

```bash
pip install onnxruntime opencv-python numpy pillow pyyaml
```

That is all. No PyTorch, no CUDA, no heavy installs needed for the live test.

---

### Step 2 — Connect your camera

**Option A — Built-in laptop camera (easiest)**
No setup needed. The built-in camera is always index `0`.

**Option B — USB webcam**

1. Plug the webcam into any USB port.
2. Find its index:

```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera found at index {i}')
        cap.release()
"
```

The first number printed is your camera index. Usually `0` for built-in,
`1` for the first USB webcam.

**Option C — Phone as webcam (DroidCam / EpocCam)**

1. Install DroidCam on your phone and on the laptop.
2. Start the DroidCam app on both.
3. Connect over USB or Wi-Fi.
4. The phone camera appears as a regular webcam at index `1` or `2`.

---

### Step 3 — Run the test UI

```bash
cd pi_deploy
python test_ui.py --camera 0
```

Replace `0` with your camera index from Step 2 if needed.

A 960×480 window opens immediately:

```
┌──────────────────────────┬────────────────┐
│                          │ OMNIVISION3D   │
│   640×480 camera feed    │ Phase: SCANNING│
│   with detection overlay │ FPS:   37.0    │
│                          │ ──────────────  │
│                          │ YOLO:  0%      │
│                          │ ID:    0%      │
│                          │ Conf:  0/3     │
│                          │                │
│                          │ RECENT DETECTS │
└──────────────────────────┴────────────────┘
  Q=Quit  E=Enhance  R=Reset  S=Screenshot
```

The border is **green** — the system is scanning and nothing has been detected yet.

---

### Step 4 — Fly the DJI Mini 4 Pro in front of the camera

Follow these steps for a clean first detection:

**1. Set up the camera position**
- Place the laptop or webcam on a table pointing outward.
- You need a clear sky or plain background behind where the drone will fly.
- Distance from camera: start at **3–8 metres**. Too close and the drone fills
  the entire frame; too far and it becomes a tiny speck.

**2. Power on the DJI Mini 4 Pro**
- Insert a charged battery.
- Place the drone on a flat surface at least 2 metres in front of the camera.
- Power on — propellers do NOT need to spin for the detection to work.
  You can test with the drone on the ground first.

**3. Point the camera at the drone**
- Look at the live feed in the test UI window.
- The drone should be visible somewhere in the frame.
- You will see the border turn **yellow** — this means YOLO has detected
  something and is confirming identity.

**4. What the UI will show as the drone moves**

| State | Border colour | What it means |
|-------|--------------|---------------|
| SCANNING | Green | Nothing detected yet |
| CONFIRMING | Yellow | YOLO fired, waiting for 3 confirmed ID frames |
| LOCKED | Red border + green box | Identity confirmed — intercept committed |
| SEARCHING | Orange | Drone was lost, holding last position < 3s |

**5. Hold the drone still at 3–5 metres**
Within 3 frames (about 0.1 seconds) the system should lock on. You will see:
- A **green bounding box** appear around the drone
- The border turn **red**
- The status panel show `LOCKED` in green
- A line drawn from the frame centre to the drone centre (the intercept vector)

**6. Move the drone around**
- Move it slowly left, right, up, down.
- The bounding box tracks the drone in real time.
- If you move it behind something or out of frame, the border turns **orange**
  (SEARCHING) and after 3 seconds it goes back to SCANNING.

**7. Try different distances**
- **Very close (< 1m):** The drone fills the frame. YOLO may not detect because
  the shape is unfamiliar at extreme close range. Normal.
- **3–8m:** This is the optimal detection range. Should lock within 3 frames.
- **10–20m:** The drone appears smaller. Detection still works but may take 1–2
  extra seconds to confirm.

---

### Step 5 — Keyboard controls during the test

| Key | Action |
|-----|--------|
| `Q` | Quit the UI |
| `E` | Toggle CLAHE image enhancement on/off. Try this in low light |
| `R` | Reset the pipeline — clears lock and goes back to SCANNING |
| `S` | Save a screenshot of the current camera frame |

---

### Step 6 — Test with a video file (no camera needed)

If you recorded a video of the drone earlier, you can run detection on it:

```bash
python test_ui.py --video path/to/your/video.mp4
```

The UI is identical. Press `Q` to stop, `S` to screenshot any frame.

---

### Step 7 — Adjust confidence if needed

If the system is not locking on, open `config.yaml` and lower the thresholds:

```yaml
yolo:
  confidence_threshold: 0.40    # default 0.50 — lower to detect more

identity_confirmation:
  confidence_threshold: 0.75    # default 0.85 — lower to lock faster
  consecutive_required: 2       # default 3 — fewer frames needed
```

If the system is giving false locks (locking on birds, trees, people), raise them:

```yaml
yolo:
  confidence_threshold: 0.60

identity_confirmation:
  confidence_threshold: 0.90
  consecutive_required: 4
```

---

## Full pipeline — train from scratch

### 1. Place your 3D model in `models/`

### 2. Render synthetic views
```bash
python -m renderer.render_views --config configs/default.yaml --obj models/dji-mini-4-pro.stl
```

### 3. Composite over sky backgrounds
```bash
python -m renderer.sky_compositor \
  --input dataset/raw/dji-mini-4-pro/ \
  --output dataset/processed/DJI_Mini_4_Pro/ \
  --backgrounds dataset/backgrounds/
```

### 4. Generate thermal variants
```bash
python -m renderer.thermal_filter \
  --input dataset/processed/DJI_Mini_4_Pro/ \
  --output dataset/processed/DJI_Mini_4_Pro_thermal/ \
  --palette iron
```

### 5. Generate no_drone class
```bash
python -c "
import os, random
import numpy as np
from PIL import Image, ImageFilter
os.makedirs('dataset/processed/no_drone', exist_ok=True)
backgrounds = os.listdir('dataset/backgrounds')
count = 0
for bg_file in backgrounds:
    bg = Image.open(f'dataset/backgrounds/{bg_file}').resize((512, 512))
    for i in range(18):
        arr = np.array(bg).astype(np.float32)
        arr = np.clip(arr * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(f'dataset/processed/no_drone/{count:04d}.png')
        count += 1
print(f'Generated {count} no_drone images')
"
```

### 6. Train MobileNetV2
```bash
python -m training.train --config configs/default.yaml
```

### 7. Convert dataset to YOLO format
```bash
python training/convert_to_yolo.py
```

### 8. Train YOLOv8
```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='yolo_dataset/dataset.yaml',
    epochs=50, imgsz=512, batch=16,
    name='dji_detector', project='checkpoints/yolo',
    patience=10, device='cpu', workers=0,
)
"
```

### 9. Export YOLO to ONNX
```bash
python -c "
from ultralytics import YOLO
import shutil, glob
model = YOLO(glob.glob('runs/**/best.pt', recursive=True)[0])
model.export(format='onnx', imgsz=320, simplify=True)
shutil.copy(glob.glob('runs/**/best.onnx', recursive=True)[0], 'pi_deploy/vision/yolo_dji.onnx')
print('Done')
"
```

### 10. Export MobileNetV2 to ONNX
```bash
python -c "
import torch, shutil
from training.model import OmniVisionModel
model = OmniVisionModel(num_classes=2)
model.load_state_dict(torch.load('checkpoints/best.pt', map_location='cpu'))
model.eval()
torch.onnx.export(model, torch.randn(1,3,224,224), 'checkpoints/omnivision3d.onnx',
    input_names=['image'], output_names=['prediction'],
    dynamic_axes={'image': {0: 'batch'}}, opset_version=12)
shutil.copy('checkpoints/omnivision3d.onnx', 'pi_deploy/vision/omnivision3d.onnx')
print('Done')
"
```

---

## Raspberry Pi deployment

Copy the deployment package to the Pi and install:

```bash
scp -r pi_deploy/ pi@<PI_IP>:~/omnivision3d/
ssh pi@<PI_IP>
cd ~/omnivision3d
pip install -r requirements_pi.txt
```

Run in simulation mode first (no hardware commands sent):

```bash
python main.py --sim
```

Run with GPS coordinates when ready to fly:

```bash
python main.py --lat 24.7136 --lon 46.6753
```

Run against a recorded video:

```bash
python main.py --video tests/test_video.mp4 --sim
```

---

## Project structure

```
OmniVision3D/
├── models/                  3D input files (.stl, .obj, .glb, .fbx, .step)
├── renderer/                Rendering pipeline
│   ├── render_views.py      Fibonacci sphere sampling, vertex colour painting
│   ├── sky_compositor.py    Sky background generation and compositing
│   ├── thermal_filter.py    Thermal camera simulation (CLAHE, iron LUT, bloom)
│   └── camera_utils.py      Viewpoint math
├── training/                Dataset and training
│   ├── train.py             MobileNetV2 training loop, early stop, confusion matrix
│   ├── model.py             OmniVisionModel (MobileNetV2 binary classifier)
│   ├── dataset_loader.py    Folder→label mapping, train/val/test split
│   └── convert_to_yolo.py   Raw render → YOLO bbox labels
├── inference/               Standalone inference scripts
├── pi_deploy/               Self-contained Raspberry Pi package
│   ├── vision/
│   │   ├── yolo_detector.py       YOLOv8 ONNX wrapper (imgsz=320)
│   │   ├── identity_confirmer.py  MobileNetV2 crop classifier
│   │   ├── pipeline.py            SCANNING→CONFIRMING→LOCKED state machine
│   │   ├── camera_stream.py       Camera in separate OS process + shared memory
│   │   └── frame_enhancer.py      CLAHE enhancement toggle
│   ├── control/tracker.py         PID pitch/yaw + LOCKED/SEARCHING/ABORT states
│   ├── navigation/gps_nav.py      GPS navigation (MAVLink stubs)
│   ├── test_ui.py                 960×480 live test window (laptop use)
│   ├── main.py                    Full mission loop
│   ├── config.yaml                All thresholds and mission settings
│   └── requirements_pi.txt        onnxruntime opencv-python numpy pillow pyyaml
├── checkpoints/             Saved weights and training artifacts
├── configs/default.yaml     Rendering and training configuration
├── tests/                   19 unit and integration tests
└── yolo_dataset/            Generated YOLO format dataset
```

---

## Stack

- **Python 3.10**
- **Open3D + Trimesh** — 3D mesh loading and offscreen rendering
- **PyTorch + torchvision** — MobileNetV2 training
- **Ultralytics YOLOv8** — YOLO detection training
- **ONNX + onnxruntime** — Pi deployment, no PyTorch needed on device
- **OpenCV** — image processing, video I/O, overlay rendering
- **NumPy / Pillow** — array and image manipulation
- **PyYAML** — configuration

---

## Tests

```bash
python -m pytest tests/ -v
```

19 tests covering the renderer (camera utils, sky compositor, thermal filter)
and inference engine (predictor, visualizer). All passing.

---

## Friendly Fire Prevention — Deconfliction Theory

When multiple interceptor drones launch together
each one runs OmniVision3D independently.
Without a safety system each drone could detect
its teammates and attempt to intercept them.

Our approach is GPS-based deconfliction:

**BEFORE LAUNCH:**
The command center sends each drone a deconfliction
manifest — a list of the GPS coordinates and flight
paths of all other friendly drones in the mission.
No signal is broadcast during flight.

**DURING FLIGHT:**
Before committing to LOCKED state the drone
calculates the estimated GPS position of the
detected object using its own GPS position,
compass heading, camera field of view, and
the object's position in the frame.

This estimated position is compared against
the expected positions of all friendly drones
at this exact moment in time.

If the detected object is within 20 meters of
a known friendly position the target is skipped
and the drone returns to SCANNING.

**WHY THIS CANNOT BE JAMMED:**
Traditional IFF systems broadcast a radio signal.
Jamming that signal makes all drones look like
threats. Our system broadcasts nothing during flight.
There is no signal to jam. The deconfliction data
was loaded before launch and exists only as a file
on each drone's local storage. An enemy cannot
interfere with math.

**WHY THIS CANNOT BE SPOOFED:**
Traditional IFF systems can be spoofed by
broadcasting a fake friendly signal. Our system
accepts no signals during flight. A fake broadcast
would be ignored completely.

**CURRENT STATUS:**
Deconfliction logic is designed and documented.
Full implementation requires live GPS and compass
integration with the Pixhawk flight controller.
This is planned for the next hardware integration
phase after motors and ESCs are installed.

---

## Roadmap

- **MAVLink flight controller wiring** — replace PID stubs with RC-override commands
- **Serial GPS** — replace position stubs with u-blox or dronekit reads
- **Night / thermal camera** — the model was trained on thermal data; a second
  capture index can feed a parallel inference pass
- **Multi-drone tracking** — extend to simultaneous tracking of multiple targets
- **Offline maps** — pre-downloaded GeoTIFFs for autonomous search patterns
