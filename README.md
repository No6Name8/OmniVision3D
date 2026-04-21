# OmniVision3D

OmniVision3D is the vision module of a drone-based detection and response system. It runs onboard a fixed-wing response drone on a Raspberry Pi and detects enemy drones in real-time from aerial footage — both regular and thermal camera input.

The core problem: most object recognition systems need hundreds of labeled photos. OmniVision3D takes a different approach — give it one 3D model of the target, and it builds its own training dataset from scratch. Render the model from hundreds of synthetic viewpoints, train on the results, deploy to the Pi.

---

## Results

- **7,200 synthetic training images** generated from a single Shahed 136 STEP file
- **100% test accuracy** on drone vs no_drone binary classification
- **0% false negative rate** — no drone frames missed
- **0% false positive rate** — no false alarms on sky-only frames
- **8.5 MB ONNX model** — fits comfortably in Pi 4 memory
- **Estimated 15–20 FPS** on Raspberry Pi 4 (4-thread CPU inference)

---

## How it works

**1. Input a 3D model**
Drop in a `.obj`, `.glb`, `.stl`, `.fbx`, or `.step` file of the target drone.

**2. Synthetic view generation**
Open3D renders the model from 100 Fibonacci-sphere viewpoints × 6 distances = 600 renders per model. Fibonacci sampling gives uniform sphere coverage with no clustering at the poles. Each render uses a white background and mesh-scaling to simulate correct apparent size at each distance.

**3. Sky compositing**
Each render is composited over one of 100 synthetic sky backgrounds (5 procedural types × 20 seeds). Three sky variants per render = 1,800 composited drone images.

**4. Thermal simulation**
The same 1,800 images are passed through a thermal filter (iron / rainbow / grayscale LUTs, CLAHE contrast enhancement, heat bloom, sensor noise, scanlines) to produce 1,800 thermal equivalents.

**5. Training**
7,200 total images across four folders — drone RGB, drone thermal, no_drone RGB, no_drone thermal — train a MobileNetV2 classifier with binary labels: drone (1) and no_drone (0). MobileNetV2 was chosen for its low parameter count and CPU efficiency on Raspberry Pi.

**6. Deployment**
The model is exported to ONNX and deployed in `pi_deploy/` alongside a PID tracker and GPS navigation stubs. The Pi reads from the camera, runs inference on every frame, and sends pitch/yaw corrections to the flight controller when a drone is detected.

---

## Why 3D instead of 2D

If you only have 20 photos of a drone, your model only understands 20 perspectives. Rotate the target 45 degrees and recognition falls apart.

With a 3D model, the rendering step covers the full viewpoint sphere. The model ends up with a complete spatial understanding of the target — which matters when the threat can approach from any direction, altitude, or distance.

---

## How to run

**1. Place your 3D model in `models/`**

**2. Render synthetic views**
```bash
python -m renderer.render_views --config configs/default.yaml --obj models/yourmodel.step
```

**3. Composite over sky backgrounds**
```bash
python -m renderer.sky_compositor \
  --input dataset/raw/yourmodel/ \
  --output dataset/processed/yourmodel/ \
  --backgrounds dataset/backgrounds/
```

**4. Generate thermal variants**
```bash
python -m renderer.thermal_filter \
  --input dataset/processed/yourmodel/ \
  --output dataset/processed/yourmodel_thermal/ \
  --palette iron
```

**5. Train**
```bash
python -m training.train --config configs/default.yaml
```

**6. Run inference on an image or folder**
```bash
python inference/predict.py --model checkpoints/best.pt --input yourimage.jpg
python inference/predict.py --model checkpoints/best.pt --input tests/test_images/ --output tests/test_results/
```

**7. Run inference on a video**
```bash
python inference/predict.py --model checkpoints/best.pt --video yourvideo.mp4 --output result.mp4
```

---

## Deployment

Copy `pi_deploy/` to the Raspberry Pi and run the mission loop:

```bash
scp -r pi_deploy/ pi@<PI_IP>:~/omnivision3d/
ssh pi@<PI_IP>
cd ~/omnivision3d && pip install -r requirements_pi.txt
python main.py --lat 24.7136 --lon 46.6753
```

Test with a video file before flying:
```bash
python main.py --lat 24.7136 --lon 46.6753 --video tests/test_video.mp4
```

The system boots in simulation mode (`config.yaml: simulation_mode: true`). Navigation and flight controller calls are printed but not sent to hardware until wiring is complete.

---

## Stack

- **Python 3.10**
- **Open3D + Trimesh** — 3D mesh loading and offscreen rendering (Windows GPU pipeline)
- **PyTorch + torchvision** — MobileNetV2 training
- **ONNX + onnxruntime** — Pi deployment, no PyTorch needed on the device
- **OpenCV** — image processing, video I/O, overlay rendering
- **NumPy / Pillow** — image manipulation
- **FastAPI** — inference API (optional, for network-connected deployments)
- **PyYAML** — experiment and mission configuration

---

## Project structure

```
OmniVision3D/
├── models/              3D input files (.obj, .glb, .stl, .fbx, .step)
├── renderer/            Rendering pipeline (views, sky compositor, thermal)
├── training/            Dataset loader, MobileNetV2 model, training loop
├── inference/           Inference engine, visualization, annotated video output
├── pi_deploy/           Self-contained Raspberry Pi deployment package
│   ├── vision/          ONNX inference + camera loop
│   ├── control/         PID pitch/yaw tracker
│   ├── navigation/      GPS navigation stubs
│   ├── main.py          Mission loop
│   └── config.yaml      Mission settings
├── checkpoints/         Saved model weights and training artifacts
├── configs/             YAML experiment configs
├── tests/               19 unit and integration tests
└── README.md
```

---

## Future upgrades

- **MAVLink flight controller wiring** — replace PID output stubs with RC-override commands
- **Serial GPS integration** — replace position stubs with u-blox or dronekit reads
- **Bounding box output** — wire the detection head (`cx, cy, w, h, objectness`) into the tracker for pixel-precise targeting
- **Multi-drone detection** — extend to simultaneous tracking of multiple targets
- **Offline maps** — pre-downloaded GeoTIFFs for autonomous search pattern generation
- **Night / thermal camera input** — the model was trained on thermal data; a second capture index can run a parallel inference pass

---

## Tests

```bash
python -m pytest tests/ -v
```

19 tests covering the renderer (camera utils, sky compositor, thermal filter) and inference engine (predictor, visualizer). All passing.
