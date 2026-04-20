# OmniVision3D

OmniVision3D is the vision module of a drone-based detection and response system built for competition. It runs onboard a fixed-wing response drone on a Raspberry Pi and detects enemy drones in real-time from aerial footage — both regular and thermal camera input.

The core problem: most object recognition systems need hundreds of labeled photos. OmniVision3D takes a different approach — give it one 3D model of the target, and it builds its own training dataset from scratch.

The system renders that model from 1,440 synthetic viewpoints (24 azimuths × 5 elevations × 3 distances × 4 lighting conditions), trains a lightweight recognition model on the results, then deploys that model to the Pi for live detection.

---

## How it works

**1. Input a 3D model**
Drop in a `.obj`, `.glb`, `.stl`, or `.fbx` file of the target drone.

**2. Synthetic view generation**
PyTorch3D renders the model from a full sphere of viewpoints — 24 horizontal rotations, 5 elevation levels, 3 distances, and 4 lighting variations. That comes out to 1,440 images per object, all automatically generated and labeled.

**3. Training**
The synthetic images train a MobileNetV2-based recognition model. MobileNetV2 was chosen for its low parameter count and CPU efficiency — both necessary for real-time inference on a Raspberry Pi.

**4. Inference**
The model runs onboard the response drone. It processes frames from the onboard camera (regular or thermal), identifies target drones, draws bounding boxes, and returns confidence scores. The model is exported to ONNX for optimized Pi deployment.

---

## Why 3D instead of 2D

If you only have 20 photos of an object, your model only understands 20 perspectives. Rotate the object 45 degrees and recognition falls apart.

With a 3D model, the rendering step covers the entire viewpoint sphere systematically. The model ends up with a complete spatial understanding of the target rather than a collection of snapshots — which matters when the target can approach from any direction at any altitude.

---

## Stack

- **Python** — core language
- **PyTorch3D** — 3D mesh rendering and synthetic view generation
- **PyTorch + torchvision** — MobileNetV2 training and inference
- **Trimesh** — mesh loading for non-.obj formats (.fbx, .glb, .stl)
- **OpenCV** — real-world image and video processing
- **FastAPI** — inference API
- **ONNX** — model export for Raspberry Pi deployment
- **YAML** — experiment and config management

---

## Project structure

```
OmniVision3D/
├── models/          # 3D input files (.obj, .glb, .stl, .fbx)
├── renderer/        # Synthetic view generation (PyTorch3D)
├── dataset/         # Generated training images
├── training/        # Model training pipeline
├── inference/       # Recognition engine
├── api/             # FastAPI inference server
├── configs/         # YAML experiment configs
├── tests/           # Unit and integration tests
└── README.md
```

---

## Status

Active development. Rendering pipeline and model architecture are complete. Integration with the drone platform and thermal camera input are in progress.

---

## Goal

One 3D model of an enemy drone is enough to build a detection system that works from any viewpoint, any distance, any lighting — deployable on a Pi, running live on a fixed-wing response drone.
