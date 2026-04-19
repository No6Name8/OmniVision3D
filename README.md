# OmniVision3D

Most object recognition systems need hundreds of labeled photos to learn what something looks like. OmniVision3D takes a different approach — give it one 3D model, and it figures out the rest.

The system renders that model from over a thousand angles, distances, and lighting conditions to build its own training dataset. No photography. No manual labeling. Just a mesh file and some compute time.

The trained model can then spot that object in real-world images or video, regardless of where the camera is, how far away it is, or what the lighting looks like.

---

## How it works

**1. Input a 3D model**
Drop in a `.obj`, `.glb`, or `.stl` file of the object you want to recognize.

**2. Synthetic view generation**
PyTorch3D renders the model from a full sphere of viewpoints — 24 horizontal rotations, 5 elevation levels, 3 distances, and 3–4 lighting variations. That comes out to roughly 1,080 images per object, all automatically generated and labeled.

**3. Training**
Those synthetic images feed into a PyTorch-based recognition model. It learns what the object looks like from every direction.

**4. Inference**
Point the model at a real image or video feed. It identifies the object, draws a bounding box, and returns a confidence score — even from angles it's never seen in real life.

---

## Why 3D instead of 2D

If you only have 20 photos of an object, your model only understands 20 perspectives. Rotate the object 45 degrees and recognition falls apart.

With a 3D model, you're not limited by what you happened to photograph. The rendering step covers the entire viewpoint sphere systematically. The model ends up with a complete spatial understanding of the object rather than a collection of snapshots.

---

## Stack

- **Python** — core language
- **PyTorch3D** — 3D mesh rendering and synthetic view generation
- **PyTorch + torchvision** — model training and inference
- **OpenCV** — real-world image and video processing
- **FastAPI** — inference API
- **YAML** — experiment and config management

---

## Project structure

```
OmniVision3D/
├── models/          # 3D input files (.obj, .glb, .stl)
├── renderer/        # Synthetic view generation
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

Early development. Folder structure and rendering pipeline are being set up.

---

## Goal

Show that one 3D model is enough to build a recognition system that works across any viewpoint, any distance, any environment — without collecting a single real photo.