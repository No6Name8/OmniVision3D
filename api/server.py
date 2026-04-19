"""
FastAPI inference server for OmniVision3D.

Exposes a /predict endpoint that accepts an uploaded image and returns the
predicted object class and confidence score as JSON.

Environment variables:
    OMNIVISION_CONFIG      Path to the YAML config (default: configs/default.yaml)
    OMNIVISION_CHECKPOINT  Path to the model checkpoint (default: checkpoints/best.pt)
    USE_GPU                Set to any value to run inference on CUDA

Usage:
    uvicorn api.server:app --reload
"""

import os
from typing import List

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from inference.predict import Predictor

app = FastAPI(title="OmniVision3D", version="0.1.0")

_CONFIG_PATH = os.environ.get("OMNIVISION_CONFIG", "configs/default.yaml")
_CHECKPOINT_PATH = os.environ.get("OMNIVISION_CHECKPOINT", "checkpoints/best.pt")

with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

_classes: List[str] = _cfg.get("classes", [])
_device = "cuda" if os.environ.get("USE_GPU") else "cpu"
_predictor = Predictor(
    checkpoint_path=_CHECKPOINT_PATH,
    classes=_classes,
    device=_device,
)


class PredictionResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health() -> dict:
    """Liveness check — returns 200 when the server is ready."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accept an image upload and return the predicted object class.

    The image may be any format supported by OpenCV (JPEG, PNG, BMP, etc.).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image data.")

    label, confidence = _predictor.predict_image(image)
    return PredictionResponse(label=label, confidence=confidence)
