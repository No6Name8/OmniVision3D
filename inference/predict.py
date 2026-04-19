"""
Inference engine for OmniVision3D object recognition.

Supports single-image prediction and frame-by-frame video processing,
returning the predicted class label and softmax confidence score for each input.

Usage:
    from inference.predict import Predictor
    predictor = Predictor("checkpoints/best.pt", classes=["cube", "sphere"], device="cuda")
    label, conf = predictor.predict_image(bgr_image)
"""

from typing import Generator, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from training.model import load_model

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Predictor:
    """Wraps an OmniVisionModel checkpoint for single-image or video inference."""

    def __init__(
        self,
        checkpoint_path: str,
        classes: List[str],
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.classes = classes
        self.model = load_model(checkpoint_path, len(classes), self.device)

    def predict_image(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict the class of a single BGR image (as returned by cv2.imread).

        Returns:
            (class_name, confidence) where confidence is in [0, 1].
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = INFERENCE_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]

        top_idx = int(probs.argmax())
        return self.classes[top_idx], float(probs[top_idx])

    def predict_video(
        self, video_path: str
    ) -> Generator[Tuple[int, str, float], None, None]:
        """
        Yield (frame_index, class_name, confidence) for every frame in a video file.
        """
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            label, conf = self.predict_image(frame)
            yield frame_idx, label, conf
            frame_idx += 1
        cap.release()
