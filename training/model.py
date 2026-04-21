"""
Recognition model built on a pretrained MobileNetV2 backbone.

MobileNetV2 is chosen for its low parameter count and efficient depthwise
separable convolutions — both critical for CPU inference on Raspberry Pi.

The classifier head is replaced to match the number of object classes.
A separate detection head outputs bounding-box deltas and an objectness score.
embed() exposes the 1280-dim penultimate features for retrieval or downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class OmniVisionModel(nn.Module):
    """MobileNetV2 backbone with classification and lightweight detection heads."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        self.features   = backbone.features   # outputs (B, 1280, H, W)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )
        # Lightweight detection head: (cx, cy, w, h, objectness)
        self.det_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._pool(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1280-dim feature embeddings (no classification head)."""
        return self._pool(x)

    def detect(self, x: torch.Tensor) -> torch.Tensor:
        """Return (cx, cy, w, h, objectness) predictions of shape (B, 5)."""
        return self.det_head(self._pool(x))


def load_model(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
) -> OmniVisionModel:
    """Instantiate an OmniVisionModel and load weights from a checkpoint file."""
    model = OmniVisionModel(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
