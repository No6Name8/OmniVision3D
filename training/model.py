"""
Recognition model built on a pretrained ResNet-50 backbone.

The final fully-connected layer is replaced to match the number of object classes.
An embed() method exposes penultimate-layer features for similarity search or
nearest-neighbour retrieval without the classification head.
"""

import torch
import torch.nn as nn
from torchvision import models


class OmniVisionModel(nn.Module):
    """ResNet-50 backbone with a custom classification head for object recognition."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-dim feature embeddings from the layer before the FC head."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)


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
