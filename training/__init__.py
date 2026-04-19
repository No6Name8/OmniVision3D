"""Training package — dataset loading, model architecture, and training loop."""

from .dataset_loader import SyntheticViewDataset
from .model import OmniVisionModel, load_model

__all__ = ["SyntheticViewDataset", "OmniVisionModel", "load_model"]
