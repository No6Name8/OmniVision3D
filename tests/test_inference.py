"""
Integration tests for the inference module.

Verifies that Predictor loads a model from a checkpoint, performs a forward pass
on a synthetic random image, and returns a valid (label, confidence) pair.
Also tests that visualize.draw_prediction returns a same-shape annotated copy.
"""

import numpy as np
import pytest
import torch

from inference.predict import Predictor
from inference.visualize import draw_prediction
from training.model import OmniVisionModel

DUMMY_CLASSES = ["cube", "sphere", "cylinder"]


@pytest.fixture
def predictor(tmp_path):
    """Predictor backed by a randomly-initialised (untrained) checkpoint."""
    model = OmniVisionModel(num_classes=len(DUMMY_CLASSES), pretrained=False)
    ckpt = tmp_path / "test.pt"
    torch.save(model.state_dict(), ckpt)
    return Predictor(checkpoint_path=str(ckpt), classes=DUMMY_CLASSES, device="cpu")


def test_predict_image_returns_known_class(predictor):
    """The returned label must be one of the registered class names."""
    fake_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    label, confidence = predictor.predict_image(fake_bgr)
    assert label in DUMMY_CLASSES


def test_predict_image_confidence_in_range(predictor):
    """Confidence must be a probability in [0, 1]."""
    fake_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, confidence = predictor.predict_image(fake_bgr)
    assert 0.0 <= confidence <= 1.0


def test_draw_prediction_preserves_shape():
    """draw_prediction must return an image with the same HxWxC shape as input."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_prediction(image, label="cube", confidence=0.95)
    assert annotated.shape == image.shape


def test_draw_prediction_returns_copy():
    """draw_prediction must not modify the original image in-place."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_prediction(image, label="sphere", confidence=0.72)
    assert annotated is not image
    assert np.array_equal(image, np.zeros((480, 640, 3), dtype=np.uint8))
