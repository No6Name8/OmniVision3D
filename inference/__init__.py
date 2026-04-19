"""Inference package — object recognition on images and video."""

from .predict import Predictor
from .visualize import draw_prediction, annotate_video

__all__ = ["Predictor", "draw_prediction", "annotate_video"]
