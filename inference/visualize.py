"""
Visualization utilities for OmniVision3D inference results.

Draws predicted class labels and confidence scores onto images or video frames
using OpenCV, and writes annotated video output to disk when given a prediction list.
"""

from typing import List, Tuple

import cv2
import numpy as np


def draw_prediction(
    image: np.ndarray,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 200, 0),
    font_scale: float = 0.9,
    thickness: int = 2,
) -> np.ndarray:
    """
    Overlay a label and confidence score onto a copy of a BGR image.

    A filled rectangle is drawn behind the text so it stays readable over
    any background colour.

    Args:
        image:      BGR image array (H, W, 3).
        label:      Predicted class name.
        confidence: Confidence score in [0, 1].
        color:      BGR colour used for the background rectangle.
        font_scale: OpenCV font scale factor.
        thickness:  Text stroke thickness in pixels.

    Returns:
        Annotated copy of the input image.
    """
    out = image.copy()
    text = f"{label}: {confidence:.2%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad + baseline), color, -1)
    cv2.putText(
        out, text, (pad, th + pad),
        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )
    return out


def annotate_video(
    video_path: str,
    output_path: str,
    predictions: List[Tuple[int, str, float]],
) -> None:
    """
    Write an annotated copy of a video with prediction overlays on each frame.

    Args:
        video_path:  Path to the source video file.
        output_path: Path where the annotated video will be saved.
        predictions: List of (frame_index, label, confidence) from Predictor.predict_video.
    """
    pred_map = {fi: (lbl, conf) for fi, lbl, conf in predictions}
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in pred_map:
            label, conf = pred_map[frame_idx]
            frame = draw_prediction(frame, label, conf)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
