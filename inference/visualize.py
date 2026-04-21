"""
Visualization utilities for OmniVision3D inference results.

Draws prediction overlays onto images:
  - DRONE DETECTED : green bounding box + label
  - NO DRONE       : red border + label
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def draw_prediction(
    image: np.ndarray,
    label: str,
    confidence: float,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Overlay prediction label, confidence, and optional bounding box onto a BGR image.

    For drone detections: draws a green bounding box (or full-frame box if no bbox).
    For no-drone:         draws a 6-pixel red border around the frame.

    Args:
        image:      BGR image array (H, W, 3).
        label:      Predicted class name ("drone" or "no_drone").
        confidence: Confidence score in [0, 1].
        bbox:       Optional (x1, y1, x2, y2) in pixel coords for drone box.

    Returns:
        Annotated copy of the input image.
    """
    out = image.copy()
    h, w = out.shape[:2]
    is_drone = label == "drone"

    if is_drone:
        box_color = (0, 220, 0)      # green
        text_bg   = (0, 180, 0)
        display   = "DRONE DETECTED"
        # Draw bounding box
        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            # No bbox: full-frame box inset by 10 px
            margin = 10
            x1, y1, x2, y2 = margin, margin, w - margin, h - margin
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 3)
    else:
        box_color = (0, 0, 220)      # red
        text_bg   = (0, 0, 180)
        display   = "NO DRONE"
        # Red border
        border = 6
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), box_color, border)

    # Label banner
    text = f"{display}  {confidence:.1%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(w, h) / 600)
    thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad + baseline), text_bg, -1)
    cv2.putText(out, text, (pad, th + pad), font, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
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
        predictions: List of (frame_index, label, confidence).
    """
    pred_map = {fi: (lbl, conf) for fi, lbl, conf in predictions}
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
