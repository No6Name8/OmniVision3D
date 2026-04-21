"""
Inference engine for OmniVision3D drone detection.

Supports single-image prediction and frame-by-frame video processing.
Each prediction returns the class label and softmax confidence score.

CLI usage:
    # Image folder (mirrors subfolder structure into output)
    python inference/predict.py --model checkpoints/best.pt \
        --input tests/test_images/ --output tests/test_results/

    # Video file
    python inference/predict.py --model checkpoints/best.pt \
        --video tests/test_video.mp4 --output tests/test_video_result.mp4

Module usage:
    from inference.predict import Predictor
    predictor = Predictor("checkpoints/best.pt", classes=["no_drone", "drone"])
    label, conf = predictor.predict_image(bgr_image)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Generator, List, Tuple

# Ensure the project root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from training.model import load_model
from inference.visualize import draw_prediction, annotate_video

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class Predictor:
    """Wraps an OmniVisionModel checkpoint for image or video inference."""

    def __init__(
        self,
        checkpoint_path: str,
        classes: List[str],
        device: str = "cpu",
    ) -> None:
        self.device  = torch.device(device)
        self.classes = classes
        self.model   = load_model(checkpoint_path, len(classes), self.device)

    def predict_image(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict the class of a single BGR image (as returned by cv2.imread).

        Returns:
            (class_name, confidence) where confidence is in [0, 1].
        """
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = INFERENCE_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]

        top_idx = int(probs.argmax())
        return self.classes[top_idx], float(probs[top_idx])

    def predict_video(
        self, video_path: str
    ) -> Generator[Tuple[int, str, float], None, None]:
        """Yield (frame_index, class_name, confidence) for every frame in a video."""
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


def _run_image_folder(
    predictor: Predictor,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Run inference on all images under input_dir, save annotated copies to output_dir."""
    image_files = [
        p for p in sorted(input_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS
    ]
    if not image_files:
        print(f"No images found under {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_time_ms = 0.0

    # Determine true label from parent folder name (drone / no_drone)
    def true_label(path: Path) -> str:
        for part in path.parts:
            if part == "drone":
                return "drone"
            if part == "no_drone":
                return "no_drone"
        return "unknown"

    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  WARNING: could not read {img_path}")
            continue

        t0 = time.perf_counter()
        label, conf = predictor.predict_image(bgr)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_time_ms += elapsed_ms

        annotated = draw_prediction(bgr, label, conf)

        # Mirror relative path into output dir
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)

        gt = true_label(img_path)
        correct = (label == gt) if gt != "unknown" else None
        results.append((img_path.name, gt, label, conf, correct, elapsed_ms))

    # ---- summary ----
    n_total   = len(results)
    n_drone   = sum(1 for r in results if r[1] == "drone")
    n_nodrone = sum(1 for r in results if r[1] == "no_drone")
    n_correct_drone   = sum(1 for r in results if r[1] == "drone"    and r[4])
    n_correct_nodrone = sum(1 for r in results if r[1] == "no_drone" and r[4])
    n_known   = sum(1 for r in results if r[4] is not None)
    n_correct = sum(1 for r in results if r[4])
    avg_conf  = sum(r[3] for r in results) / n_total if n_total else 0
    avg_ms    = total_time_ms / n_total if n_total else 0

    print(f"\n{'='*54}")
    print("INFERENCE SUMMARY — IMAGE FOLDER")
    print(f"{'='*54}")
    print(f"  Total images         : {n_total}")
    print(f"  Drone images         : {n_drone}  (correct: {n_correct_drone}/{n_drone})")
    print(f"  No-drone images      : {n_nodrone}  (correct: {n_correct_nodrone}/{n_nodrone})")
    if n_known:
        print(f"  Overall accuracy     : {n_correct}/{n_known}  ({n_correct/n_known:.1%})")
    print(f"  Avg confidence       : {avg_conf:.1%}")
    print(f"  Avg inference time   : {avg_ms:.1f} ms/image")
    print(f"  Output saved to      : {output_dir}")
    print(f"{'='*54}\n")

    print(f"  {'File':<35} {'True':>10} {'Pred':>10} {'Conf':>7} {'OK':>4}")
    print("  " + "-" * 70)
    for fname, gt, pred, conf, ok, ms in results:
        ok_str = "YES" if ok else ("NO" if ok is False else "?")
        print(f"  {fname:<35} {gt:>10} {pred:>10} {conf:>6.1%} {ok_str:>4}")


def _run_video(
    predictor: Predictor,
    video_path: Path,
    output_path: Path,
) -> None:
    """Run inference on a video file, write annotated output, report FPS."""
    # Collect all predictions first (needed for annotate_video)
    predictions = []
    t0 = time.perf_counter()
    for frame_idx, label, conf in predictor.predict_video(str(video_path)):
        predictions.append((frame_idx, label, conf))
    elapsed = time.perf_counter() - t0

    n_frames = len(predictions)
    fps      = n_frames / elapsed if elapsed > 0 else 0
    avg_ms   = (elapsed / n_frames * 1000) if n_frames else 0

    n_drone   = sum(1 for _, lbl, _ in predictions if lbl == "drone")
    n_nodrone = n_frames - n_drone

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotate_video(str(video_path), str(output_path), predictions)

    print(f"\n{'='*54}")
    print("INFERENCE SUMMARY — VIDEO")
    print(f"{'='*54}")
    print(f"  Total frames         : {n_frames}")
    print(f"  Drone frames         : {n_drone}")
    print(f"  No-drone frames      : {n_nodrone}")
    print(f"  Inference FPS        : {fps:.1f}")
    print(f"  Avg time per frame   : {avg_ms:.1f} ms")
    print(f"  Output saved to      : {output_path}")
    print(f"{'='*54}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniVision3D inference")
    parser.add_argument("--model",   required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--classes", nargs="+", default=["no_drone", "drone"])
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--input",   help="Input image folder")
    parser.add_argument("--output",  help="Output folder / video path")
    parser.add_argument("--video",   help="Input video file")
    args = parser.parse_args()

    predictor = Predictor(
        checkpoint_path=args.model,
        classes=args.classes,
        device=args.device,
    )

    if args.video:
        output = Path(args.output) if args.output else Path(args.video).with_stem(
            Path(args.video).stem + "_result"
        )
        _run_video(predictor, Path(args.video), output)
    elif args.input:
        output = Path(args.output) if args.output else Path(args.input) / "results"
        _run_image_folder(predictor, Path(args.input), output)
    else:
        parser.error("Provide --input (image folder) or --video")


if __name__ == "__main__":
    main()
