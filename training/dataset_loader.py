"""
Dataset loader for OmniVision3D — binary drone-detection classification.

Two physical folder groups are merged into two logical classes:
    label 1  (drone)    : "Shahed 136 Drone"          + "Shahed 136 Drone_thermal"
    label 0  (no_drone) : "no_drone"                  + "no_drone_thermal"

This teaches the model the single question the Raspberry Pi needs answered:
"Is there a drone in this frame?"

Augmentations are applied to training images only.
Val/test images get resize + normalize only.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Maps every physical folder name to a binary label.
# Add new drone variants here in the future without changing anything else.
FOLDER_TO_LABEL: Dict[str, int] = {
    "Shahed 136 Drone":          1,   # drone — RGB
    "Shahed 136 Drone_thermal":  1,   # drone — thermal
    "no_drone":                  0,   # background — RGB
    "no_drone_thermal":          0,   # background — thermal
}

CLASS_NAMES = ["no_drone", "drone"]


class SyntheticViewDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_splits(
    root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[SyntheticViewDataset, SyntheticViewDataset, SyntheticViewDataset, List[str]]:
    """
    Load all four folders, assign binary labels, and return stratified splits.

    Splitting is done per physical folder so each source is proportionally
    represented in every split.

    Returns:
        train_ds, val_ds, test_ds, class_names (["no_drone", "drone"])
    """
    root_path = Path(root)
    rng = random.Random(seed)

    train_samples: List[Tuple[Path, int]] = []
    val_samples:   List[Tuple[Path, int]] = []
    test_samples:  List[Tuple[Path, int]] = []

    folder_counts: Dict[str, Dict[str, int]] = {}

    for folder_name, label in FOLDER_TO_LABEL.items():
        folder_path = root_path / folder_name
        if not folder_path.exists():
            print(f"  WARNING: folder not found — {folder_path}")
            continue

        samples = [(p, label) for p in sorted(folder_path.glob("*.png"))]
        rng.shuffle(samples)

        n       = len(samples)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        n_test  = n - n_train - n_val

        train_samples.extend(samples[:n_train])
        val_samples.extend(  samples[n_train : n_train + n_val])
        test_samples.extend( samples[n_train + n_val :])

        folder_counts[folder_name] = {"total": n, "train": n_train,
                                      "val": n_val, "test": n_test,
                                      "label": label}

    # Print summary
    print(f"\nDataset root : {root_path}")
    print(f"{'Folder':<35} {'Label':<10} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}")
    print("-" * 72)
    for name, c in folder_counts.items():
        lbl = CLASS_NAMES[c['label']]
        print(f"  {name:<33} {lbl:<10} {c['total']:>6}  {c['train']:>6}  {c['val']:>5}  {c['test']:>5}")

    drone_train    = sum(1 for _, l in train_samples if l == 1)
    no_drone_train = sum(1 for _, l in train_samples if l == 0)
    print(f"\n  Train  : {len(train_samples):>5}  (drone={drone_train}, no_drone={no_drone_train})")
    print(f"  Val    : {len(val_samples):>5}")
    print(f"  Test   : {len(test_samples):>5}\n")

    train_ds = SyntheticViewDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds   = SyntheticViewDataset(val_samples,   transform=EVAL_TRANSFORM)
    test_ds  = SyntheticViewDataset(test_samples,  transform=EVAL_TRANSFORM)

    return train_ds, val_ds, test_ds, CLASS_NAMES
