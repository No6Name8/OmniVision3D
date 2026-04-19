"""
PyTorch Dataset for loading synthetic rendered images from the processed dataset directory.

Expects images organised in class sub-directories under the processed/ folder,
following the ImageFolder convention:  processed/<class_name>/<image>.png
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SyntheticViewDataset(Dataset):
    """Dataset that loads rendered synthetic images from a class-organised directory tree."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = DEFAULT_TRANSFORM,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        self.classes: List[str] = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            for img_path in sorted((self.root / cls).glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
