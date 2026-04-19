"""
Main training script for the OmniVision3D recognition model.

Loads the synthetic dataset from the processed/ directory, instantiates the
ResNet-based model, and runs a cross-entropy training loop with a held-out
validation split. The best checkpoint (by validation accuracy) is saved to disk.

Usage:
    python -m training.train --config configs/default.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from .dataset_loader import SyntheticViewDataset
from .model import OmniVisionModel


def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_root = str(Path(cfg["rendering"]["output_path"]) / "processed")
    dataset = SyntheticViewDataset(root=processed_root)
    num_classes = len(dataset.classes)
    print(f"Classes ({num_classes}): {dataset.classes}")

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )

    model = OmniVisionModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch:>3}/{tcfg['epochs']}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
