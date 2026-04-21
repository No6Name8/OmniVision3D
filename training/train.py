"""
Training script for OmniVision3D — MobileNetV2 on synthetic drone renders.

Usage:
    python -m training.train --config configs/default.yaml
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .dataset_loader import load_splits
from .model import OmniVisionModel


def evaluate_test_set(model, test_loader, device, class_names, ckpt_dir):
    """
    Run inference on the full test set and print a confusion matrix plus
    false-positive / false-negative rates.

    Binary task:
        label 0 = no_drone
        label 1 = drone

    False positive  : no_drone frame predicted as drone
    False negative  : drone frame predicted as no_drone   ← must be minimised
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    n_classes = len(class_names)
    # cm[true][pred]
    cm = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # For binary: class 1 = drone (positive), class 0 = no_drone (negative)
    tp = cm[1][1]   # drone correctly detected
    fn = cm[1][0]   # drone missed           ← false negative
    fp = cm[0][1]   # no_drone called drone  ← false positive
    tn = cm[0][0]   # no_drone correctly rejected

    total        = tp + fn + fp + tn
    accuracy     = (tp + tn) / total if total else 0.0
    fp_rate      = fp / (fp + tn) if (fp + tn) else 0.0   # FPR = FP / N
    fn_rate      = fn / (fn + tp) if (fn + tp) else 0.0   # FNR = FN / P
    precision    = tp / (tp + fp) if (tp + fp) else 0.0
    recall       = tp / (tp + fn) if (tp + fn) else 0.0   # = 1 - FNR

    print(f"\n{'='*54}")
    print("TEST-SET EVALUATION")
    print(f"{'='*54}")
    print(f"\n  Confusion matrix  (rows=true, cols=predicted)\n")
    header = f"  {'':>14}" + "".join(f"  {c:>12}" for c in class_names)
    print(header)
    print("  " + "-" * (14 + 14 * n_classes))
    for i, row_name in enumerate(class_names):
        row = f"  {row_name:>14}" + "".join(f"  {cm[i][j]:>12}" for j in range(n_classes))
        print(row)

    print(f"\n  Total test images : {total}")
    print(f"  Accuracy          : {accuracy:.4f}  ({(tp+tn)}/{total})")
    print(f"\n  Drone detection (positive class = drone)")
    print(f"    True  positives : {tp:>5}  (drone correctly detected)")
    print(f"    False negatives : {fn:>5}  (drone MISSED)          FNR = {fn_rate:.4f}")
    print(f"    True  negatives : {tn:>5}  (no_drone correctly rejected)")
    print(f"    False positives : {fp:>5}  (sky called drone)      FPR = {fp_rate:.4f}")
    print(f"\n  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}  (= 1 - FNR)")

    if fn_rate > 0.02:
        print(f"\n  WARNING: FNR {fn_rate:.4f} exceeds 2% — drones are being missed!")
    if fp_rate > 0.05:
        print(f"  WARNING: FPR {fp_rate:.4f} exceeds 5% — too many false alarms!")
    if fn_rate <= 0.02 and fp_rate <= 0.05:
        print(f"\n  OK: FNR <= 2%  and  FPR <= 5% — deployment thresholds met.")

    print(f"{'='*54}\n")

    # Save confusion matrix as CSV
    cm_path = ckpt_dir / "confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + class_names)
        for i, row_name in enumerate(class_names):
            w.writerow([row_name] + cm[i])
    print(f"  Confusion matrix saved -> {cm_path}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow([[cm[i][j] for j in range(n_classes)] for i in range(n_classes)],
                   interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > (max(max(r) for r in cm) / 2) else "black")
    plt.tight_layout()
    cm_img = ckpt_dir / "confusion_matrix.png"
    plt.savefig(cm_img, dpi=120)
    plt.close()
    print(f"  Confusion matrix plot  -> {cm_img}")


def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed_root = str(Path(cfg["rendering"]["output_path"]) / "processed")
    train_ds, val_ds, test_ds, classes = load_splits(processed_root)
    num_classes = len(classes)

    # num_workers=0 avoids Windows multiprocessing issues
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,  batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )

    model     = OmniVisionModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.csv"

    max_epochs          = 50
    early_stop_patience = 5
    best_val_acc        = 0.0
    best_epoch          = 0
    no_improve          = 0

    train_losses: list = []
    val_losses:   list = []
    val_accs:     list = []

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])

    print(f"Classes: {classes}")
    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)}\n")

    start = time.time()

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # ---- validate ----
        model.eval()
        val_running = 0.0
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_running += criterion(out, labels).item()
                correct += (out.argmax(dim=1) == labels).sum().item()
                total   += labels.size(0)
        val_loss = val_running / len(val_loader)
        val_acc  = correct / total
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:>2}/50  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={current_lr:.6f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                 f"{val_acc:.4f}", f"{current_lr:.6f}"]
            )

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"           -> new best saved (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"\nEarly stopping — no improvement for {early_stop_patience} epochs.")
                break

    elapsed = time.time() - start

    # ---- training curve ----
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, train_losses, label="Train loss")
    ax1.plot(epochs_range, val_losses,   label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs_range, val_accs, color="green", label="Val accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    curve_path = ckpt_dir / "training_curve.png"
    plt.savefig(curve_path, dpi=120)
    plt.close()

    # ---- load best checkpoint for test evaluation ----
    best_ckpt = ckpt_dir / "best.pt"
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state)

    # ---- final report ----
    print(f"\n{'='*50}")
    print(f"Training complete")
    print(f"  Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Final train loss  : {train_losses[-1]:.4f}")
    print(f"  Final val loss    : {val_losses[-1]:.4f}")
    print(f"  Total time        : {elapsed / 60:.1f} min")
    print(f"  Checkpoint        : {best_ckpt}")
    print(f"  Training curve    : {curve_path}")
    print(f"{'='*50}")

    evaluate_test_set(model, test_loader, device, classes, ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
