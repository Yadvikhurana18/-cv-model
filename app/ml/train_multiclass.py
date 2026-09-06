"""
GPU Multi-Class Training Recipe for Electronic Component Defect Diagnosis.
Trains a 6-class model: [Normal, Bent_Pin, Missing_Pin, Surface_Crack, Solder_Bridge, Orientation_Fault]
using NVIDIA RTX 3050 GPU with Mixed Precision (AMP) and Cosine Annealing scheduler.
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.config import config
from app.dataset.dataset import load_multiclass_data
from app.dataset.generator import ComponentGenerator
from app.ml.model import ComponentDefectNet


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch_multiclass(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == "cuda"

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None and use_cuda)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None and use_cuda:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_multiclass(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 6,
) -> tuple[float, float, list[list[int]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    conf_matrix = [[0] * num_classes for _ in range(num_classes)]

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        for t, p in zip(targets.cpu().tolist(), preds.cpu().tolist()):
            conf_matrix[t][p] += 1

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, conf_matrix


def run_multiclass_training(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    samples_per_class: int = 120,
    progress_callback=None,
    auto_generate_if_empty: bool = True,
) -> dict:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("A.R.G.U.S. Component Screening - Multi-Class Defect Diagnostic Trainer")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 65)

    multiclass_dir = config.dataset_dir / "multiclass"
    existing_samples = list(multiclass_dir.glob("*/*.png")) + list(multiclass_dir.glob("*/*.jpg"))

    if (len(existing_samples) < 10) and auto_generate_if_empty:
        print("[!] Multi-class dataset is empty. Generating synthetic dataset...")
        gen = ComponentGenerator()
        gen.generate_multiclass_dataset(samples_per_class=samples_per_class)

    train_loader, val_loader, stats, class_names = load_multiclass_data(
        batch_size=batch_size, img_size=config.img_size
    )
    print(f"Dataset stats: {stats['class_counts']} (Train: {stats['train_samples']}, Val: {stats['val_samples']})")

    model = ComponentDefectNet(
        num_classes=len(class_names),
        backbone_name="mobilenet_v3_large",
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    save_path = config.models_dir / "component_multiclass_net.pth"
    best_acc = 0.0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        train_loss, train_acc = train_epoch_multiclass(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, conf_mat = evaluate_multiclass(
            model, val_loader, criterion, device, num_classes=len(class_names)
        )
        scheduler.step()
        ep_time = time.time() - t_start

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        history.append(record)

        if progress_callback is not None:
            progress_callback(epoch, epochs, record)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_acc": best_acc,
                "conf_matrix": conf_mat,
            }, str(save_path))
            print(f"  --> Saved new best multi-class model checkpoint! (Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"Multi-Class Training Complete in {total_time:.1f}s!")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {save_path}")
    print("=" * 65)

    return {
        "best_acc": best_acc,
        "history": history,
        "class_names": class_names,
        "total_time": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_multiclass_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
