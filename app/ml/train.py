"""
GPU-Accelerated PyTorch Training Recipe for Electronic Component Defect Screening.
Implements mixed precision (AMP), CosineAnnealingLR, early stopping, and metric tracking.
"""
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from app.config import config
from app.dataset.dataset import load_component_data
from app.dataset.generator import ComponentGenerator
from app.ml.model import ComponentDefectNet


def set_seed(seed: int = 42) -> None:
    """Ensures full reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(
    y_true: list[int], y_pred: list[int]
) -> dict[str, float]:
    """Computes Accuracy, Precision, Recall, and F1-Score for binary classification."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Executes one training epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == "cuda"

    for batch_idx, (images, targets, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluates the model on validation data."""
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    total = 0

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        total += targets.size(0)

    avg_loss = total_loss / total
    metrics = calculate_metrics(y_true, y_pred)
    return avg_loss, metrics


def run_training(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: Path | None = None,
    device_name: str | None = None,
    progress_callback=None,
    auto_generate_if_empty: bool = True,
) -> dict:
    """Main training execution function with optional live progress callback."""
    set_seed(42)

    if save_path is None:
        save_path = config.best_model_path

    # Determine Device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("A.R.G.U.S. Component Defect Screening - PyTorch Vision Trainer")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    if device.type == "cuda":
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
        print(f"VRAM Reserved:  {torch.cuda.memory_reserved(0)/(1024**2):.1f} MB")
    print("=" * 60)

    # Check if dataset exists, generate synthetic samples if needed
    normal_samples = list(config.normal_dir.glob("*.png")) + list(config.normal_dir.glob("*.jpg"))
    defective_samples = list(config.defective_dir.glob("*.png")) + list(config.defective_dir.glob("*.jpg"))

    if (len(normal_samples) < 5 or len(defective_samples) < 5) and auto_generate_if_empty:
        print("[!] Dataset is empty or small. Generating synthetic IC training dataset...")
        gen = ComponentGenerator()
        gen.generate_dataset(num_normal=150, num_defective=150)

    # Load Data
    train_loader, val_loader, stats = load_component_data(
        batch_size=batch_size, img_size=config.img_size
    )
    print(f"Dataset stats: {stats}")

    # Build Model
    model = ComponentDefectNet(
        num_classes=2,
        backbone_name=config.model_name,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1 = 0.0
    best_val_acc = 0.0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_score"]
        epoch_time = time.time() - epoch_start

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": current_lr,
        }
        history.append(record)

        if progress_callback is not None:
            progress_callback(epoch, epochs, record)

        # Save Best Model Checkpoint
        if val_f1 >= best_val_f1 and val_acc >= best_val_acc:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                str(save_path),
            )
            print(f"  --> Saved new best model checkpoint to {save_path.name} (Val F1: {val_f1:.4f})")

    total_training_time = time.time() - start_time
    print("=" * 60)
    print(f"Training Complete in {total_training_time:.2f}s!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% | Best F1-Score: {best_val_f1:.4f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)

    return {
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "history": history,
        "total_time": total_training_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISRO Component Defect Screening Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
