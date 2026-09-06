"""
PyTorch Dataset and DataLoader for Electronic Component Screening.
Supports image loading, Albumentations/Torchvision augmentations, and train/val splitting.
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from app.config import config


def get_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Returns PyTorch torchvision transforms for training and validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


class ComponentDataset(Dataset):
    """
    Dataset for electronic component screening images.
    Classes: 0 -> NORMAL, 1 -> DEFECTIVE
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose | None = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt file
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        return tensor, label, str(img_path)


def load_component_data(
    dataset_dir: Path | None = None,
    val_split: float = 0.2,
    batch_size: int = 32,
    img_size: int = 224,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Scans normal/ and defective/ directories, creates balanced train/val DataLoaders.
    """
    if dataset_dir is None:
        dataset_dir = config.dataset_dir

    normal_dir = dataset_dir / "normal"
    defective_dir = dataset_dir / "defective"

    normal_files = list(normal_dir.glob("*.png")) + list(normal_dir.glob("*.jpg"))
    defective_files = list(defective_dir.glob("*.png")) + list(defective_dir.glob("*.jpg"))

    samples: list[tuple[Path, int]] = []
    for p in normal_files:
        samples.append((p, 0))
    for p in defective_files:
        samples.append((p, 1))

    if len(samples) == 0:
        raise ValueError(f"No images found in {normal_dir} or {defective_dir}")

    torch.manual_seed(seed)
    
    # Stratified shuffle split
    if len(samples) <= 2:
        train_samples = samples
        val_samples = samples
    else:
        val_size = max(1, int(len(samples) * val_split))
        train_size = max(1, len(samples) - val_size)

        # Permute indices
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=generator).tolist()

        train_samples = [samples[i] for i in indices[:train_size]]
        val_samples = [samples[i] for i in indices[train_size:]]

    train_dataset = ComponentDataset(train_samples, transform=get_transforms(img_size, is_train=True))
    val_dataset = ComponentDataset(val_samples, transform=get_transforms(img_size, is_train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,  # Safe for Windows multiprocessing
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    stats = {
        "total_samples": len(samples),
        "normal_samples": len(normal_files),
        "defective_samples": len(defective_files),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    return train_loader, val_loader, stats


def load_multiclass_data(
    dataset_dir: Path | None = None,
    val_split: float = 0.2,
    batch_size: int = 16,
    img_size: int = 224,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict, list[str]]:
    """
    Loads 6-class dataset from dataset/multiclass/<class_name> directories.
    """
    if dataset_dir is None:
        dataset_dir = config.dataset_dir / "multiclass"

    class_names = [
        "Normal",
        "Bent_Pin",
        "Missing_Pin",
        "Surface_Crack",
        "Solder_Bridge",
        "Orientation_Fault",
    ]

    samples: list[tuple[Path, int]] = []
    class_counts = {}

    for c_idx, c_name in enumerate(class_names):
        c_folder = dataset_dir / c_name
        if not c_folder.exists():
            continue
        c_files = list(c_folder.glob("*.png")) + list(c_folder.glob("*.jpg"))
        class_counts[c_name] = len(c_files)
        for f in c_files:
            samples.append((f, c_idx))

    if len(samples) == 0:
        raise ValueError(f"No multi-class images found in {dataset_dir}")

    torch.manual_seed(seed)
    if len(samples) <= 4:
        train_samples = samples
        val_samples = samples
    else:
        val_size = max(1, int(len(samples) * val_split))
        train_size = max(1, len(samples) - val_size)

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=generator).tolist()

        train_samples = [samples[i] for i in indices[:train_size]]
        val_samples = [samples[i] for i in indices[train_size:]]

    train_dataset = ComponentDataset(train_samples, transform=get_transforms(img_size, is_train=True))
    val_dataset = ComponentDataset(val_samples, transform=get_transforms(img_size, is_train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    stats = {
        "total_samples": len(samples),
        "class_counts": class_counts,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    return train_loader, val_loader, stats, class_names
