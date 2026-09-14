import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class InspectionConfig:
    # Directories
    base_dir: Path = BASE_DIR
    dataset_dir: Path = BASE_DIR / "dataset"
    normal_dir: Path = BASE_DIR / "dataset" / "normal"
    defective_dir: Path = BASE_DIR / "dataset" / "defective"
    reference_dir: Path = BASE_DIR / "reference"
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"
    
    # Model Configuration
    model_name: str = "mobilenet_v3_large"
    num_classes: int = 2  # 0: Normal, 1: Defective
    img_size: int = 224
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    best_model_path: Path = BASE_DIR / "models" / "component_defect_net.pth"
    
    # Vision & Classical Inspection Thresholds
    ssim_threshold: float = 0.88         # SSIM < 0.88 indicates genuine physical anomaly
    diff_area_threshold: int = 30        # Minimum pixel area to consider as anomaly (filters noise)
    diff_pixel_thresh: int = 40          # Grayscale diff intensity threshold (resists lighting jitter)
    blur_kernel_size: int = 5
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    
    # Camera Defaults
    default_camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    
    # Defect Class Mapping
    defect_classes: list[str] = field(default_factory=lambda: ["NORMAL", "DEFECTIVE"])
    defect_subtypes: list[str] = field(default_factory=lambda: [
        "Normal", "Bent_Pin", "Missing_Pin", "Surface_Crack", "Solder_Bridge", "Orientation_Fault"
    ])

config = InspectionConfig()

# Ensure directories exist
for folder in [
    config.dataset_dir,
    config.normal_dir,
    config.defective_dir,
    config.reference_dir,
    config.models_dir,
    config.logs_dir,
]:
    folder.mkdir(parents=True, exist_ok=True)
