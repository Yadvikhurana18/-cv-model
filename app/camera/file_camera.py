"""
Simulation / File Camera Source for offline testing, batch dataset streaming, and synthetic component generation.
"""
from pathlib import Path
import random
import cv2
import numpy as np
from app.camera.base_camera import BaseCamera
from app.dataset.generator import ComponentGenerator


class FileCamera(BaseCamera):
    """Simulates a camera feed by cycling through a directory of images or generating synthetic frames on the fly."""

    def __init__(
        self,
        image_dir: Path | str | None = None,
        synthetic_mode: bool = False,
    ):
        self.synthetic_mode = synthetic_mode
        self.image_paths: list[Path] = []
        self.current_idx = 0
        self.generator = ComponentGenerator() if synthetic_mode else None

        if image_dir:
            p = Path(image_dir)
            if p.exists():
                self.image_paths = sorted(
                    list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
                )

        self.connected = True

    def connect(self) -> bool:
        self.connected = True
        return True

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        if not self.connected:
            return False, None

        if self.synthetic_mode and self.generator is not None:
            defect_choice = random.choice([
                "Normal", "Bent_Pin", "Missing_Pin", "Surface_Crack", "Solder_Bridge"
            ])
            frame, _ = self.generator.generate_component(defect_type=defect_choice)
            return True, frame

        if not self.image_paths:
            # Fallback black test frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No Images in Directory", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return True, blank

        img_path = self.image_paths[self.current_idx % len(self.image_paths)]
        self.current_idx += 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected
