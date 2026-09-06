"""
Synthetic Electronic Component Dataset Generator for ISRO Screening Inspection.
Generates photorealistic IC packages (DIP-14/16/28, QFP, SMD) with realistic textures,
pins, laser markings, and injected defect types (bent pins, missing pins, cracks, solder bridges).
"""
import math
import random
from pathlib import Path
import cv2
import numpy as np
from app.config import config


class ComponentGenerator:
    """Procedural generator for electronic IC and PCB component images."""

    def __init__(self, width: int = 400, height: int = 400):
        self.width = width
        self.height = height

    def _create_ic_body(
        self,
        img: np.ndarray,
        center_x: int,
        center_y: int,
        body_w: int,
        body_h: int,
        has_notch: bool = True,
        notch_top: bool = True,
        text_label: str = "ISRO-RAD750",
    ) -> tuple[int, int, int, int]:
        """Draws the dark epoxy IC body with texture, bevel, notch, and laser etching."""
        x1 = center_x - body_w // 2
        y1 = center_y - body_h // 2
        x2 = x1 + body_w
        y2 = y1 + body_h

        # IC epoxy body base color (dark charcoal / matte black)
        base_color = random.randint(30, 45)
        body_patch = np.random.normal(
            loc=base_color, scale=3, size=(body_h, body_w, 3)
        ).astype(np.uint8)
        img[y1:y2, x1:x2] = body_patch

        # Chamfer/bevel edge
        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
        cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (60, 60, 60), 1)

        # Pin 1 index dot or orientation notch
        if has_notch:
            if notch_top:
                notch_pos = (center_x, y1)
                cv2.circle(img, notch_pos, body_w // 10, (20, 20, 20), -1)
                cv2.circle(img, (x1 + 15, y1 + 15), 5, (75, 75, 75), -1)  # Pin 1 dot
            else:
                # Inverted notch (Defect: wrong orientation)
                notch_pos = (center_x, y2)
                cv2.circle(img, notch_pos, body_w // 10, (20, 20, 20), -1)
                cv2.circle(img, (x2 - 15, y2 - 15), 5, (75, 75, 75), -1)

        # Laser marking text (e.g. part number, date code)
        cv2.putText(
            img,
            text_label,
            (x1 + 15, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        date_code = f"W26{random.randint(10, 52)}"
        cv2.putText(
            img,
            date_code,
            (x1 + 25, center_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

        return x1, y1, x2, y2

    def _draw_dip_pins(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        num_pins_per_side: int = 8,
        defect: str = "Normal",
    ) -> list[dict]:
        """Draws metallic silver/gold pins protruding from left and right sides."""
        body_h = y2 - y1
        pin_spacing = body_h // (num_pins_per_side + 1)
        pin_length = random.randint(22, 28)
        pin_thickness = max(4, pin_spacing // 2)

        # Pin metallic silver color
        pin_color = (195, 200, 205)
        pin_highlight = (230, 235, 240)
        pin_shadow = (110, 115, 120)

        pin_info = []

        defect_pin_idx = random.randint(1, num_pins_per_side - 2) if defect != "Normal" else -1
        defect_side = random.choice(["left", "right"])

        for side in ["left", "right"]:
            for i in range(num_pins_per_side):
                py = y1 + (i + 1) * pin_spacing
                is_defect_pin = (defect != "Normal" and i == defect_pin_idx and side == defect_side)

                if side == "left":
                    px_start = x1 - pin_length
                    px_end = x1
                else:
                    px_start = x2
                    px_end = x2 + pin_length

                if is_defect_pin and defect == "Missing_Pin":
                    # Pin is missing, draw only a small broken solder pad
                    pad_x = x1 if side == "left" else x2 - 5
                    cv2.rectangle(
                        img,
                        (pad_x, py - pin_thickness // 2),
                        (pad_x + 5, py + pin_thickness // 2),
                        (90, 90, 90),
                        -1,
                    )
                    pin_info.append({"side": side, "pin": i, "status": "missing"})
                    continue

                if is_defect_pin and defect == "Bent_Pin":
                    # Bent pin angle
                    bend_offset = random.choice([-12, 12])
                    pts = np.array(
                        [
                            [px_end if side == "left" else px_start, py],
                            [
                                (px_start + px_end) // 2,
                                py + bend_offset // 2,
                            ],
                            [
                                px_start if side == "left" else px_end,
                                py + bend_offset,
                            ],
                        ],
                        np.int32,
                    )
                    cv2.polylines(img, [pts], False, pin_color, pin_thickness)
                    pin_info.append({"side": side, "pin": i, "status": "bent"})
                    continue

                # Normal pin drawing
                cv2.rectangle(
                    img,
                    (px_start, py - pin_thickness // 2),
                    (px_end, py + pin_thickness // 2),
                    pin_color,
                    -1,
                )
                # Pin shine line
                cv2.line(
                    img,
                    (px_start, py - 1),
                    (px_end, py - 1),
                    pin_highlight,
                    1,
                )
                # Pin shadow line
                cv2.line(
                    img,
                    (px_start, py + pin_thickness // 2 - 1),
                    (px_end, py + pin_thickness // 2 - 1),
                    pin_shadow,
                    1,
                )

                pin_info.append({"side": side, "pin": i, "status": "normal"})

        # Handle solder bridge defect between adjacent pins
        if defect == "Solder_Bridge":
            bridge_pin = random.randint(1, num_pins_per_side - 3)
            py1 = y1 + (bridge_pin + 1) * pin_spacing
            py2 = y1 + (bridge_pin + 2) * pin_spacing
            px = (x1 - pin_length // 2) if defect_side == "left" else (x2 + pin_length // 2)
            # Solder blob joining two pins
            cv2.ellipse(
                img,
                (px, (py1 + py2) // 2),
                (random.randint(6, 9), (py2 - py1) // 2 + 3),
                0,
                0,
                360,
                (170, 180, 185),
                -1,
            )
            cv2.circle(
                img,
                (px, (py1 + py2) // 2),
                3,
                (220, 225, 230),
                -1,
            )

        return pin_info

    def _inject_surface_crack(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """Draws realistic jagged surface fracture/scratch on the IC packaging."""
        start_x = random.randint(x1 + 10, x2 - 20)
        start_y = random.randint(y1 + 10, y2 - 20)
        curr_x, curr_y = start_x, start_y

        num_segments = random.randint(5, 10)
        for _ in range(num_segments):
            next_x = curr_x + random.randint(-15, 20)
            next_y = curr_y + random.randint(5, 18)
            next_x = max(x1 + 5, min(x2 - 5, next_x))
            next_y = max(y1 + 5, min(y2 - 5, next_y))
            # Crack shadow + highlight
            cv2.line(img, (curr_x, curr_y), (next_x, next_y), (10, 10, 10), 2)
            cv2.line(
                img,
                (curr_x + 1, curr_y + 1),
                (next_x + 1, next_y + 1),
                (80, 80, 80),
                1,
            )
            curr_x, curr_y = next_x, next_y

    def generate_component(
        self,
        defect_type: str = "Normal",
        add_noise: bool = True,
        random_rotation: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Generates a complete component image with optional defects and inspection metadata.
        
        Args:
            defect_type: One of ["Normal", "Bent_Pin", "Missing_Pin", "Surface_Crack", 
                                  "Solder_Bridge", "Orientation_Fault"]
        """
        # Background: PCB green substrate or inspection tray surface
        bg_type = random.choice(["pcb_green", "tray_blue", "industrial_gray"])
        if bg_type == "pcb_green":
            bg_base = np.array([25, 60, 20], dtype=np.uint8)
        elif bg_type == "tray_blue":
            bg_base = np.array([65, 45, 30], dtype=np.uint8)
        else:
            bg_base = np.array([50, 50, 50], dtype=np.uint8)

        img = np.tile(bg_base, (self.height, self.width, 1))
        # Add slight texture
        noise = np.random.normal(0, 4, (self.height, self.width, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw PCB copper trace lines in background
        for _ in range(random.randint(4, 8)):
            tx1 = random.randint(0, self.width)
            ty1 = random.randint(0, self.height)
            tx2 = random.randint(0, self.width)
            ty2 = random.randint(0, self.height)
            cv2.line(img, (tx1, ty1), (tx2, ty2), (35, 80, 25), 1)

        center_x = self.width // 2 + random.randint(-6, 6)
        center_y = self.height // 2 + random.randint(-6, 6)
        body_w = random.randint(110, 130)
        body_h = random.randint(200, 230)

        notch_top = (defect_type != "Orientation_Fault")
        x1, y1, x2, y2 = self._create_ic_body(
            img, center_x, center_y, body_w, body_h, has_notch=True, notch_top=notch_top
        )

        # Draw Pins
        pins_info = self._draw_dip_pins(img, x1, y1, x2, y2, num_pins_per_side=8, defect=defect_type)

        # Surface Crack defect
        if defect_type == "Surface_Crack":
            self._inject_surface_crack(img, x1, y1, x2, y2)

        # Slight random rotation (e.g. slight placement jitter ±3 degrees for realism)
        if random_rotation:
            angle = random.uniform(-4.0, 4.0)
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            img = cv2.warpAffine(
                img,
                M,
                (self.width, self.height),
                borderMode=cv2.BORDER_REFLECT,
            )

        # Lighting gradient variation
        if add_noise:
            y_coords, x_coords = np.mgrid[0 : self.height, 0 : self.width]
            lighting = 1.0 + 0.08 * (
                np.sin(x_coords / 40.0) + np.cos(y_coords / 40.0)
            )
            img = np.clip(img * lighting[:, :, np.newaxis], 0, 255).astype(np.uint8)

        label = 0 if defect_type == "Normal" else 1
        metadata = {
            "defect_type": defect_type,
            "label": label,
            "label_name": "DEFECTIVE" if label == 1 else "NORMAL",
            "bbox": [x1, y1, x2, y2],
            "pins": pins_info,
        }

        return img, metadata

    def generate_multiclass_dataset(
        self,
        samples_per_class: int = 100,
        output_dir: Path | None = None,
    ) -> dict[str, int]:
        """Batch generates 6-class dataset saved to class-specific subfolders."""
        if output_dir is None:
            output_dir = config.dataset_dir

        class_names = [
            "Normal",
            "Bent_Pin",
            "Missing_Pin",
            "Surface_Crack",
            "Solder_Bridge",
            "Orientation_Fault",
        ]

        counts = {}
        for c_idx, c_name in enumerate(class_names):
            class_folder = output_dir / "multiclass" / c_name
            class_folder.mkdir(parents=True, exist_ok=True)
            for i in range(samples_per_class):
                img, meta = self.generate_component(defect_type=c_name)
                cv2.imwrite(str(class_folder / f"{c_name.lower()}_{i:04d}.png"), img)
            counts[c_name] = samples_per_class

        print(f"[Dataset] Multi-class dataset generated: {counts}")
        return counts

    def generate_dataset(
        self,
        num_normal: int = 150,
        num_defective: int = 150,
        output_dir: Path | None = None,
    ) -> dict[str, int]:
        """Batch generates synthetic training and validation images saved to disk."""
        if output_dir is None:
            output_dir = config.dataset_dir

        normal_dir = output_dir / "normal"
        defective_dir = output_dir / "defective"
        reference_dir = config.reference_dir
        normal_dir.mkdir(parents=True, exist_ok=True)
        defective_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)

        # Generate Golden Reference Image (defect-free, zero rotation, clean)
        ref_img, _ = self.generate_component(
            defect_type="Normal", add_noise=False, random_rotation=False
        )
        cv2.imwrite(str(reference_dir / "golden_reference.png"), ref_img)

        defect_types = [
            "Bent_Pin",
            "Missing_Pin",
            "Surface_Crack",
            "Solder_Bridge",
            "Orientation_Fault",
        ]

        print(f"Generating {num_normal} normal component samples...")
        for i in range(num_normal):
            img, _ = self.generate_component(defect_type="Normal")
            cv2.imwrite(str(normal_dir / f"sample_normal_{i:04d}.png"), img)

        print(f"Generating {num_defective} defective component samples...")
        for i in range(num_defective):
            d_type = defect_types[i % len(defect_types)]
            img, _ = self.generate_component(defect_type=d_type)
            cv2.imwrite(str(defective_dir / f"sample_defect_{d_type.lower()}_{i:04d}.png"), img)

        counts = {"normal": num_normal, "defective": num_defective}
        print(f"Dataset generation complete: {counts}")
        return counts


if __name__ == "__main__":
    generator = ComponentGenerator()
    generator.generate_dataset(num_normal=120, num_defective=120)
