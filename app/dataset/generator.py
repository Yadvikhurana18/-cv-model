"""
Procedural Dataset Generator for Arduino Uno Microcontroller Board Inspection.
Generates photorealistic Arduino Uno samples with authentic physical variations,
lighting gradients, perspective shifts, and synthetic defect injections:
- Bent Pins (Digital/Analog/Power headers, ICSP, ATmega328P pins)
- Missing Pins (Header pin gaps, empty socket cavities)
- Missing Components (Missing ATmega328P IC, missing 47uF capacitor)
- Surface Cracks / Scratches (PCB solder mask gouges, severed traces)
- Solder Bridges (Solder shorts between adjacent header pins/pads)
- Orientation Faults (Inverted ATmega328P chip with reverse notch/markings)
"""
import math
import random
from pathlib import Path
import cv2
import numpy as np
from app.config import config


class ComponentGenerator:
    """Procedural generator for Arduino Uno board inspection samples."""

    def __init__(
        self,
        reference_path: Path | str | None = None,
        width: int = 512,
        height: int = 512,
    ):
        self.width = width
        self.height = height

        if reference_path is None:
            if (config.reference_dir / "arduino_uno_reference.png").exists():
                reference_path = config.reference_dir / "arduino_uno_reference.png"
            else:
                reference_path = config.reference_dir / "golden_reference.png"

        self.ref_path = Path(reference_path)
        if self.ref_path.exists():
            self.base_ref = cv2.imread(str(self.ref_path))
        else:
            self.base_ref = None

        # Calibrated landmarks on 1024x744 Arduino Uno reference image:
        # ATmega328P DIP-28 IC: (x1, y1, x2, y2)
        self.atmega_bbox = (465, 435, 940, 570)
        # Capacitors (2x 47uF):
        self.cap1_bbox = (305, 545, 385, 655)
        self.cap2_bbox = (395, 545, 475, 655)
        # Crystal 16MHz:
        self.crystal_bbox = (285, 345, 425, 415)
        # Top Digital Header:
        self.digital_header_bbox = (260, 35, 940, 95)
        # Bottom Power & Analog Headers:
        self.power_header_bbox = (455, 660, 715, 715)
        self.analog_header_bbox = (735, 660, 945, 715)
        # ICSP 2x3 Header:
        self.icsp_bbox = (885, 305, 955, 415)
        # USB Connector:
        self.usb_bbox = (25, 140, 205, 305)
        # DC Barrel Jack:
        self.barrel_bbox = (80, 545, 275, 675)

    def _get_base_image(self) -> np.ndarray:
        """Returns a pristine copy of the base Arduino Uno board in standard 1024x744 canvas."""
        if self.base_ref is not None:
            if self.base_ref.shape[:2] != (744, 1024):
                return cv2.resize(self.base_ref, (1024, 744), interpolation=cv2.INTER_LINEAR)
            return self.base_ref.copy()
        # Fallback procedural Arduino Uno canvas if file not found
        img = np.full((744, 1024, 3), 245, dtype=np.uint8)
        # Blue PCB
        cv2.rectangle(img, (100, 40), (970, 700), (145, 85, 25), -1)
        cv2.rectangle(img, (100, 40), (970, 700), (100, 50, 15), 3)
        # ATmega328P
        cv2.rectangle(img, (465, 435), (940, 570), (35, 35, 35), -1)
        cv2.putText(img, "ATMEGA328P-PU", (500, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        return img

    def _inject_missing_pin(self, img: np.ndarray) -> dict:
        """Removes a pin/socket from Digital, Analog, Power, or ICSP headers."""
        choice = random.choice(["digital", "power", "analog", "icsp"])
        if choice == "digital":
            x_min, x_max, y_min, y_max = 280, 920, 40, 85
        elif choice == "power":
            x_min, x_max, y_min, y_max = 470, 700, 665, 710
        elif choice == "analog":
            x_min, x_max, y_min, y_max = 750, 930, 665, 710
        else:
            x_min, x_max, y_min, y_max = 890, 950, 315, 405

        px = random.randint(x_min, x_max - 20)
        py = random.randint(y_min, y_max - 15)
        pw = random.randint(16, 24)
        ph = random.randint(16, 24)

        # Draw empty / broken socket cavity exposing substrate / missing pin gap
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (180, 160, 90), -1)  # Exposed brass/solder pad
        cv2.rectangle(img, (px + 3, py + 3), (px + pw - 3, py + ph - 3), (20, 20, 25), -1)  # Socket hole
        cv2.circle(img, (px + pw // 2, py + ph // 2), 3, (210, 215, 220), -1)  # Pad center

        return {"type": "Missing_Pin", "bbox": (px, py, px + pw, py + ph), "region": choice}

    def _inject_bent_pin(self, img: np.ndarray) -> dict:
        """Injects a deformed/bent pin protruding at an unnatural angle."""
        choice = random.choice(["digital_header", "power_header", "icsp_header", "atmega_pin"])
        if choice == "digital_header":
            px = random.randint(300, 900)
            py = random.randint(45, 75)
            # Draw bent metallic pin
            bend_dx = random.choice([-15, -12, 12, 15])
            bend_dy = random.choice([-10, 10])
            pts = np.array([[px, py], [px + bend_dx // 2, py + bend_dy // 2], [px + bend_dx, py + bend_dy]], np.int32)
            cv2.polylines(img, [pts], False, (220, 225, 230), 4)
            cv2.polylines(img, [pts], False, (250, 250, 255), 2)
            bbox = (min(px, px + bend_dx) - 4, min(py, py + bend_dy) - 4, max(px, px + bend_dx) + 4, max(py, py + bend_dy) + 4)
        elif choice == "icsp_header":
            px = random.randint(895, 945)
            py = random.randint(320, 395)
            bend_dx = random.choice([-14, 14])
            pts = np.array([[px, py], [px + bend_dx // 2, py - 6], [px + bend_dx, py - 12]], np.int32)
            cv2.polylines(img, [pts], False, (215, 220, 225), 3)
            bbox = (min(px, px + bend_dx) - 4, py - 16, max(px, px + bend_dx) + 4, py + 4)
        else:
            # ATmega328P side pin bent
            is_top_side = random.choice([True, False])
            px = random.randint(500, 890)
            py = 438 if is_top_side else 568
            dy = -14 if is_top_side else 14
            dx = random.choice([-8, 8])
            pts = np.array([[px, py], [px + dx // 2, py + dy // 2], [px + dx, py + dy]], np.int32)
            cv2.polylines(img, [pts], False, (195, 200, 205), 3)
            bbox = (px - 8, min(py, py + dy) - 4, px + 12, max(py, py + dy) + 4)

        return {"type": "Bent_Pin", "bbox": bbox, "region": choice}

    def _inject_solder_bridge(self, img: np.ndarray) -> dict:
        """Injects a solder short between adjacent header pins or IC pins."""
        choice = random.choice(["digital_header", "power_header", "atmega_pins", "crystal_pads"])
        if choice == "digital_header":
            px = random.randint(320, 880)
            py = random.randint(48, 72)
            # Solder blob spanning ~25px
            cv2.ellipse(img, (px, py), (random.randint(12, 18), random.randint(7, 10)), 0, 0, 360, (180, 190, 195), -1)
            cv2.ellipse(img, (px - 2, py - 2), (random.randint(6, 9), random.randint(3, 5)), 0, 0, 360, (235, 240, 245), -1)
            bbox = (px - 18, py - 10, px + 18, py + 10)
        elif choice == "power_header":
            px = random.randint(490, 680)
            py = random.randint(675, 700)
            cv2.ellipse(img, (px, py), (random.randint(12, 16), random.randint(6, 9)), 0, 0, 360, (175, 185, 190), -1)
            cv2.ellipse(img, (px - 2, py - 2), (6, 3), 0, 0, 360, (230, 235, 240), -1)
            bbox = (px - 16, py - 9, px + 16, py + 9)
        else:
            # Solder bridge on ATmega328P pins
            px = random.randint(520, 860)
            py = random.choice([440, 565])
            cv2.ellipse(img, (px, py), (random.randint(10, 15), random.randint(8, 12)), 0, 0, 360, (185, 195, 200), -1)
            cv2.ellipse(img, (px - 1, py - 1), (5, 4), 0, 0, 360, (240, 245, 250), -1)
            bbox = (px - 15, py - 12, px + 15, py + 12)

        return {"type": "Solder_Bridge", "bbox": bbox, "region": choice}

    def _inject_surface_crack(self, img: np.ndarray) -> dict:
        """Injects deep scratches / cracked traces across the Arduino Uno board."""
        x1 = random.randint(200, 850)
        y1 = random.randint(120, 620)
        curr_x, curr_y = x1, y1
        num_segments = random.randint(6, 12)
        min_x, min_y, max_x, max_y = x1, y1, x1, y1

        for _ in range(num_segments):
            next_x = curr_x + random.randint(-25, 30)
            next_y = curr_y + random.randint(-15, 25)
            next_x = max(60, min(960, next_x))
            next_y = max(50, min(700, next_y))

            min_x = min(min_x, curr_x, next_x)
            min_y = min(min_y, curr_y, next_y)
            max_x = max(max_x, curr_x, next_x)
            max_y = max(max_y, curr_y, next_y)

            # Draw crack core and highlight
            cv2.line(img, (curr_x, curr_y), (next_x, next_y), (25, 25, 30), random.randint(2, 3))
            cv2.line(img, (curr_x + 1, curr_y + 1), (next_x + 1, next_y + 1), (190, 190, 195), 1)
            curr_x, curr_y = next_x, next_y

        return {"type": "Surface_Crack", "bbox": (min_x, min_y, max_x, max_y)}

    def _inject_orientation_fault(self, img: np.ndarray) -> dict:
        """Inverts the ATmega328P DIP-28 IC (180 degree rotation of the chip body and notch)."""
        x1, y1, x2, y2 = self.atmega_bbox
        chip_crop = img[y1:y2, x1:x2].copy()
        # Rotate chip 180 degrees
        inverted_chip = cv2.rotate(chip_crop, cv2.ROTATE_180)
        img[y1:y2, x1:x2] = inverted_chip
        return {"type": "Orientation_Fault", "bbox": (x1, y1, x2, y2), "component": "ATMEGA328P_INVERTED"}

    def _inject_missing_component(self, img: np.ndarray) -> dict:
        """Removes the ATmega328P IC (leaving empty DIP socket) or a capacitor."""
        choice = random.choice(["atmega328p", "capacitor"])
        if choice == "atmega328p":
            x1, y1, x2, y2 = self.atmega_bbox
            # Draw empty DIP-28 black socket with dual rows of contact holes
            cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 28), -1)
            cv2.rectangle(img, (x1 + 10, y1 + 12), (x2 - 10, y2 - 12), (15, 15, 18), -1)
            # Draw empty socket pin holes (14 per side)
            pin_step = (x2 - x1 - 40) // 14
            for i in range(14):
                px = x1 + 25 + i * pin_step
                # Top hole
                cv2.rectangle(img, (px, y1 + 16), (px + 10, y1 + 28), (8, 8, 10), -1)
                cv2.rectangle(img, (px + 2, y1 + 18), (px + 8, y1 + 26), (110, 115, 120), 1)
                # Bottom hole
                cv2.rectangle(img, (px, y2 - 28), (px + 10, y2 - 16), (8, 8, 10), -1)
                cv2.rectangle(img, (px + 2, y2 - 26), (px + 8, y2 - 18), (110, 115, 120), 1)
            # Center notch
            cv2.circle(img, (x1 + 12, (y1 + y2) // 2), 12, (10, 10, 12), -1)
            bbox = (x1, y1, x2, y2)
            comp = "MISSING_ATMEGA328P"
        else:
            # Remove capacitor, leaving bare circular PCB pads
            x1, y1, x2, y2 = self.cap1_bbox if random.random() < 0.5 else self.cap2_bbox
            # Fill with PCB substrate color
            cv2.rectangle(img, (x1, y1), (x2, y2), (140, 80, 20), -1)
            # Solder pad dots
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy - 18), 7, (170, 175, 180), -1)
            cv2.circle(img, (cx, cy + 18), 7, (170, 175, 180), -1)
            bbox = (x1, y1, x2, y2)
            comp = "MISSING_CAPACITOR"

        return {"type": "Missing_Pin", "bbox": bbox, "component": comp}

    def generate_component(
        self,
        defect_type: str = "Normal",
        add_noise: bool = True,
        random_rotation: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Generates an Arduino Uno board image with optional defect injection and realistic optical variations.
        
        Args:
            defect_type: One of ["Normal", "Bent_Pin", "Missing_Pin", "Surface_Crack", 
                                  "Solder_Bridge", "Orientation_Fault", "Missing_Component"]
        """
        img = self._get_base_image()
        defect_info = {"type": defect_type}

        # 1. Apply Defect
        if defect_type == "Missing_Pin":
            defect_info = self._inject_missing_pin(img)
        elif defect_type == "Bent_Pin":
            defect_info = self._inject_bent_pin(img)
        elif defect_type == "Solder_Bridge":
            defect_info = self._inject_solder_bridge(img)
        elif defect_type == "Surface_Crack":
            defect_info = self._inject_surface_crack(img)
        elif defect_type == "Orientation_Fault":
            defect_info = self._inject_orientation_fault(img)
        elif defect_type == "Missing_Component":
            defect_info = self._inject_missing_component(img)

        # 2. Geometric jitter (slight rotation & slight shift for camera realism)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        if random_rotation:
            angle = random.uniform(-4.5, 4.5)
            shift_x = random.randint(-8, 8)
            shift_y = random.randint(-8, 8)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += shift_x
            M[1, 2] += shift_y
            img = cv2.warpAffine(
                img,
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(245, 245, 245),
            )

        # 3. Photometric variations (Lighting, Exposure, Color temp, Noise)
        if add_noise:
            # Lighting gradient (spotlight or linear reflection)
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            center_light_x = random.randint(w // 4, 3 * w // 4)
            center_light_y = random.randint(h // 4, 3 * h // 4)
            dist_sq = ((x_coords - center_light_x) ** 2 + (y_coords - center_light_y) ** 2) / (float(w * h) / 2.5)
            lighting = np.clip(1.05 - 0.12 * dist_sq, 0.85, 1.15)
            img = np.clip(img.astype(np.float32) * lighting[:, :, np.newaxis], 0, 255).astype(np.uint8)

            # Brightness & contrast jitter
            alpha = random.uniform(0.92, 1.08)
            beta = random.randint(-12, 12)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            # Mild Gaussian noise
            noise = np.random.normal(0, 2.5, (h, w, 3)).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Resize to requested output size
        if (self.width, self.height) != (w, h):
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        label = 0 if defect_type == "Normal" else 1
        metadata = {
            "board": "Arduino Uno R3",
            "defect_type": defect_type,
            "label": label,
            "label_name": "DEFECTIVE" if label == 1 else "NORMAL",
            "defect_info": defect_info,
        }

        return img, metadata

    def generate_multiclass_dataset(
        self,
        samples_per_class: int = 100,
        output_dir: Path | None = None,
    ) -> dict[str, int]:
        """Batch generates 6-class Arduino Uno dataset saved to class subfolders."""
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
        for c_name in class_names:
            class_folder = output_dir / "multiclass" / c_name
            class_folder.mkdir(parents=True, exist_ok=True)
            for i in range(samples_per_class):
                img, meta = self.generate_component(defect_type=c_name)
                cv2.imwrite(str(class_folder / f"uno_{c_name.lower()}_{i:04d}.png"), img)
            counts[c_name] = samples_per_class

        print(f"[Dataset] Arduino Uno multi-class dataset generated: {counts}")
        return counts

    def generate_dataset(
        self,
        num_normal: int = 150,
        num_defective: int = 150,
        output_dir: Path | None = None,
    ) -> dict[str, int]:
        """Batch generates binary Arduino Uno dataset (normal and defective)."""
        if output_dir is None:
            output_dir = config.dataset_dir

        normal_dir = output_dir / "normal"
        defective_dir = output_dir / "defective"
        reference_dir = config.reference_dir
        normal_dir.mkdir(parents=True, exist_ok=True)
        defective_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)

        # Ensure Golden Reference Image is saved
        ref_img, _ = self.generate_component(
            defect_type="Normal", add_noise=False, random_rotation=False
        )
        cv2.imwrite(str(reference_dir / "golden_reference.png"), ref_img)
        cv2.imwrite(str(reference_dir / "arduino_uno_reference.png"), ref_img)

        defect_types = [
            "Bent_Pin",
            "Missing_Pin",
            "Surface_Crack",
            "Solder_Bridge",
            "Orientation_Fault",
            "Missing_Component",
        ]

        print(f"Generating {num_normal} Arduino Uno normal samples...")
        for i in range(num_normal):
            img, _ = self.generate_component(defect_type="Normal")
            cv2.imwrite(str(normal_dir / f"uno_normal_{i:04d}.png"), img)

        print(f"Generating {num_defective} Arduino Uno defective samples...")
        for i in range(num_defective):
            d_type = defect_types[i % len(defect_types)]
            img, _ = self.generate_component(defect_type=d_type)
            cv2.imwrite(str(defective_dir / f"uno_defect_{d_type.lower()}_{i:04d}.png"), img)

        counts = {"normal": num_normal, "defective": num_defective}
        print(f"[Dataset] Arduino Uno binary dataset generation complete: {counts}")
        return counts


if __name__ == "__main__":
    generator = ComponentGenerator()
    generator.generate_dataset(num_normal=150, num_defective=150)
    generator.generate_multiclass_dataset(samples_per_class=100)
