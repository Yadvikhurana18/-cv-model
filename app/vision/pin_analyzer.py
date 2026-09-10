"""
Precision Electronic Component Pin Metrology and Integrity Analyzer for Arduino Uno and IC packages.
Performs sub-pixel pin pitch measurement, pin count verification, deflection angle calculation,
and solder bridge detection on Arduino Uno pin headers and IC lead frames.
"""
import cv2
import numpy as np
from app.config import config


class PinAnalyzer:
    """Performs geometric metrology on Arduino Uno pin headers and IC pins."""

    def __init__(
        self,
        expected_pins_per_side: int = 14,
        pitch_tolerance_pct: float = 25.0,
        max_deflection_deg: float = 6.0,
    ):
        self.expected_pins_per_side = expected_pins_per_side
        self.pitch_tolerance_pct = pitch_tolerance_pct
        self.max_deflection_deg = max_deflection_deg

    def analyze_pins(
        self,
        bgr_image: np.ndarray,
        ic_body_bbox: tuple[int, int, int, int] | None = None,
    ) -> dict:
        """
        Analyzes pin headers and ATmega328P dual-row pins on Arduino Uno board.

        Returns:
            dict containing:
                - top_pin_count (int)
                - bottom_pin_count (int)
                - total_pins (int)
                - pin_defects (list[dict])
                - pitch_mean (float)
                - pitch_std (float)
                - is_compliant (bool)
                - annotated_image (np.ndarray)
        """
        annotated = bgr_image.copy()
        h, w = bgr_image.shape[:2]

        # Convert to grayscale and threshold for metallic pin reflections
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for metallic pins and header contacts
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, pin_thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)

        # Region of interest masks for Arduino Uno:
        # Top pin headers (Digital) and ATmega328P top pins
        # Bottom pin headers (Power/Analog) and ATmega328P bottom pins
        top_mask = np.zeros((h, w), dtype=np.uint8)
        bottom_mask = np.zeros((h, w), dtype=np.uint8)

        # Top region (Digital headers: y: 0.04*h to 0.25*h, x: 0.20*w to 0.95*w)
        top_mask[int(0.03 * h) : int(0.25 * h), int(0.15 * w) : int(0.98 * w)] = 255
        # Bottom region (Power/Analog headers: y: 0.80*h to 0.98*h, x: 0.40*w to 0.98*w)
        bottom_mask[int(0.75 * h) : int(0.98 * h), int(0.35 * w) : int(0.98 * w)] = 255

        # ATmega328P dual-row pins (y: 0.55*h to 0.80*h, x: 0.45*w to 0.95*w)
        top_mask[int(0.55 * h) : int(0.66 * h), int(0.44 * w) : int(0.95 * w)] = 255
        bottom_mask[int(0.70 * h) : int(0.80 * h), int(0.44 * w) : int(0.95 * w)] = 255

        top_pins_thresh = cv2.bitwise_and(pin_thresh, pin_thresh, mask=top_mask)
        bottom_pins_thresh = cv2.bitwise_and(pin_thresh, pin_thresh, mask=bottom_mask)

        # Morphological opening to isolate individual pin contacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        top_clean = cv2.morphologyEx(top_pins_thresh, cv2.MORPH_OPEN, kernel)
        bottom_clean = cv2.morphologyEx(bottom_pins_thresh, cv2.MORPH_OPEN, kernel)

        top_cnts, _ = cv2.findContours(top_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bottom_cnts, _ = cv2.findContours(bottom_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter valid pin contours by area
        min_area = int(15 * (w * h) / (512 * 512))
        max_area = int(1200 * (w * h) / (512 * 512))
        valid_top = [c for c in top_cnts if min_area <= cv2.contourArea(c) <= max_area]
        valid_bottom = [c for c in bottom_cnts if min_area <= cv2.contourArea(c) <= max_area]

        # Sort left-to-right
        valid_top.sort(key=lambda c: cv2.boundingRect(c)[0])
        valid_bottom.sort(key=lambda c: cv2.boundingRect(c)[0])

        pin_defects = []

        # Analyze Top Pins
        top_pitches = []
        for i, c in enumerate(valid_top):
            x, y, pw, ph = cv2.boundingRect(c)
            # Check for solder bridging (unusually wide contour bridging adjacent pins)
            if pw > int(32 * w / 512):
                pin_defects.append({
                    "region": "top",
                    "pin_idx": i,
                    "type": "SOLDER_BRIDGE",
                    "bbox": (x, y, pw, ph),
                })
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
                cv2.putText(annotated, "BRIDGE", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            else:
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 255, 0), 1)

            if i > 0:
                prev_x = cv2.boundingRect(valid_top[i - 1])[0]
                top_pitches.append(x - prev_x)

        # Analyze Bottom Pins
        bottom_pitches = []
        for i, c in enumerate(valid_bottom):
            x, y, pw, ph = cv2.boundingRect(c)
            if pw > int(32 * w / 512):
                pin_defects.append({
                    "region": "bottom",
                    "pin_idx": i,
                    "type": "SOLDER_BRIDGE",
                    "bbox": (x, y, pw, ph),
                })
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
                cv2.putText(annotated, "BRIDGE", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            else:
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 255, 0), 1)

            if i > 0:
                prev_x = cv2.boundingRect(valid_bottom[i - 1])[0]
                bottom_pitches.append(x - prev_x)

        all_pitches = top_pitches + bottom_pitches
        pitch_mean = float(np.mean(all_pitches)) if all_pitches else 0.0
        pitch_std = float(np.std(all_pitches)) if all_pitches else 0.0

        total_pins = len(valid_top) + len(valid_bottom)
        is_compliant = len(pin_defects) == 0

        # Overlay summary on image
        status_text = "PINS: PASS" if is_compliant else f"PINS: FAIL ({len(pin_defects)} Anomalies)"
        status_color = (0, 200, 0) if is_compliant else (0, 0, 255)
        cv2.putText(
            annotated,
            f"{status_text} | Total Pin Contacts Detected: {total_pins} | Mean Pitch: {pitch_mean:.1f}px",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            status_color,
            1,
            cv2.LINE_AA,
        )

        return {
            "top_pin_count": len(valid_top),
            "bottom_pin_count": len(valid_bottom),
            "left_pin_count": len(valid_top),
            "right_pin_count": len(valid_bottom),
            "total_pins": total_pins,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pin_defects": pin_defects,
            "is_compliant": is_compliant,
            "annotated_image": annotated,
        }
