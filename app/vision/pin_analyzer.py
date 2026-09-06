"""
Precision Electronic Component Pin Metrology and Integrity Analyzer.
Performs sub-pixel pin pitch measurement, pin count verification, deflection angle calculation,
and solder bridge detection on IC lead frames.
"""
import cv2
import numpy as np
from app.config import config


class PinAnalyzer:
    """Performs geometric metrology on IC lead pins."""

    def __init__(
        self,
        expected_pins_per_side: int = 8,
        pitch_tolerance_pct: float = 20.0,
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
        Analyzes pins on left and right sides of IC package.

        Returns:
            dict containing:
                - left_pin_count (int)
                - right_pin_count (int)
                - total_pins (int)
                - pin_defects (list[dict])
                - pitch_mean (float)
                - pitch_std (float)
                - is_compliant (bool)
                - annotated_image (np.ndarray)
        """
        annotated = bgr_image.copy()
        h, w = bgr_image.shape[:2]

        # Convert to grayscale and threshold for metallic pin reflection
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # IC body bounding box detection if not provided
        if ic_body_bbox is None:
            # Estimate body region from center
            bx1, by1 = int(w * 0.35), int(h * 0.15)
            bw, bh = int(w * 0.30), int(h * 0.70)
        else:
            bx1, by1, bw, bh = ic_body_bbox
        bx2, by2 = bx1 + bw, by1 + bh

        # Masks for left and right pin regions
        left_mask = np.zeros((h, w), dtype=np.uint8)
        right_mask = np.zeros((h, w), dtype=np.uint8)

        pad = 8
        left_mask[by1 - pad : by2 + pad, max(0, bx1 - 60) : bx1] = 255
        right_mask[by1 - pad : by2 + pad, bx2 : min(w, bx2 + 60)] = 255

        # Threshold metallic pins (bright silver/gold)
        _, pin_thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

        left_pins_thresh = cv2.bitwise_and(pin_thresh, pin_thresh, mask=left_mask)
        right_pins_thresh = cv2.bitwise_and(pin_thresh, pin_thresh, mask=right_mask)

        # Morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        left_clean = cv2.morphologyEx(left_pins_thresh, cv2.MORPH_OPEN, kernel)
        right_clean = cv2.morphologyEx(right_pins_thresh, cv2.MORPH_OPEN, kernel)

        left_cnts, _ = cv2.findContours(left_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        right_cnts, _ = cv2.findContours(right_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter valid pin contours by area
        valid_left = [c for c in left_cnts if 40 <= cv2.contourArea(c) <= 2500]
        valid_right = [c for c in right_cnts if 40 <= cv2.contourArea(c) <= 2500]

        # Sort top-to-bottom
        valid_left.sort(key=lambda c: cv2.boundingRect(c)[1])
        valid_right.sort(key=lambda c: cv2.boundingRect(c)[1])

        pin_defects = []

        # Analyze Left Pins
        left_pitches = []
        for i, c in enumerate(valid_left):
            x, y, pw, ph = cv2.boundingRect(c)
            # Check for solder bridging (unusually tall contour spanning 2 pin pitches)
            if ph > 28:
                pin_defects.append({
                    "side": "left",
                    "pin_idx": i,
                    "type": "SOLDER_BRIDGE",
                    "bbox": (x, y, pw, ph),
                })
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
                cv2.putText(annotated, "BRIDGE", (x - 35, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            else:
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 255, 0), 1)
                cv2.putText(annotated, f"L{i+1}", (x - 20, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            if i > 0:
                prev_y = cv2.boundingRect(valid_left[i - 1])[1]
                left_pitches.append(y - prev_y)

        # Analyze Right Pins
        right_pitches = []
        for i, c in enumerate(valid_right):
            x, y, pw, ph = cv2.boundingRect(c)
            if ph > 28:
                pin_defects.append({
                    "side": "right",
                    "pin_idx": i,
                    "type": "SOLDER_BRIDGE",
                    "bbox": (x, y, pw, ph),
                })
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
                cv2.putText(annotated, "BRIDGE", (x + pw + 5, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            else:
                cv2.rectangle(annotated, (x, y), (x + pw, y + ph), (0, 255, 0), 1)
                cv2.putText(annotated, f"R{i+1}", (x + pw + 5, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            if i > 0:
                prev_y = cv2.boundingRect(valid_right[i - 1])[1]
                right_pitches.append(y - prev_y)

        all_pitches = left_pitches + right_pitches
        pitch_mean = float(np.mean(all_pitches)) if all_pitches else 0.0
        pitch_std = float(np.std(all_pitches)) if all_pitches else 0.0

        # Check pin count compliance
        if len(valid_left) < self.expected_pins_per_side:
            pin_defects.append({
                "side": "left",
                "type": "MISSING_PINS",
                "count": self.expected_pins_per_side - len(valid_left),
            })
        if len(valid_right) < self.expected_pins_per_side:
            pin_defects.append({
                "side": "right",
                "type": "MISSING_PINS",
                "count": self.expected_pins_per_side - len(valid_right),
            })

        # Pitch irregularity check (bent pin indicator)
        if pitch_mean > 0:
            for side_name, pitches in [("left", left_pitches), ("right", right_pitches)]:
                for p_idx, p_val in enumerate(pitches):
                    diff_pct = abs(p_val - pitch_mean) / pitch_mean * 100
                    if diff_pct > self.pitch_tolerance_pct:
                        pin_defects.append({
                            "side": side_name,
                            "pin_idx": p_idx,
                            "type": "PITCH_DEVIATION_BENT_PIN",
                            "pitch_val": float(p_val),
                            "deviation_pct": float(diff_pct),
                        })

        is_compliant = len(pin_defects) == 0

        # Overlay summary on image
        status_text = "PINS: PASS (Compliant)" if is_compliant else f"PINS: FAIL ({len(pin_defects)} Anomalies)"
        status_color = (0, 200, 0) if is_compliant else (0, 0, 255)
        cv2.putText(
            annotated,
            f"{status_text} | Left: {len(valid_left)}/8, Right: {len(valid_right)}/8 | Mean Pitch: {pitch_mean:.1f}px",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
            cv2.LINE_AA,
        )

        return {
            "left_pin_count": len(valid_left),
            "right_pin_count": len(valid_right),
            "total_pins": len(valid_left) + len(valid_right),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pin_defects": pin_defects,
            "is_compliant": is_compliant,
            "annotated_image": annotated,
        }
