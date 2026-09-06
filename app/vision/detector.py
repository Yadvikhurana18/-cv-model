"""
Defect Detection and Classical Vision Analysis Engine for Electronic Components.
Performs SSIM calculation, morphological difference analysis, pin integrity, and anomaly bounding box extraction.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.config import config


class DefectDetector:
    """Detects structural and visual defects by comparing an aligned test image with a reference template."""

    def __init__(
        self,
        ssim_thresh: float | None = None,
        min_defect_area: int | None = None,
        diff_pixel_thresh: int | None = None,
    ):
        self.ssim_thresh = ssim_thresh if ssim_thresh is not None else config.ssim_threshold
        self.min_defect_area = (
            min_defect_area if min_defect_area is not None else config.diff_area_threshold
        )
        self.diff_pixel_thresh = (
            diff_pixel_thresh if diff_pixel_thresh is not None else config.diff_pixel_thresh
        )

    def compute_ssim(
        self, test_gray: np.ndarray, ref_gray: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Computes Structural Similarity Index (SSIM) between test and reference grayscale images.

        Returns:
            score (float): SSIM similarity score between 0.0 and 1.0.
            diff_map (np.ndarray): SSIM difference map scaled to 0-255 uint8.
        """
        if test_gray.shape != ref_gray.shape:
            test_gray = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]))

        score, diff = ssim(ref_gray, test_gray, full=True)
        diff_map = ((1.0 - diff) * 255).astype(np.uint8)
        return float(score), diff_map

    def compute_diff_contours(
        self, test_gray: np.ndarray, ref_gray: np.ndarray
    ) -> tuple[list[dict], np.ndarray]:
        """
        Computes absolute morphological difference, filters noise, and extracts defect bounding boxes.

        Returns:
            defect_boxes (list[dict]): List of defect records with bbox, area, and contour.
            thresh_mask (np.ndarray): Binary defect mask.
        """
        if test_gray.shape != ref_gray.shape:
            test_gray = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]))

        # Absolute difference
        abs_diff = cv2.absdiff(ref_gray, test_gray)

        # Thresholding
        _, thresh = cv2.threshold(
            abs_diff, self.diff_pixel_thresh, 255, cv2.THRESH_BINARY
        )

        # Morphological opening and dilation to remove speckle noise and bridge clusters
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        dilated = cv2.dilate(cleaned, kernel_dilate, iterations=2)

        # Find defect contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_defect_area:
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({
                    "bbox": (x, y, w, h),
                    "area": float(area),
                    "contour": cnt,
                })

        return defects, dilated

    def analyze(
        self,
        aligned_test_bgr: np.ndarray,
        reference_bgr: np.ndarray,
    ) -> dict:
        """
        Full defect analysis comparing aligned inspection image with golden reference.

        Returns:
            dict containing:
                - is_defective (bool)
                - status ("PASS" / "FAIL")
                - ssim_score (float)
                - defect_count (int)
                - defects (list[dict])
                - diff_mask (np.ndarray)
                - ssim_map (np.ndarray)
                - annotated_image (np.ndarray)
        """
        test_gray = (
            cv2.cvtColor(aligned_test_bgr, cv2.COLOR_BGR2GRAY)
            if len(aligned_test_bgr.shape) == 3
            else aligned_test_bgr
        )
        ref_gray = (
            cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
            if len(reference_bgr.shape) == 3
            else reference_bgr
        )

        ssim_score, ssim_map = self.compute_ssim(test_gray, ref_gray)
        defects, diff_mask = self.compute_diff_contours(test_gray, ref_gray)

        # Defect criteria: SSIM below threshold OR significant defect regions detected
        is_defective = (ssim_score < self.ssim_thresh) or (len(defects) > 0)
        status = "FAIL" if is_defective else "PASS"

        # Render annotations
        annotated = aligned_test_bgr.copy()
        for d in defects:
            x, y, w, h = d["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"DEFECT ({int(d['area'])}px)",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw status banner
        banner_color = (0, 0, 220) if is_defective else (0, 180, 0)
        cv2.putText(
            annotated,
            f"STATUS: {status} | SSIM: {ssim_score:.3f} | Anomalies: {len(defects)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            banner_color,
            2,
            cv2.LINE_AA,
        )

        return {
            "is_defective": is_defective,
            "status": status,
            "ssim_score": ssim_score,
            "defect_count": len(defects),
            "defects": defects,
            "diff_mask": diff_mask,
            "ssim_map": ssim_map,
            "annotated_image": annotated,
        }
