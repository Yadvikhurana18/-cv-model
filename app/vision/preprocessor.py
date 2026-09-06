"""
Computer Vision Preprocessing Module for Electronic Component Screening.
Applies CLAHE, illumination normalization, bilateral denoising, and edge enhancement.
"""
import cv2
import numpy as np
from app.config import config


class ImagePreprocessor:
    """Provides modular image preprocessing steps for CV inspection."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        grid_size: tuple[int, int] = (8, 8),
        blur_kernel: int = 5,
    ):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.blur_kernel = blur_kernel

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts BGR or RGB image to single-channel grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization to enhance subtle defects."""
        return self.clahe.apply(gray_image)

    def denoise_bilateral(
        self, image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0
    ) -> np.ndarray:
        """Smooths surfaces while strictly preserving crisp pin and IC component edges."""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def normalize_illumination(self, gray_image: np.ndarray) -> np.ndarray:
        """Corrects uneven laboratory/microscope lighting gradients using morphological background division."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        background = cv2.morphologyEx(gray_image, cv2.MORPH_DILATE, kernel)
        background = cv2.GaussianBlur(background, (31, 31), 0)
        
        # Avoid division by zero
        background[background == 0] = 1
        normalized = cv2.divide(gray_image, background, scale=255)
        return normalized.astype(np.uint8)

    def detect_edges(
        self, gray_image: np.ndarray, low_thresh: int = 50, high_thresh: int = 150
    ) -> np.ndarray:
        """Computes Canny edge map."""
        blurred = cv2.GaussianBlur(gray_image, (self.blur_kernel, self.blur_kernel), 0)
        return cv2.Canny(blurred, low_thresh, high_thresh)

    def extract_component_roi(
        self, image: np.ndarray, min_area: int = 5000
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """
        Finds the component bounding box and crops the Region of Interest (ROI).
        """
        gray = self.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, None

        # Find largest contour (IC component)
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) < min_area:
            return image, None

        x, y, w, h = cv2.boundingRect(largest_cnt)
        # Add 10px margin
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        roi = image[y1:y2, x1:x2]
        return roi, (x1, y1, x2 - x1, y2 - y1)

    def preprocess_pipeline(self, bgr_image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Executes complete preprocessing chain.
        Returns dictionary containing grayscale, clahe, denoised, normalized, and edge images.
        """
        gray = self.to_grayscale(bgr_image)
        denoised = self.denoise_bilateral(gray)
        clahe_img = self.apply_clahe(denoised)
        normalized = self.normalize_illumination(clahe_img)
        edges = self.detect_edges(normalized)

        return {
            "gray": gray,
            "denoised": denoised,
            "clahe": clahe_img,
            "normalized": normalized,
            "edges": edges,
        }
