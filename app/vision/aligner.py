"""
Image Registration and Alignment Module for Electronic Component Inspection.
Aligns inspection frames with a golden reference standard using ORB feature matching and Homography.
"""
import cv2
import numpy as np


class ComponentAligner:
    """Registers and perspective-warps inspection images to match a reference template."""

    def __init__(self, n_features: int = 2500, match_ratio: float = 0.80):
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=7, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = match_ratio

    def align(
        self,
        image_to_align: np.ndarray,
        reference_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        """
        Aligns image_to_align against reference_image using ORB keypoint matching & Homography.

        Returns:
            aligned_image (np.ndarray): Warped image matching reference geometry.
            homography_matrix (np.ndarray or None): 3x3 transformation matrix.
            is_aligned (bool): True if alignment succeeded with sufficient inliers.
        """
        # Convert to grayscale for feature extraction
        gray_src = (
            cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
            if len(image_to_align.shape) == 3
            else image_to_align
        )
        gray_ref = (
            cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            if len(reference_image.shape) == 3
            else reference_image
        )

        h, w = gray_ref.shape[:2]

        kp1, des1 = self.orb.detectAndCompute(gray_src, None)
        kp2, des2 = self.orb.detectAndCompute(gray_ref, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return image_to_align.copy(), None, False

        # KNN Match with Lowe's Ratio Test
        try:
            raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        except Exception:
            return image_to_align.copy(), None, False

        good_matches = []
        for m_n in raw_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 8:
            # Not enough good matches; fallback to resized image
            resized = cv2.resize(image_to_align, (w, h))
            return resized, None, False

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate Homography using RANSAC
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.5)

        if H is None or inliers is None or np.sum(inliers) < 6:
            resized = cv2.resize(image_to_align, (w, h))
            return resized, None, False

        # Warp image with neutral border fill matching inspection background
        aligned = cv2.warpPerspective(
            image_to_align,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned, H, True
