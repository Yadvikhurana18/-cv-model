"""
Unified Inspection Pipeline combining OpenCV Classical Analysis and PyTorch Deep Learning.
Provides end-to-end component screening, multi-layer verification, and composite visualization.
"""
from pathlib import Path
import cv2
import numpy as np

from app.config import config
from app.ml.inference import DefectClassifier
from app.vision.aligner import ComponentAligner
from app.vision.detector import DefectDetector
from app.vision.preprocessor import ImagePreprocessor


class InspectionPipeline:
    """Master inspection orchestrator combining Classical CV and PyTorch Deep Learning."""

    def __init__(
        self,
        reference_image_path: Path | None = None,
        model_path: Path | None = None,
    ):
        self.preprocessor = ImagePreprocessor()
        self.aligner = ComponentAligner()
        self.detector = DefectDetector()
        self.classifier = DefectClassifier(model_path=model_path)

        self.reference_image: np.ndarray | None = None
        if reference_image_path:
            self.load_reference(reference_image_path)
        else:
            default_ref = config.reference_dir / "golden_reference.png"
            if default_ref.exists():
                self.load_reference(default_ref)

    def load_reference(self, path: Path | str) -> bool:
        """Loads and sets the golden reference template image."""
        p = Path(path)
        if not p.exists():
            return False
        img = cv2.imread(str(p))
        if img is not None:
            self.reference_image = img
            return True
        return False

    def set_reference_image(self, img: np.ndarray) -> None:
        """Sets the golden reference directly from an in-memory OpenCV BGR array."""
        self.reference_image = img.copy()

    def process_frame(
        self,
        frame: np.ndarray,
        use_deep_learning: bool = True,
        use_classical_cv: bool = True,
    ) -> dict:
        """
        Executes end-to-end screening on an incoming camera or file frame.

        Returns:
            dict containing comprehensive inspection verdict, metrics, overlays, and masks.
        """
        results = {
            "is_defective": False,
            "final_status": "PASS",
            "cv_status": "PASS",
            "ml_status": "PASS",
            "ssim_score": 1.0,
            "ml_confidence": 0.0,
            "defect_count": 0,
            "is_aligned": False,
            "aligned_image": frame.copy(),
            "annotated_frame": frame.copy(),
            "gradcam_overlay": None,
            "diff_mask": None,
            "ssim_map": None,
            "summary": "OK",
        }

        # Step 1: Classical CV Inspection against Golden Reference
        if use_classical_cv and self.reference_image is not None:
            aligned_img, _, is_aligned = self.aligner.align(frame, self.reference_image)
            results["is_aligned"] = is_aligned
            results["aligned_image"] = aligned_img

            cv_analysis = self.detector.analyze(aligned_img, self.reference_image)
            results["cv_status"] = cv_analysis["status"]
            results["ssim_score"] = cv_analysis["ssim_score"]
            results["defect_count"] = cv_analysis["defect_count"]
            results["diff_mask"] = cv_analysis["diff_mask"]
            results["ssim_map"] = cv_analysis["ssim_map"]
            results["annotated_frame"] = cv_analysis["annotated_image"]

            if cv_analysis["is_defective"]:
                results["is_defective"] = True

        # Step 2: PyTorch Deep Learning Inspection (ComponentDefectNet)
        if use_deep_learning:
            img_for_ml = results["aligned_image"] if results["is_aligned"] else frame
            ml_pred = self.classifier.predict(img_for_ml, compute_cam=True)

            results["ml_status"] = ml_pred["status"]
            results["ml_confidence"] = ml_pred["confidence"]
            results["defect_prob"] = ml_pred["defect_prob"]
            results["normal_prob"] = ml_pred["normal_prob"]
            results["gradcam_overlay"] = ml_pred["overlay_image"]

            if ml_pred["is_defective"]:
                results["is_defective"] = True

        # Final consensus verdict
        results["final_status"] = "FAIL" if results["is_defective"] else "PASS"

        # Build human-readable summary
        reasons = []
        if results["cv_status"] == "FAIL":
            reasons.append(f"CV Structural Mismatch (SSIM: {results['ssim_score']:.2f}, Defects: {results['defect_count']})")
        if results["ml_status"] == "FAIL":
            reasons.append(f"Deep Learning Anomaly Detected ({results['ml_confidence']*100:.1f}%)")

        results["summary"] = "; ".join(reasons) if reasons else "Component verified compliant."
        return results
