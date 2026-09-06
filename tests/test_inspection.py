import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from app.config import config
from app.dataset.generator import ComponentGenerator
from app.ml.model import ComponentDefectNet
from app.vision.aligner import ComponentAligner
from app.vision.detector import DefectDetector
from app.vision.preprocessor import ImagePreprocessor


class TestInspectionSystem(unittest.TestCase):
    """Test suite for CV inspection system components."""

    @classmethod
    def setUpClass(cls):
        cls.generator = ComponentGenerator(width=300, height=300)
        cls.preprocessor = ImagePreprocessor()
        cls.aligner = ComponentAligner()
        cls.detector = DefectDetector(ssim_thresh=0.90, min_defect_area=30)

        # Generate sample reference and test components
        cls.normal_img, _ = cls.generator.generate_component(defect_type="Normal", random_rotation=False)
        cls.defective_img, _ = cls.generator.generate_component(defect_type="Missing_Pin", random_rotation=False)

    def test_preprocessor(self):
        """Verify preprocessing pipeline outputs."""
        res = self.preprocessor.preprocess_pipeline(self.normal_img)
        self.assertIn("gray", res)
        self.assertIn("clahe", res)
        self.assertIn("normalized", res)
        self.assertEqual(res["gray"].shape, (300, 300))

    def test_detector_normal_component(self):
        """Verify normal component returns PASS with high SSIM against itself."""
        res = self.detector.analyze(self.normal_img, self.normal_img)
        self.assertFalse(res["is_defective"])
        self.assertEqual(res["status"], "PASS")
        self.assertGreaterEqual(res["ssim_score"], 0.95)

    def test_detector_defective_component(self):
        """Verify defective component is flagged as FAIL."""
        res = self.detector.analyze(self.defective_img, self.normal_img)
        self.assertTrue(res["is_defective"])
        self.assertEqual(res["status"], "FAIL")

    def test_pytorch_model_gpu(self):
        """Verify PyTorch ComponentDefectNet forward pass and GPU transfer."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ComponentDefectNet(num_classes=2, pretrained=False).to(device)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            preds, probs = model.predict(dummy_input)

        self.assertEqual(preds.shape, torch.Size([2]))
        self.assertEqual(probs.shape, torch.Size([2, 2]))
        self.assertAlmostEqual(probs.sum(dim=1)[0].item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
