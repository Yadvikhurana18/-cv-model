"""
Unit Tests for Advanced Pin Metrology, Certificate Generation, and API Server.
"""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from app.dataset.generator import ComponentGenerator
from app.storage.report_generator import ReportGenerator
from app.vision.pin_analyzer import PinAnalyzer


class TestAdvancedFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = ComponentGenerator(width=400, height=400)
        cls.pin_analyzer = PinAnalyzer()
        cls.report_gen = ReportGenerator()
        cls.test_img, _ = cls.generator.generate_component(defect_type="Normal", random_rotation=False)

    def test_pin_analyzer(self):
        """Test pin metrology on generated IC."""
        res = self.pin_analyzer.analyze_pins(self.test_img)
        self.assertIn("total_pins", res)
        self.assertIn("pitch_mean", res)
        self.assertIn("annotated_image", res)
        self.assertGreaterEqual(res["total_pins"], 12)

    def test_report_generator(self):
        """Test HTML certificate generation."""
        mock_results = {
            "final_status": "PASS",
            "ssim_score": 0.985,
            "defect_count": 0,
            "defect_prob": 0.005,
            "summary": "All tests passed",
            "annotated_frame": self.test_img,
            "gradcam_overlay": self.test_img,
        }
        cert_path = self.report_gen.generate_inspection_certificate("TEST_CERT_001", mock_results)
        self.assertTrue(cert_path.exists())
        self.assertGreater(cert_path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
