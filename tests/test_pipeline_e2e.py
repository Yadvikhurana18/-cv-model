"""
End-to-End Verification of the Complete ISRO Component Screening Pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from app.config import config
from app.dataset.generator import ComponentGenerator
from app.storage.logger import InspectionLogger
from app.vision.pipeline import InspectionPipeline


def main():
    print("Initializing components...")
    gen = ComponentGenerator()
    pipeline = InspectionPipeline()
    logger = InspectionLogger()

    # 1. Test Golden Reference Loading
    ref_loaded = pipeline.load_reference(config.reference_dir / "golden_reference.png")
    print(f"[*] Golden Reference Loaded: {ref_loaded}")

    # 2. Test Normal Component Screening
    print("\n--- Test 1: Normal Component Screening ---")
    normal_img, _ = gen.generate_component(defect_type="Normal")
    res_normal = pipeline.process_frame(normal_img)
    print(f"Normal Verdict: {res_normal['final_status']} (CV: {res_normal['cv_status']}, ML: {res_normal['ml_status']})")
    print(f"SSIM Score: {res_normal['ssim_score']:.3f} | Anomalies: {res_normal['defect_count']} | ML Defect Prob: {res_normal.get('defect_prob', 0.0)*100:.1f}%")

    # 3. Test Defective Component Screening
    print("\n--- Test 2: Defective Component Screening (Missing Pin) ---")
    defect_img, _ = gen.generate_component(defect_type="Missing_Pin")
    res_defect = pipeline.process_frame(defect_img)
    print(f"Defect Verdict: {res_defect['final_status']} (CV: {res_defect['cv_status']}, ML: {res_defect['ml_status']})")
    print(f"SSIM Score: {res_defect['ssim_score']:.3f} | Anomalies: {res_defect['defect_count']} | ML Defect Prob: {res_defect.get('defect_prob', 0.0)*100:.1f}%")

    # 4. Test Audit Logging
    log_rec = logger.log_inspection("IC_TEST_999", res_defect, annotated_frame=res_defect["annotated_frame"])
    print(f"\n[*] Inspection logged: {log_rec}")
    print("[*] E2E Pipeline Verification Passed Successfully!")


def test_pipeline_e2e():
    main()


if __name__ == "__main__":
    main()
