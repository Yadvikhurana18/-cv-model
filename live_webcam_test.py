"""
Real-Time Arduino Uno Webcam Inspection & Live CV Model Test.
Opens your local webcam, runs real-time ORB alignment against the Golden Reference,
computes SSIM difference contours, executes PyTorch ComponentDefectNet deep learning inference,
and renders real-time Grad-CAM defect heatmaps directly on screen.

Controls:
    - Press 'q' to Quit
    - Press 's' to Save Inspection Snapshot & Certificate
    - Press 'c' to Toggle Classical CV vs Deep Learning view
    - Press '0', '1', '2' to switch Camera Index
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from app.config import config
from app.storage.logger import InspectionLogger
from app.storage.report_generator import ReportGenerator
from app.vision.pin_analyzer import PinAnalyzer
from app.vision.pipeline import InspectionPipeline


def run_live_webcam(camera_index: int = 0):
    print("=" * 65)
    print("🛰️ A.R.G.U.S. - Arduino Uno Live Webcam Screening")
    print(f"Opening Camera Index: {camera_index}...")
    print("Controls:")
    print("  [q] -> Quit")
    print("  [s] -> Save Snapshot & Certificate")
    print("  [c] -> Toggle View (Normal / Grad-CAM / SSIM Diff)")
    print("=" * 65)

    pipeline = InspectionPipeline()
    logger = InspectionLogger()
    reporter = ReportGenerator()
    pin_analyzer = PinAnalyzer()

    # Load Golden Reference
    ref_loaded = pipeline.load_reference(config.reference_dir / "golden_reference.png")
    if not ref_loaded:
        print("[!] Warning: golden_reference.png not found in reference directory.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] Error: Could not open webcam device {camera_index}.")
        print("    Try changing camera index to 1 or 2.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    view_mode = 0  # 0: Composite, 1: Grad-CAM Overlay, 2: Diff Mask
    view_names = ["Master Inspection Composite", "Deep Learning Grad-CAM", "SSIM Difference Mask"]
    fps_start = time.time()
    frame_count = 0
    fps = 0.0

    cv2.namedWindow("A.R.G.U.S. - Arduino Uno Live Screening", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("A.R.G.U.S. - Arduino Uno Live Screening", 1100, 750)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from webcam.")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10.0 / (time.time() - fps_start)
            fps_start = time.time()

        # Run inspection pipeline
        results = pipeline.process_frame(frame, use_deep_learning=True, use_classical_cv=True)
        pin_res = pin_analyzer.analyze_pins(results["aligned_image"])

        # Decide which view to display
        if view_mode == 0:
            display_img = results["annotated_frame"].copy()
        elif view_mode == 1:
            display_img = (
                results["gradcam_overlay"]
                if results["gradcam_overlay"] is not None
                else results["annotated_frame"].copy()
            )
        else:
            if results["diff_mask"] is not None:
                mask_bgr = cv2.cvtColor(results["diff_mask"], cv2.COLOR_GRAY2BGR)
                display_img = cv2.addWeighted(results["aligned_image"], 0.6, mask_bgr, 0.4, 0)
            else:
                display_img = results["annotated_frame"].copy()

        h, w = display_img.shape[:2]

        # Top Status HUD Banner
        is_pass = (results["final_status"] == "PASS")
        status_color = (30, 180, 50) if is_pass else (30, 40, 220)
        cv2.rectangle(display_img, (0, 0), (w, 65), (20, 20, 25), -1)
        cv2.rectangle(display_img, (0, 0), (12, 65), status_color, -1)

        # Verdict text
        cv2.putText(
            display_img,
            f"VERDICT: {results['final_status']}",
            (25, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            status_color,
            2,
            cv2.LINE_AA,
        )

        metrics_text = (
            f"SSIM: {results['ssim_score']:.3f} | "
            f"ML Defect: {results.get('defect_prob', 0.0)*100:.1f}% | "
            f"Anomalies: {results['defect_count']} | "
            f"Pins: {pin_res['total_pins']} | "
            f"FPS: {fps:.1f}"
        )
        cv2.putText(
            display_img,
            metrics_text,
            (25, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )

        # Bottom Instructions Bar
        cv2.rectangle(display_img, (0, h - 30), (w, h), (15, 15, 20), -1)
        info_text = f"View: [{view_names[view_mode]}] | [c] Switch View | [s] Save Snapshot & Cert | [q] Quit"
        cv2.putText(
            display_img,
            info_text,
            (15, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("A.R.G.U.S. - Arduino Uno Live Screening", display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            view_mode = (view_mode + 1) % 3
            print(f"Switched view to: {view_names[view_mode]}")
        elif key == ord("s"):
            cid = f"UNO_WEBCAM_{int(time.time())}"
            log_rec = logger.log_inspection(cid, results, annotated_frame=display_img)
            cert_path = reporter.generate_inspection_certificate(cid, results)
            print(f"[*] Snapshot & Certificate saved! Component ID: {cid}")
            print(f"    Certificate: {cert_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Live webcam screening session closed.")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_live_webcam(camera_index=idx)
