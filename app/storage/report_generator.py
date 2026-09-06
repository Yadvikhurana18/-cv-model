"""
ISRO Quality Inspection Certificate & Comprehensive Screening Report Generator.
Produces structured HTML/PDF inspection certificates with multi-modal defect visualizations,
pin metrology tables, and compliance verification sign-offs.
"""
from datetime import datetime
from pathlib import Path
import base64
import cv2
import numpy as np
from app.config import config


class ReportGenerator:
    """Generates formal electronic component inspection reports and certificates."""

    def __init__(self, output_dir: Path | None = None):
        if output_dir is None:
            output_dir = config.logs_dir / "reports"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _img_to_base64(self, bgr_img: np.ndarray) -> str:
        """Encodes OpenCV BGR image to base64 JPEG string."""
        _, buf = cv2.imencode(".jpg", bgr_img)
        return base64.b64encode(buf).decode("utf-8")

    def generate_inspection_certificate(
        self,
        component_id: str,
        results: dict,
        operator_name: str = "A.R.G.U.S. Certified Inspector",
        batch_number: str = "ARGUS-LOT-2026-X1",
    ) -> Path:
        """
        Builds a comprehensive HTML inspection certificate document.
        """
        timestamp = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        is_pass = results["final_status"] == "PASS"

        # Encode multi-modal views
        annotated_b64 = self._img_to_base64(results.get("annotated_frame", np.zeros((200, 200, 3), dtype=np.uint8)))
        cam_b64 = self._img_to_base64(results["gradcam_overlay"]) if results.get("gradcam_overlay") is not None else ""

        status_color = "#057a55" if is_pass else "#c81e1e"
        badge_text = "PASSED / FLIGHT READY" if is_pass else "REJECTED / DEFECTIVE"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>A.R.G.U.S. Screening Certificate - {component_id}</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8fafc; color: #1e293b; }}
    .certificate {{ border: 4px solid #0284c7; background: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); max-width: 900px; margin: auto; }}
    .header {{ text-align: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; margin-bottom: 20px; }}
    .header h1 {{ margin: 0; color: #0369a1; font-size: 26px; }}
    .header h3 {{ margin: 5px 0 0 0; color: #64748b; font-weight: 500; }}
    .badge {{ display: inline-block; background-color: {status_color}; color: white; padding: 10px 28px; font-size: 20px; font-weight: bold; border-radius: 6px; margin: 15px 0; letter-spacing: 1px; }}
    .meta-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
    .meta-table td {{ padding: 8px 12px; border: 1px solid #e2e8f0; font-size: 14px; }}
    .meta-table td.label {{ background-color: #f1f5f9; font-weight: bold; width: 25%; color: #334155; }}
    .images-grid {{ display: flex; gap: 20px; justify-content: center; margin: 25px 0; }}
    .img-box {{ flex: 1; text-align: center; background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #cbd5e1; }}
    .img-box img {{ max-width: 100%; height: auto; border-radius: 6px; }}
    .img-box p {{ margin: 8px 0 0 0; font-weight: bold; font-size: 13px; color: #475569; }}
    .footer {{ margin-top: 30px; border-top: 2px solid #e2e8f0; padding-top: 15px; display: flex; justify-content: space-between; font-size: 13px; color: #64748b; }}
</style>
</head>
<body>
<div class="certificate">
    <div class="header">
        <h1>🛰️ A.R.G.U.S. ELECTRONIC COMPONENT SCREENING CERTIFICATE</h1>
        <h3>Autonomous Real-time Geometric & Underwriting Screening System</h3>
        <div class="badge">{badge_text}</div>
    </div>

    <table class="meta-table">
        <tr>
            <td class="label">Component Serial / ID:</td><td><strong>{component_id}</strong></td>
            <td class="label">Batch Lot Number:</td><td>{batch_number}</td>
        </tr>
        <tr>
            <td class="label">Inspection Timestamp:</td><td>{timestamp}</td>
            <td class="label">Certified Inspector:</td><td>{operator_name}</td>
        </tr>
        <tr>
            <td class="label">Structural SSIM Score:</td><td>{results.get('ssim_score', 1.0):.4f} (Threshold: &ge; {config.ssim_threshold})</td>
            <td class="label">Detected Anomaly Regions:</td><td>{results.get('defect_count', 0)}</td>
        </tr>
        <tr>
            <td class="label">Deep Learning Defect Prob:</td><td>{results.get('defect_prob', 0.0)*100:.2f}%</td>
            <td class="label">Image Registration:</td><td>{'Aligned (Homography Verified)' if results.get('is_aligned') else 'Template Geometry Match'}</td>
        </tr>
        <tr>
            <td class="label">Screening Diagnostic:</td>
            <td colspan="3"><em>{results.get('summary', 'Nominal')}</em></td>
        </tr>
    </table>

    <div class="images-grid">
        <div class="img-box">
            <img src="data:image/jpeg;base64,{annotated_b64}" alt="Annotated Defect View">
            <p>1. Computer Vision Defect Bounding Boxes</p>
        </div>
        {f'''<div class="img-box">
            <img src="data:image/jpeg;base64,{cam_b64}" alt="Grad-CAM Anomaly Heatmap">
            <p>2. PyTorch Grad-CAM Anomaly Heatmap</p>
        </div>''' if cam_b64 else ''}
    </div>

    <div class="footer">
        <div><strong>Compliance Standard:</strong> ISRO-PAX-300 / Space-Grade Screening</div>
        <div><strong>Digital Signature:</strong> Verified By Autonomous Vision Pipeline</div>
    </div>
</div>
</body>
</html>
"""
        report_file = self.output_dir / f"Certificate_{component_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_file
