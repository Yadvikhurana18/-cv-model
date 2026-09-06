"""
FastAPI Backend Server & WebSocket Stream for ISRO Component Screening System.
Provides REST endpoints for edge device integration, hardware inspection, and report downloads.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.config import config
from app.storage.logger import InspectionLogger
from app.storage.report_generator import ReportGenerator
from app.vision.pin_analyzer import PinAnalyzer
from app.vision.pipeline import InspectionPipeline

app = FastAPI(
    title="ISRO Electronic Component Screening API",
    description="Autonomous Computer Vision & Deep Learning Inspection Server",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = InspectionPipeline()
logger = InspectionLogger()
reporter = ReportGenerator()
pin_analyzer = PinAnalyzer()


@app.get("/api/health")
def health_check():
    """Returns GPU status, CUDA memory, and model availability."""
    cuda_avail = torch.cuda.is_available()
    return {
        "status": "online",
        "gpu_available": cuda_avail,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2) if cuda_avail else 0,
        "torch_version": torch.__version__,
        "reference_loaded": pipeline.reference_image is not None,
    }


@app.post("/api/inspect")
async def inspect_component(file: UploadFile = File(...), component_id: str = "AUTO_IC"):
    """
    Accepts an uploaded component image file and performs full CV + Deep Learning screening.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # Process through pipeline
    results = pipeline.process_frame(frame, use_deep_learning=True, use_classical_cv=True)
    pin_results = pin_analyzer.analyze_pins(frame)
    results["pin_analysis"] = {
        "is_compliant": pin_results["is_compliant"],
        "total_pins": pin_results["total_pins"],
        "pitch_mean": pin_results["pitch_mean"],
        "defects": pin_results["pin_defects"],
    }

    # Log to CSV
    log_rec = logger.log_inspection(component_id, results, annotated_frame=results["annotated_frame"])

    # Base64 encode annotated frame for client display
    _, buf = cv2.imencode(".jpg", results["annotated_frame"])
    annotated_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "component_id": component_id,
        "final_status": results["final_status"],
        "cv_status": results["cv_status"],
        "ml_status": results["ml_status"],
        "ssim_score": results["ssim_score"],
        "defect_count": results["defect_count"],
        "ml_defect_prob": results.get("defect_prob", 0.0),
        "summary": results["summary"],
        "pin_analysis": results["pin_analysis"],
        "annotated_image_base64": annotated_b64,
        "log_record": log_rec,
    }


@app.get("/api/certificate/{component_id}", response_class=HTMLResponse)
def get_certificate(component_id: str):
    """Generates on-demand certificate HTML for given component."""
    # Synthetic / sample certificate view
    mock_res = {
        "final_status": "PASS",
        "cv_status": "PASS",
        "ml_status": "PASS",
        "ssim_score": 0.962,
        "defect_count": 0,
        "defect_prob": 0.012,
        "is_aligned": True,
        "summary": "Passed all physical and structural tolerance tests.",
    }
    cert_path = reporter.generate_inspection_certificate(component_id, mock_res)
    with open(cert_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="127.0.0.1", port=8000, reload=False)
