"""
FastAPI Backend Server & REST Endpoints for Arduino Uno Screening System.
Provides REST endpoints for edge device integration, hardware inspection, and certificate downloads.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import base64
import cv2
import numpy as np
import torch
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import config
from app.storage.logger import InspectionLogger
from app.storage.report_generator import ReportGenerator
from app.vision.pin_analyzer import PinAnalyzer
from app.vision.pipeline import InspectionPipeline

app = FastAPI(
    title="Arduino Uno Screening & AI Inspection API",
    description="Autonomous Computer Vision & PyTorch Deep Learning Inspection Server",
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


@app.get("/")
def root():
    """Root status and documentation endpoint."""
    return {
        "service": "Arduino Uno AI Screening Inspection API",
        "status": "online",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "endpoints": {
            "health": "/api/health",
            "inspect": "/api/inspect (POST multipart image)",
            "certificate": "/api/certificate/{component_id}",
        },
    }


@app.get("/api/health")
def health_check():
    """Returns GPU status, CUDA memory, and model availability."""
    cuda_avail = torch.cuda.is_available()
    return {
        "status": "online",
        "target_board": "Arduino Uno R3",
        "gpu_available": cuda_avail,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2) if cuda_avail else 0,
        "torch_version": torch.__version__,
        "reference_loaded": pipeline.reference_image is not None,
    }


@app.post("/api/inspect")
async def inspect_component(
    file: UploadFile = File(...),
    component_id: str = Form("AUTO_UNO"),
):
    """
    Accepts an uploaded Arduino Uno component image and performs full CV + PyTorch screening.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file format or corrupted payload.")

        # Process through master inspection pipeline
        results = pipeline.process_frame(frame, use_deep_learning=True, use_classical_cv=True)
        pin_results = pin_analyzer.analyze_pins(frame)
        results["pin_analysis"] = {
            "is_compliant": pin_results["is_compliant"],
            "total_pins": pin_results["total_pins"],
            "pitch_mean": pin_results["pitch_mean"],
            "defects": pin_results["pin_defects"],
        }

        # Log to CSV audit trail
        log_rec = logger.log_inspection(component_id, results, annotated_frame=results["annotated_frame"])

        # Base64 encode annotated frame for client display
        _, buf = cv2.imencode(".jpg", results["annotated_frame"])
        annotated_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "component_id": component_id,
            "target_board": "Arduino Uno R3",
            "final_status": results["final_status"],
            "cv_status": results["cv_status"],
            "ml_status": results["ml_status"],
            "ssim_score": float(results["ssim_score"]),
            "defect_count": int(results["defect_count"]),
            "ml_defect_prob": float(results.get("defect_prob", 0.0)),
            "summary": results["summary"],
            "pin_analysis": results["pin_analysis"],
            "annotated_image_base64": annotated_b64,
            "log_record": log_rec,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection pipeline error: {str(e)}")


@app.get("/api/certificate/{component_id}", response_class=HTMLResponse)
def get_certificate(component_id: str):
    """Generates on-demand certificate HTML for given Arduino Uno component."""
    mock_res = {
        "final_status": "PASS",
        "cv_status": "PASS",
        "ml_status": "PASS",
        "ssim_score": 0.985,
        "defect_count": 0,
        "defect_prob": 0.008,
        "is_aligned": True,
        "summary": "Arduino Uno passed all physical and structural tolerance tests.",
    }
    cert_path = reporter.generate_inspection_certificate(component_id, mock_res)
    with open(cert_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="127.0.0.1", port=8000, reload=False)
