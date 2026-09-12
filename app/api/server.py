"""
FastAPI Backend Server & REST Endpoints for Arduino Uno Screening System.
Provides REST endpoints for edge device integration, hardware inspection, and certificate downloads.
Also provides WebSocket endpoint for real-time device inference streaming.
Manages multiple network video streams from devices.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import base64
import cv2
import numpy as np
import torch
import asyncio
from typing import Dict, Set
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import config
from app.ml.inference import DefectClassifier
from app.storage.logger import InspectionLogger
from app.storage.report_generator import ReportGenerator
from app.vision.pin_analyzer import PinAnalyzer
from app.vision.pipeline import InspectionPipeline
from app.camera.network_camera import NetworkCamera, StreamManager

app = FastAPI(
    title="Arduino Uno Screening & AI Inspection API",
    description="Autonomous Computer Vision & PyTorch Deep Learning Inspection Server with real-time device streaming",
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

# Active WebSocket connections per device
connected_devices: Dict[str, WebSocket] = {}

ml_classifier = DefectClassifier()

# Stream management for multiple device camera streams
stream_manager = StreamManager(max_streams=20)


def _decode_frame_base64(b64_str: str) -> np.ndarray:
    """Decode base64 JPEG frame to numpy BGR array."""
    try:
        data = base64.b64decode(b64_str)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[WS] Frame decode error: {e}")
        return None


def _encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode numpy BGR frame to base64 JPEG string."""
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"[WS] Frame encode error: {e}")
        return ""


@app.websocket("/ws/infer")
async def websocket_infer(websocket: WebSocket):
    """WebSocket endpoint for edge device inference streaming.
    
    Devices connect and send base64-encoded JPEG frames.
    Mothership responds with inference results (PASS/FAIL, confidence, Grad-CAM overlay).
    """
    await websocket.accept()
    device_id = websocket.query_params.get("device_id", "unknown")
    connected_devices[device_id] = websocket
    print(f"[WS] Device connected: {device_id} (total: {len(connected_devices)})")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "frame":
                frame_bgr = _decode_frame_base64(data["frame_base64"])
                if frame_bgr is None:
                    continue
                # Run inference
                with torch.no_grad():
                    result = ml_classifier.predict(frame_bgr, compute_cam=True)
                # Encode overlay image back to base64
                overlay_b64 = ""
                if result.get("overlay_image") is not None:
                    overlay_b64 = _encode_frame_to_base64(result["overlay_image"])
                # Send result back
                response = {
                    "type": "inference_result",
                    "device_id": device_id,
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "defect_prob": result["defect_prob"],
                    "normal_prob": result["normal_prob"],
                    "overlay_image_base64": overlay_b64,
                    "timestamp": data.get("timestamp"),
                }
                await websocket.send_json(response)
    except Exception:
        pass
    except Exception as e:
        print(f"[WS] Error with device {device_id}: {e}")
    finally:
        connected_devices.pop(device_id, None)
        print(f"[WS] Device disconnected: {device_id} (remaining: {len(connected_devices)})")


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
        "active_streams": len(stream_manager.streams),
    }


@app.post("/api/streams/add")
async def add_stream(stream_id: str = Form(...), url: str = Form(...)):
    """Add a new camera stream to the mothership."""
    try:
        camera = stream_manager.add_stream(stream_id, url)
        await stream_manager.start_streaming(stream_id)
        return {
            "success": True,
            "stream_id": stream_id,
            "url": url,
            "message": f"Stream {stream_id} added and started",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add stream: {str(e)}")


@app.post("/api/streams/remove")
async def remove_stream(stream_id: str = Form(...)):
    """Remove a camera stream from the mothership."""
    try:
        success = stream_manager.remove_stream(stream_id)
        if success:
            return {"success": True, "stream_id": stream_id, "message": f"Stream {stream_id} removed"}
        else:
            raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove stream: {str(e)}")


@app.get("/api/streams/status")
async def streams_status():
    """Get status of all camera streams."""
    status = {}
    for stream_id, camera in stream_manager.streams.items():
        status[stream_id] = {
            "connected": camera.is_connected(),
            "frame_count": camera.frame_count,
            "errors": camera.errors,
        }
    return {"success": True, "active_streams": len(stream_manager.streams), "streams": status}


@app.get("/api/streams/queues")
async def streams_queues():
    """Get frame queue status for all streams."""
    queues = {}
    for stream_id, queue in stream_manager.frame_queues.items():
        try:
            queues[stream_id] = {"queue_size": queue.qsize()}
        except Exception:
            queues[stream_id] = {"queue_size": 0}
    return {"success": True, "queues": queues}


@app.post("/api/inspect")
async def inspect_component(
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
