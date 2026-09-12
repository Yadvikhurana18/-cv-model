"""
Edge Device Client for Mothership ML Model Inference.

This module provides a client that edge devices (e.g., Raspberry Pi, Jetson Nano,
or embedded systems) can use to connect to the mothership server and receive
real-time ML model inference results.

Communication Protocol (JSON over WebSocket):
  Device -> Mothership:
    {
      "type": "frame",
      "device_id": "edge-001",
      "frame_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUQEhUVFhUVFhUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFR/2wBDAQkJDAsPD...",
      "timestamp": 1726531200.123
    }

  Mothership -> Device:
    {
      "type": "inference_result",
      "device_id": "edge-001",
      "status": "PASS",       // "PASS" or "FAIL"
      "confidence": 0.93,     // float confidence score
      "defect_prob": 0.07,    // probability of defective class
      "normal_prob": 0.93,    // probability of normal class
      "overlay_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUQEhUVFhUVFhUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFR/2wBDAQkJDAsPD...",
      "timestamp": 1726531200.456
    }
"""

import asyncio
import base64
import json
import time
import cv2
import numpy as np


class MothershipClient:
    """WebSocket client for edge device inference streaming."""
    
    def __init__(self, mothership_url: str = "ws://localhost:8000",
                 device_id: str = None):
        self.mothership_url = mothership_url
        self.device_id = device_id or f"edge-{int(time.time())}"
        self.websocket = None
    
    def _encode_frame_base64(self, frame: np.ndarray) -> str:
        """Encode numpy BGR frame to base64 JPEG string."""
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"[Client] Frame encode error: {e}")
            return ""
    
    def _decode_frame_base64(self, b64_str: str) -> np.ndarray:
        """Decode base64 JPEG frame to numpy BGR array."""
        try:
            data = base64.b64decode(b64_str)
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"[Client] Frame decode error: {e}")
            return None
    
    async def connect(self):
        """Connect to mothership WebSocket."""
        query = f"device_id={self.device_id}"
        full_url = f"{self.mothership_url}/ws/infer?{query}"
        self.websocket = await websockets.connect(full_url)
        print(f"[Client] Connected to mothership as {self.device_id}")
        return self.websocket
    
    async def disconnect(self):
        """Disconnect from mothership."""
        if self.websocket:
            await self.websocket.close()
            print(f"[Client] Disconnected from mothership")
    
    async def send_frame(self, frame: np.ndarray, timestamp: float = None):
        """Send a frame to mothership for inference.
        
        Returns inference result dict or None on error.
        """
        if not self.websocket:
            await self.connect()
        
        # Encode frame to base64
        frame_b64 = self._encode_frame_base64(frame)
        if not frame_b64:
            return None
        
        # Build and send message
        msg = json.dumps({
            "type": "frame",
            "device_id": self.device_id,
            "frame_base64": frame_b64,
            "timestamp": timestamp or time.time()
        })
        
        await self.websocket.send(msg)
        
        # Receive result
        try:
            response = await asyncio.wait_for(
                self.websocket.recv(), timeout=5.0
            )
            result = json.loads(response)
            
            if result.get("type") == "inference_result":
                return {
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "defect_prob": result["defect_prob"],
                    "normal_prob": result["normal_prob"],
                    "overlay_image_base64": result.get("overlay_image_base64", ""),
                    "device_id": result.get("device_id", self.device_id),
                    "timestamp": result.get("timestamp"),
                }
        except asyncio.TimeoutError:
            print("[Client] Timeout waiting for inference result")
        except json.JSONDecodeError as e:
            print(f"[Client] JSON decode error: {e}")
        except Exception as e:
            print(f"[Client] Error: {e}")
        
        return None
    
    async def continuous_stream(self, frame_callback, 
                                 pause_callback=None,
                                 stop_callback=None,
                                 max_frames: int = None):
        """Continuous frame streaming loop.
        
        Args:
            frame_callback: async function that returns (frame, timestamp) or None
            pause_callback: optional async function that returns True to pause
            stop_callback: optional async function that returns True to stop
            max_frames: maximum number of frames to process (None = unlimited)
        """
        frame_count = 0
        
        await self.connect()
        
        try:
            while True:
                # Check stop condition
                if stop_callback and await stop_callback():
                    print("[Client] Stop signal received")
                    break
                
                # Check pause condition
                if pause_callback and await pause_callback():
                    print("[Client] Paused - waiting...")
                    while pause_callback and not await pause_callback():
                        await asyncio.sleep(0.1)
                    print("[Client] Resumed")
                
                # Get frame
                result = await frame_callback()
                if result is None:
                    await asyncio.sleep(0.01)
                    continue
                
                frame, timestamp = result
                
                # Send for inference
                inference_result = await self.send_frame(frame, timestamp)
                if inference_result:
                    #print(f"[Client] Inference: {inference_result['status']} "
                    #      f"{inference_result['confidence']*100:.1f}%")
                    yield inference_result
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    print(f"[Client] Processed {max_frames} frames")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("[Client] Connection closed by mothership")
        except Exception as e:
            print(f"[Client] Error in continuous stream: {e}")
        finally:
            await self.disconnect()


async def capture_from_opencv(camera_index: int = 0):
    """Async generator that yields frames from OpenCV webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Client] Error: Could not open camera {camera_index}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Client] Failed to grab frame")
                await asyncio.sleep(0.1)
                continue
            
            yield (frame, time.time())
            await asyncio.sleep(0.03)  # ~30 FPS
    finally:
        cap.release()


async def main():
    """Example usage of MothershipClient."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Edge Device Mothership Client")
    parser.add_argument("--mothership", default="ws://localhost:8000",
                        help="Mothership WebSocket URL")
    parser.add_argument("--device-id", default=None,
                        help="Unique device identifier")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuous streaming mode")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (continuous mode)")
    
    args = parser.parse_args()
    
    client = MothershipClient(
        mothership_url=args.mothership,
        device_id=args.device_id
    )
    
    if args.continuous:
        print(f"[Client] Starting continuous streaming to {args.mothership}")
        print(f"[Client] Device ID: {client.device_id}")
        
        async def get_frame():
            return await capture_from_opencv(args.camera)
        
        async def should_stop():
            # Stop after max_frames or never
            return False
        
        async def should_pause():
            return False
        
        try:
            async for result in client.continuous_stream(
                frame_callback=get_frame,
                pause_callback=should_pause,
                stop_callback=should_stop,
                max_frames=args.max_frames
            ):
                status = result["status"]
                confidence = result["confidence"]
                print(f"\r[Client] Status: {status} | "
                      f"Confidence: {confidence*100:.1f}%   ", end="")
        except KeyboardInterrupt:
            print("\n[Client] Stopped by user")
    else:
        # Single frame mode - use webcam
        print(f"[Client] Single frame mode - Device: {client.device_id}")
        
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[Client] Error: Could not open camera {args.camera}")
            return
        
        ret, frame = cap.read()
        if not ret:
            print("[Client] Error: Could not grab frame")
            return
        
        cap.release()
        
        # Run inference
        result = await client.send_frame(frame)
        if result:
            print(f"\n[Client] Inference Result:")
            print(f"  Status: {result['status']}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            print(f"  Defect Prob: {result['defect_prob']*100:.1f}%")
            print(f"  Normal Prob: {result['normal_prob']*100:.1f}%")
            
            if result["overlay_image_base64"]:
                print(f"  Overlay image: base64 encoded ({len(result['overlay_image_base64'])} chars)")
        else:
            print("[Client] No inference result received")


if __name__ == "__main__":
    asyncio.run(main())