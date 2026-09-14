"""
Network / IP Camera Source for Raspberry Pi + Arducam Streams (RTSP / HTTP / MJPEG).
Supports continuous frame capture with auto-reconnect and frame metadata.
"""

import asyncio
import time
import cv2
import numpy as np
from pathlib import Path
from app.camera.base_camera import BaseCamera


class NetworkCamera(BaseCamera):
    """Captures stream from RTSP, MJPEG, or HTTP IP cameras."""

    def __init__(self, stream_url: str, reconnect_timeout: float = 3.0,
                 buffer_size: int = 10):
        self.stream_url = stream_url
        self.reconnect_timeout = reconnect_timeout
        self.buffer_size = buffer_size
        self.cap: cv2.VideoCapture | None = None
        self.last_connect_time: float = 0.0
        self.frame_count: int = 0
        self.errors: int = 0
        self._frame_queue: "asyncio.Queue" | None = None

    def connect(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True

        curr_time = time.time()
        if curr_time - self.last_connect_time < self.reconnect_timeout:
            return False

        self.last_connect_time = curr_time
        self.cap = cv2.VideoCapture(self.stream_url)
        
        # Attempt to set buffer size for RTSP optimization
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except Exception:
            pass

        return self.cap.isOpened()

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return False, None

        ret, frame = self.cap.read()
        if not ret:
            self.errors += 1
            self.release()
            return False, None
        self.frame_count += 1
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_count = 0
        self.errors = 0

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def set_frame_queue(self, queue: "asyncio.Queue") -> None:
        """Set async queue for streaming pipeline integration."""
        self._frame_queue = queue

    async def run_capture_loop(self):
        """Continuous coroutine that captures frames and pushes them to the queue."""
        while True:
            ret, frame = self.get_frame()
            if ret and frame is not None:
                timestamp = time.time()
                if self._frame_queue is not None:
                    try:
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except Exception:
                                pass
                        await self._frame_queue.put((timestamp, frame))
                    except Exception:
                        pass
            await asyncio.sleep(0.033)  # ~30 FPS

    async def stream_frames(self):
        """Async generator yielding (timestamp, frame) tuples."""
        while True:
            ret, frame = self.get_frame()
            if ret and frame is not None:
                timestamp = time.time()
                if self._frame_queue is not None:
                    try:
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except Exception:
                                pass
                        await self._frame_queue.put((timestamp, frame))
                    except Exception:
                        pass
                yield timestamp, frame
            else:
                await asyncio.sleep(0.033)  # ~30 FPS


class StreamManager:
    """Manages multiple network camera streams for the mothership."""
    
    def __init__(self, max_streams: int = 10):
        self.max_streams = max_streams
        self.streams: dict[str, NetworkCamera] = {}
        self.frame_queues: dict[str, asyncio.Queue] = {}
        self._tasks: dict[str, asyncio.Task] = {}
    
    def add_stream(self, stream_id: str, url: str) -> NetworkCamera:
        """Add a new camera stream."""
        if stream_id in self.streams:
            print(f"[Stream] Stream {stream_id} already exists, replacing...")
            self.remove_stream(stream_id)
        
        camera = NetworkCamera(url)
        self.streams[stream_id] = camera
        
        # Create frame queue for this stream
        queue = asyncio.Queue(maxsize=self.streams[stream_id].buffer_size)
        self.frame_queues[stream_id] = queue
        camera.set_frame_queue(queue)
        
        print(f"[Stream] Added stream '{stream_id}': {url}")
        return camera
    
    def remove_stream(self, stream_id: str) -> bool:
        """Remove a camera stream."""
        if stream_id not in self.streams:
            return False
        
        camera = self.streams.pop(stream_id)
        camera.release()
        
        if stream_id in self.frame_queues:
            self.frame_queues.pop(stream_id)
        
        # Cancel any running task
        if stream_id in self._tasks:
            self._tasks[stream_id].cancel()
            self._tasks.pop(stream_id)
        
        print(f"[Stream] Removed stream '{stream_id}'")
        return True
    
    async def start_streaming(self, stream_id: str) -> asyncio.Task:
        """Start continuous frame capture task for a stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream '{stream_id}' not found")
        
        if stream_id in self._tasks:
            self._tasks[stream_id].cancel()
        
        camera = self.streams[stream_id]
        
        task = asyncio.create_task(
            camera.run_capture_loop(), name=f"stream-{stream_id}"
        )
        self._tasks[stream_id] = task
        
        print(f"[Stream] Started streaming for '{stream_id}'")
        return task
    
    async def stop_all(self) -> None:
        """Stop all streaming tasks."""
        for stream_id in list(self._tasks.keys()):
            self._tasks[stream_id].cancel()
        self._tasks.clear()
        
        for camera in self.streams.values():
            camera.release()
        self.streams.clear()
        self.frame_queues.clear()
        
        print("[Stream] Stopped all streams")
