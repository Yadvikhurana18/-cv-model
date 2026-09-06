"""
Network / IP Camera Source for Raspberry Pi + Arducam Streams (RTSP / HTTP / MJPEG).
"""
import time
import cv2
import numpy as np
from app.camera.base_camera import BaseCamera


class NetworkCamera(BaseCamera):
    """Captures stream from RTSP, MJPEG, or HTTP IP cameras."""

    def __init__(self, stream_url: str, reconnect_timeout: float = 3.0):
        self.stream_url = stream_url
        self.reconnect_timeout = reconnect_timeout
        self.cap: cv2.VideoCapture | None = None
        self.last_connect_time: float = 0.0

    def connect(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True

        curr_time = time.time()
        if curr_time - self.last_connect_time < self.reconnect_timeout:
            return False

        self.last_connect_time = curr_time
        self.cap = cv2.VideoCapture(self.stream_url)
        return self.cap.isOpened()

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return False, None

        ret, frame = self.cap.read()
        if not ret:
            self.release()
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
