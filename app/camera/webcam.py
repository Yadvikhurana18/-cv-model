"""
Webcam Camera Source Implementation using OpenCV VideoCapture.
"""
import cv2
import numpy as np
from app.camera.base_camera import BaseCamera


class WebcamCamera(BaseCamera):
    """Local USB or integrated camera capture."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

    def connect(self) -> bool:
        """Opens camera connection and sets resolution."""
        if self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        """Reads a frame from the webcam."""
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        """Releases the camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
