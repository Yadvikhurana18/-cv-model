"""
Camera Module
Provides a uniform abstraction across Webcam and Network (Raspberry Pi + Arducam) streams.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseCamera(ABC):
    """Abstract Base Class for all camera sources."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Initialize and connect to the camera source."""
        pass
    
    @abstractmethod
    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        """Return (success, frame) where frame is a standard BGR OpenCV numpy array."""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if camera is currently connected and active."""
        pass
