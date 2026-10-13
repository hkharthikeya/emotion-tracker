"""
EmotiCam Face Detection Subsystem.
Implements robust, real-time multi-face detection using optimized OpenCV cascades with CLAHE contrast enhancement.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Real-time Face Detector using OpenCV Haar Cascades with adaptive contrast equalization.
    """
    def __init__(self, min_face_size: int = 50, scale_factor: float = 1.15, min_neighbors: int = 5):
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
        # Load default OpenCV frontal face cascade
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade XML not found at {cascade_path}")
            
        self.classifier = cv2.CascadeClassifier(cascade_path)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for ESP32-CAM lighting variations
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        logger.info("FaceDetector initialized successfully with Haar Cascade.")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in a BGR image.
        Returns: List of (x, y, w, h) bounding boxes.
        """
        if image is None or image.size == 0:
            return []
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply CLAHE to handle harsh shadows or low-light conditions from ESP32-CAM
        enhanced_gray = self.clahe.apply(gray)
        
        # Detect multi-scale faces
        faces = self.classifier.detectMultiScale(
            enhanced_gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return []
            
        # Convert numpy array to list of tuples: (x, y, w, h)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
