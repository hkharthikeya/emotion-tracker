"""
EmotiCam End-to-End Vision Pipeline Orchestrator.
Coordinates frame acquisition, face detection, multi-face tracking, emotion inference,
HUD rendering, FPS/latency measurement, and database persistence.
"""

import cv2
import numpy as np
import time
import os
import logging
from typing import Dict, Any, Optional, Tuple
from django.conf import settings

from .camera import ESP32Camera
from .face_detector import FaceDetector
from .face_tracker import CentroidTracker
from .emotion_detector import EmotionDetector
from .utils import (
    draw_hud_box,
    draw_no_face_hud,
    encode_image_to_base64_jpeg,
    encode_image_to_jpeg_bytes,
    EMOTION_EMOJIS
)

logger = logging.getLogger(__name__)

class StreamProcessor:
    """
    Main Real-Time Vision & Emotion Analysis Pipeline.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            camera_url = getattr(settings, 'ESP32_CAMERA_URL', 'http://192.168.1.100')
            model_path = getattr(settings, 'EMOTION_MODEL_PATH', 'models/emotion-ferplus-8.onnx')
            cls._instance = cls(camera_url=camera_url, model_path=model_path)
        return cls._instance

    def __init__(self, camera_url: str = "http://192.168.1.100", model_path: str = "models/emotion-ferplus-8.onnx"):
        self.camera_url = camera_url
        self.model_path = model_path
        
        # Initialize Subsystems
        self.camera = ESP32Camera(base_url=camera_url)
        self.detector = FaceDetector(
            min_face_size=getattr(settings, 'FACE_DETECTION_MIN_SIZE', 50)
        )
        self.tracker = CentroidTracker(max_disappeared=10, max_distance=90.0)
        self.emotion_detector = EmotionDetector(model_path=model_path)
        
        # Pipeline State & Performance Metrics
        self.is_active = True
        self.fps = 0.0
        self.latency_ms = 0.0
        self._prev_frame_time = time.perf_counter()
        self._fps_smoothing = 0.85
        
        # Start camera grabber worker
        target_fps = getattr(settings, 'CAMERA_FPS', 15)
        self.camera.start(target_fps=target_fps)
        logger.info("StreamProcessor initialized and active.")

    def set_camera_url(self, new_url: str):
        """Update ESP32-CAM target IP."""
        self.camera_url = new_url
        self.camera.set_camera_url(new_url)
        self.tracker.reset()

    def process_raw_frame(self, raw_frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full AI pipeline on a caller-supplied BGR frame (e.g. from browser webcam).
        Skips the camera grab step entirely.
        """
        t_start = time.perf_counter()

        processed_frame = raw_frame.copy()
        h_img, w_img = processed_frame.shape[:2]

        # Face Detection
        raw_bboxes = self.detector.detect(processed_frame)

        # Multi-Face Tracking
        tracked_faces = self.tracker.update(raw_bboxes)

        # Clean up stale track history
        self.emotion_detector.clear_track_history(list(tracked_faces.keys()))

        faces_data = []
        dominant_session_emotion = "Neutral"
        highest_confidence = 0.0
        overall_distribution = {e: 0.0 for e in EmotionDetector.EMOTION_CLASSES}

        if len(tracked_faces) == 0:
            draw_no_face_hud(processed_frame)
        else:
            for track_id, (x, y, w, h) in tracked_faces.items():
                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)
                x1_pad = max(0, x - pad_w)
                y1_pad = max(0, y - pad_h)
                x2_pad = min(w_img, x + w + pad_w)
                y2_pad = min(h_img, y + h + pad_h)
                face_crop = raw_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if face_crop is None or face_crop.size == 0:
                    face_crop = raw_frame[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
                emotion, confidence, dist = self.emotion_detector.predict(face_crop, track_id=track_id)
                draw_hud_box(processed_frame, (x, y, w, h), track_id=track_id, emotion=emotion, confidence=confidence)
                faces_data.append({
                    "track_id": track_id,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "emotion": emotion,
                    "emoji": EMOTION_EMOJIS.get(emotion, "❓"),
                    "confidence": confidence,
                    "distribution": dist
                })
                if confidence > highest_confidence or len(faces_data) == 1:
                    highest_confidence = confidence
                    dominant_session_emotion = emotion
                    overall_distribution = dist

        # FPS Calculation
        now = time.perf_counter()
        delta = now - self._prev_frame_time
        self._prev_frame_time = now
        instant_fps = (1.0 / delta) if delta > 0 else 0.0
        self.fps = (self._fps_smoothing * self.fps) + ((1.0 - self._fps_smoothing) * instant_fps) if self.fps > 0 else instant_fps

        total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        self.latency_ms = total_latency_ms

        quality = getattr(settings, 'JPEG_QUALITY', 80)
        base64_image = encode_image_to_base64_jpeg(processed_frame, quality=quality)

        return {
            "type": "frame",
            "image": base64_image,
            "faces": faces_data,
            "primary_emotion": dominant_session_emotion if faces_data else "No Face",
            "primary_emoji": EMOTION_EMOJIS.get(dominant_session_emotion, "❓") if faces_data else "🔍",
            "confidence": highest_confidence if faces_data else 0.0,
            "face_count": len(faces_data),
            "fps": round(self.fps, 1),
            "latency_ms": total_latency_ms,
            "camera_status": "ONLINE",
            "ai_status": "PROCESSING",
            "error": None,
            "distribution": overall_distribution if faces_data else {e: 0.0 for e in EmotionDetector.EMOTION_CLASSES},
            "timestamp": time.strftime("%H:%M:%S")
        }

    def create_offline_placeholder(self, width: int = 640, height: int = 480, message: str = "CAMERA OFFLINE") -> np.ndarray:
        """Generates an aesthetic dark standby canvas with connection instructions."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        for y in range(height):
            img[y, :, :] = (15 + int(y / height * 10), 20 + int(y / height * 10), 28 + int(y / height * 15))
        
        # Outer dashed/glow border
        cv2.rectangle(img, (20, 20), (width - 20, height - 20), (45, 55, 72), 1)
        
        # Status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "EMOTICAM - ESP32-CAM FEED"
        (tw1, th1), _ = cv2.getTextSize(title, font, 0.6, 1)
        cv2.putText(img, title, ((width - tw1) // 2, height // 2 - 50), font, 0.6, (148, 163, 184), 1, cv2.LINE_AA)
        
        (tw2, th2), _ = cv2.getTextSize(message, font, 0.8, 2)
        cv2.putText(img, message, ((width - tw2) // 2, height // 2 - 10), font, 0.8, (239, 68, 68), 2, cv2.LINE_AA)
        
        hint = f"Target IP: {self.camera_url}"
        (tw3, th3), _ = cv2.getTextSize(hint, font, 0.5, 1)
        cv2.putText(img, hint, ((width - tw3) // 2, height // 2 + 25), font, 0.5, (100, 116, 139), 1, cv2.LINE_AA)
        
        action = "Check Wi-Fi connection and ESP32 power."
        (tw4, th4), _ = cv2.getTextSize(action, font, 0.45, 1)
        cv2.putText(img, action, ((width - tw4) // 2, height // 2 + 55), font, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        
        return img

    def process_frame(self) -> Dict[str, Any]:
        """
        Executes the entire end-to-end vision pipeline:
          1. Grab latest camera frame
          2. Measure real processing latency
          3. Detect and Track faces
          4. Crop and classify emotions
          5. Render HUD bounding boxes and annotations
          6. Compute FPS and assemble comprehensive telemetry payload
        """
        t_start = time.perf_counter()
        
        # 1. Fetch raw frame from ESP32 camera client
        raw_frame, fetch_latency_ms, camera_status, err_msg = self.camera.get_latest_frame()
        
        # Fallback if camera is offline or no frame yet
        if raw_frame is None or camera_status == "OFFLINE":
            placeholder = self.create_offline_placeholder(
                message="CAMERA OFFLINE" if camera_status == "OFFLINE" else "CONNECTING TO ESP32..."
            )
            base64_img = encode_image_to_base64_jpeg(placeholder, quality=75)
            
            return {
                "type": "frame",
                "image": base64_img,
                "faces": [],
                "primary_emotion": "Unknown",
                "primary_emoji": EMOTION_EMOJIS["Unknown"],
                "confidence": 0.0,
                "face_count": 0,
                "fps": 0.0,
                "latency_ms": round(fetch_latency_ms, 1),
                "camera_status": camera_status,
                "ai_status": "STANDBY",
                "error": err_msg or "No camera stream available",
                "distribution": {e: 0.0 for e in EmotionDetector.EMOTION_CLASSES},
                "timestamp": time.strftime("%H:%M:%S")
            }

        # Frame exists -> Proceed with AI Pipeline
        processed_frame = raw_frame.copy()
        h_img, w_img = processed_frame.shape[:2]
        
        # 2. Face Detection
        raw_bboxes = self.detector.detect(processed_frame)
        
        # 3. Multi-Face Tracking
        tracked_faces = self.tracker.update(raw_bboxes)
        
        # 4. Clean up emotion history for deregistered tracks
        self.emotion_detector.clear_track_history(list(tracked_faces.keys()))
        
        faces_data = []
        dominant_session_emotion = "Neutral"
        highest_confidence = 0.0
        overall_distribution = {e: 0.0 for e in EmotionDetector.EMOTION_CLASSES}

        if len(tracked_faces) == 0:
            draw_no_face_hud(processed_frame)
        else:
            # 5. Process each tracked face
            for track_id, (x, y, w, h) in tracked_faces.items():
                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)
                x1_pad = max(0, x - pad_w)
                y1_pad = max(0, y - pad_h)
                x2_pad = min(w_img, x + w + pad_w)
                y2_pad = min(h_img, y + h + pad_h)
                face_crop = raw_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if face_crop is None or face_crop.size == 0:
                    face_crop = raw_frame[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
                
                # Predict emotion
                emotion, confidence, dist = self.emotion_detector.predict(face_crop, track_id=track_id)
                
                # Draw on-frame futuristic HUD
                draw_hud_box(
                    processed_frame,
                    (x, y, w, h),
                    track_id=track_id,
                    emotion=emotion,
                    confidence=confidence
                )
                
                faces_data.append({
                    "track_id": track_id,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "emotion": emotion,
                    "emoji": EMOTION_EMOJIS.get(emotion, "❓"),
                    "confidence": confidence,
                    "distribution": dist
                })

                # Determine primary face (highest confidence)
                if confidence > highest_confidence or len(faces_data) == 1:
                    highest_confidence = confidence
                    dominant_session_emotion = emotion
                    overall_distribution = dist

        # 6. FPS Calculation
        now = time.perf_counter()
        delta = now - self._prev_frame_time
        self._prev_frame_time = now
        instant_fps = (1.0 / delta) if delta > 0 else 0.0
        self.fps = (self._fps_smoothing * self.fps) + ((1.0 - self._fps_smoothing) * instant_fps) if self.fps > 0 else instant_fps

        # 7. Total End-to-End Latency Calculation
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0 + fetch_latency_ms
        self.latency_ms = round(total_latency_ms, 1)

        # 8. Encode processed frame to Base64 JPEG
        quality = getattr(settings, 'JPEG_QUALITY', 80)
        base64_image = encode_image_to_base64_jpeg(processed_frame, quality=quality)

        return {
            "type": "frame",
            "image": base64_image,
            "faces": faces_data,
            "primary_emotion": dominant_session_emotion if len(faces_data) > 0 else "No Face",
            "primary_emoji": EMOTION_EMOJIS.get(dominant_session_emotion, "❓") if len(faces_data) > 0 else "🔍",
            "confidence": highest_confidence if len(faces_data) > 0 else 0.0,
            "face_count": len(faces_data),
            "fps": round(self.fps, 1),
            "latency_ms": self.latency_ms,
            "camera_status": camera_status,
            "ai_status": "PROCESSING",
            "error": None,
            "distribution": overall_distribution if len(faces_data) > 0 else {e: 0.0 for e in EmotionDetector.EMOTION_CLASSES},
            "timestamp": time.strftime("%H:%M:%S")
        }

    def get_raw_jpeg_stream(self):
        """Generator for MJPEG HTTP streaming endpoint."""
        while self.is_active:
            payload = self.process_frame()
            # If we need raw JPEG bytes:
            raw_frame, _, _, _ = self.camera.get_latest_frame()
            if raw_frame is not None:
                frame_to_send = raw_frame
            else:
                frame_to_send = self.create_offline_placeholder()
                
            jpeg_bytes = encode_image_to_jpeg_bytes(frame_to_send, quality=75)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            time.sleep(0.04)
