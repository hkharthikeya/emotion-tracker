"""
EmotiCam Facial Emotion Recognition Subsystem.
Executes lightweight Deep CNN inference via OpenCV DNN ONNX Engine with temporal smoothing.
"""

import cv2
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

class EmotionDetector:
    """
    Real-Time Deep Learning Emotion Classifier using OpenCV DNN.
    Evaluates 7 standardized facial emotion classes:
      - Happy
      - Neutral
      - Sad
      - Angry
      - Surprise
      - Fear
      - Disgust
    """
    # Canonical emotion label set
    EMOTION_CLASSES = [
        "Neutral",
        "Happy",
        "Surprise",
        "Sad",
        "Angry",
        "Disgust",
        "Fear"
    ]

    # Mapping raw FERPlus 8-class outputs to our standardized classes:
    # FERPlus raw classes: 0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt
    RAW_FERPLUS_MAP = {
        0: "Neutral",
        1: "Happy",
        2: "Surprise",
        3: "Sad",
        4: "Angry",
        5: "Disgust",
        6: "Fear",
        7: "Neutral" # Map subtle contempt to Neutral
    }

    def __init__(self, model_path: str = "models/emotion-ferplus-8.onnx", smoothing_factor: float = 0.65):
        self.model_path = model_path
        self.smoothing_factor = smoothing_factor  # Exponential Moving Average weight
        self.net = None
        self.is_loaded = False
        
        # Adaptive contrast enhancer for varying webcam/ambient lighting
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Per-track temporal probability history for smooth transitions: {track_id: np.ndarray(probs)}
        self._history: Dict[int, np.ndarray] = {}
        
        self._load_model()

    def _load_model(self):
        """Loads the ONNX emotion recognition model into OpenCV DNN runtime."""
        target_path = self.model_path
        if not os.path.isabs(target_path):
            if not os.path.exists(target_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                candidate = os.path.join(project_root, target_path)
                if os.path.exists(candidate):
                    target_path = candidate

        if not os.path.exists(target_path):
            logger.error(f"Emotion model file not found at: {self.model_path}")
            self.is_loaded = False
            return
            
        try:
            self.net = cv2.dnn.readNetFromONNX(target_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.is_loaded = True
            logger.info(f"EmotionDetector loaded ONNX model successfully from {target_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model into cv2.dnn: {e}")
            self.is_loaded = False

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax activation."""
        e_x = np.exp(x - np.max(x))
        return e_x / (np.sum(e_x) + 1e-9)

    def preprocess_face(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Prepares a cropped facial image ROI for CNN input (1, 1, 64, 64) normalized float32.
        Applies CLAHE contrast enhancement to normalize shadows, backlighting, and dark rooms.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        # Convert to grayscale
        if len(face_bgr.shape) == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr

        # Apply CLAHE to equalize facial micro-contrast across challenging lighting conditions
        try:
            enhanced = self.clahe.apply(gray)
        except Exception:
            enhanced = gray

        # Use cubic interpolation for smaller faces to preserve sharp brow and lip features
        interp = cv2.INTER_CUBIC if (enhanced.shape[0] < 64 or enhanced.shape[1] < 64) else cv2.INTER_AREA
        resized = cv2.resize(enhanced, (64, 64), interpolation=interp)

        # Float32 tensor for CNN
        blob = resized.astype(np.float32)
        
        # Reshape to NCHW format: (1, 1, 64, 64)
        blob = np.expand_dims(blob, axis=0)
        blob = np.expand_dims(blob, axis=0)
        return blob

    def predict(
        self,
        face_roi: np.ndarray,
        track_id: Optional[int] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Infers emotion for a single face crop with calibrated class weights and adaptive EMA smoothing.
        Returns:
          - dominant_emotion: str (e.g. "Happy")
          - confidence_percentage: float (0.0 - 100.0)
          - probabilities_dict: { "Happy": 92.4, "Neutral": 5.1, ... }
        """
        if not self.is_loaded or self.net is None:
            return "EMOTION UNKNOWN", 0.0, {k: 0.0 for k in self.EMOTION_CLASSES}

        blob = self.preprocess_face(face_roi)
        if blob is None:
            return "EMOTION UNKNOWN", 0.0, {k: 0.0 for k in self.EMOTION_CLASSES}

        try:
            self.net.setInput(blob)
            raw_logits = self.net.forward() # shape (1, 8)
            raw_probs = self._softmax(raw_logits[0]) # 8 raw probs

            # FERPlus raw output mapping:
            # 0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt
            # Map canonical classes cleanly without artificially inflating neutral with contempt
            class_probs = {
                "Neutral": float(raw_probs[0]),
                "Happy": float(raw_probs[1]),
                "Surprise": float(raw_probs[2]),
                "Sad": float(raw_probs[3]),
                "Angry": float(raw_probs[4]),
                "Disgust": float(raw_probs[5]),
                "Fear": float(raw_probs[6]),
            }

            # Calibrated sensitivity weights to balance natural expressions and prevent resting-face neutral lock:
            # - Neutral: slightly damped so subtle smiles or surprise can be recognized naturally
            # - Happy: boosted sensitivity so genuine smiles register promptly
            # - Surprise / Angry / Disgust / Fear: boosted to accurately detect expressive micro-cues
            CLASS_SENSITIVITY = {
                "Neutral": 0.88,
                "Happy": 1.25,
                "Surprise": 1.18,
                "Sad": 0.95,
                "Angry": 1.15,
                "Disgust": 1.12,
                "Fear": 1.12
            }

            weighted_probs = {c: class_probs[c] * CLASS_SENSITIVITY[c] for c in self.EMOTION_CLASSES}
            total_sum = sum(weighted_probs.values()) + 1e-9
            prob_vector = np.array([weighted_probs[c] / total_sum for c in self.EMOTION_CLASSES], dtype=np.float32)

            # Adaptive Temporal Exponential Moving Average smoothing
            if track_id is not None:
                if track_id in self._history:
                    prev_vector = self._history[track_id]
                    prev_max_idx = int(np.argmax(prev_vector))
                    curr_max_idx = int(np.argmax(prob_vector))
                    curr_confidence = prob_vector[curr_max_idx]

                    # When face clearly shifts expression with confidence (> 35%), adapt quickly (alpha = 0.35)
                    # When holding the same expression, keep high stability (alpha = 0.72) to eliminate jitter
                    if curr_max_idx != prev_max_idx and curr_confidence > 0.35:
                        alpha = 0.35
                    else:
                        alpha = 0.72

                    smoothed_vector = alpha * prev_vector + (1.0 - alpha) * prob_vector
                    smoothed_vector = smoothed_vector / (np.sum(smoothed_vector) + 1e-9)
                    self._history[track_id] = smoothed_vector
                    prob_vector = smoothed_vector
                else:
                    self._history[track_id] = prob_vector

            # Extract highest probability emotion
            max_idx = int(np.argmax(prob_vector))
            dominant_emotion = self.EMOTION_CLASSES[max_idx]
            confidence_pct = round(float(prob_vector[max_idx]) * 100.0, 1)

            # Build formatted distribution dictionary
            distribution = {
                c: round(float(prob_vector[i]) * 100.0, 1)
                for i, c in enumerate(self.EMOTION_CLASSES)
            }

            return dominant_emotion, confidence_pct, distribution

        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return "EMOTION UNKNOWN", 0.0, {k: 0.0 for k in self.EMOTION_CLASSES}

    def clear_track_history(self, active_track_ids: List[int]):
        """Clean up smoothing history for deregistered tracks."""
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self._history if tid not in active_set]
        for tid in stale_ids:
            del self._history[tid]
