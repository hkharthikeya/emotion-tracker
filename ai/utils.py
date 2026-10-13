"""
EmotiCam AI Utilities - HUD drawing, color schemes, image conversions.
"""

import cv2
import numpy as np
import base64

# Emotion color scheme (BGR format for OpenCV)
# Curated vibrant palette with high contrast against dark video backgrounds
EMOTION_COLORS = {
    "Happy": (34, 197, 94),     # Bright Emerald Green
    "Neutral": (59, 130, 246),  # Modern Cyan Blue
    "Sad": (249, 115, 22),     # Amber Orange
    "Angry": (68, 68, 239),     # Crimson Red
    "Surprise": (234, 179, 8),  # Vibrant Gold / Yellow
    "Fear": (168, 85, 247),    # Neon Violet
    "Disgust": (16, 185, 129),  # Teal / Mint
    "Unknown": (148, 163, 184)  # Slate Gray
}

# Emotion Emoji mapping
EMOTION_EMOJIS = {
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Angry": "😠",
    "Surprise": "😲",
    "Fear": "😨",
    "Disgust": "🤢",
    "Unknown": "❓"
}

def draw_hud_box(
    image: np.ndarray,
    bbox: tuple,
    track_id: int,
    emotion: str,
    confidence: float,
    color: tuple = None
) -> np.ndarray:
    """
    Draws a futuristic, professional AI HUD bounding box around a detected face
    with corner brackets, header badge for tracking ID, and bottom badge for emotion/confidence.
    """
    x, y, w, h = bbox
    if color is None:
        color = EMOTION_COLORS.get(emotion, (59, 130, 246))

    # Bounding box coordinates
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # 1. Subtle semi-transparent bounding box outline
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # 2. Glowing corner brackets for futuristic AI aesthetic
    bracket_len = max(12, int(min(w, h) * 0.18))
    thickness = 3
    
    # Top-Left
    cv2.line(image, (x1, y1), (x1 + bracket_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1), (x1, y1 + bracket_len), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(image, (x2, y1), (x2 - bracket_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1), (x2, y1 + bracket_len), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(image, (x1, y2), (x1 + bracket_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y2), (x1, y2 - bracket_len), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(image, (x2, y2), (x2 - bracket_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y2), (x2, y2 - bracket_len), color, thickness, cv2.LINE_AA)

    # 3. Top Tracking Badge: "Face #<id>"
    badge_text = f"Face #{track_id}" if track_id is not None else "Face"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, font_thickness)
    
    badge_y1 = max(0, y1 - th - 10)
    badge_y2 = y1
    badge_x2 = min(image.shape[1], x1 + tw + 14)
    
    # Draw badge background
    cv2.rectangle(image, (x1, badge_y1), (badge_x2, badge_y2), (15, 23, 42), -1)
    cv2.rectangle(image, (x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)
    cv2.putText(image, badge_text, (x1 + 7, badge_y2 - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # 4. Bottom Emotion & Confidence Badge: "<EMOTION> <CONF>%"
    conf_str = f"{confidence:.0f}%" if confidence is not None and confidence > 0 else ""
    label_text = f"{emotion.upper()} {conf_str}".strip()
    
    (lw, lh), _ = cv2.getTextSize(label_text, font, 0.55, 2)
    lbl_y1 = y2
    lbl_y2 = min(image.shape[0], y2 + lh + 12)
    lbl_x2 = min(image.shape[1], x1 + lw + 16)
    
    # Emotion badge banner
    cv2.rectangle(image, (x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
    # Dark text for high contrast on bright color banner
    text_color = (15, 23, 42) if emotion in ["Happy", "Surprise", "Disgust", "Neutral"] else (255, 255, 255)
    cv2.putText(image, label_text, (x1 + 8, lbl_y2 - 6), font, 0.55, text_color, 2, cv2.LINE_AA)

    return image

def draw_no_face_hud(image: np.ndarray) -> np.ndarray:
    """Draws a subtle on-screen status banner when no faces are detected in view."""
    h, w, _ = image.shape
    banner_text = "NO FACE DETECTED - SEARCHING..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(banner_text, font, 0.5, 1)
    
    # Top center banner
    bx1 = (w - tw) // 2 - 14
    bx2 = (w + tw) // 2 + 14
    by1 = 12
    by2 = by1 + th + 14
    
    overlay = image.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (100, 116, 139), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    
    cv2.putText(image, banner_text, ((w - tw) // 2, by2 - 6), font, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
    return image

def encode_image_to_base64_jpeg(image: np.ndarray, quality: int = 80) -> str:
    """Encodes an OpenCV BGR numpy image to base64 JPEG data URL."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode('.jpg', image, encode_params)
    if not success:
        return ""
    b64_bytes = base64.b64encode(buffer)
    return f"data:image/jpeg;base64,{b64_bytes.decode('utf-8')}"

def encode_image_to_jpeg_bytes(image: np.ndarray, quality: int = 80) -> bytes:
    """Encodes an OpenCV BGR numpy image to raw JPEG bytes for MJPEG streaming."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode('.jpg', image, encode_params)
    if not success:
        return b""
    return buffer.tobytes()
