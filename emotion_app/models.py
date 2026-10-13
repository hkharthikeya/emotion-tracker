"""
EmotiCam Database Models.
Stores real-time emotion telemetry sessions and historical records without raw facial images.
"""

from django.db import models

class EmotionLog(models.Model):
    """
    Historical log entry of a detected facial expression event.
    """
    EMOTION_CHOICES = [
        ('Happy', 'Happy'),
        ('Neutral', 'Neutral'),
        ('Sad', 'Sad'),
        ('Angry', 'Angry'),
        ('Surprise', 'Surprise'),
        ('Fear', 'Fear'),
        ('Disgust', 'Disgust'),
        ('Unknown', 'Unknown'),
    ]

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    face_id = models.IntegerField(default=1, help_text="Session Tracking ID (Face #N)")
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    confidence = models.FloatField(help_text="Confidence percentage (0-100)")
    face_count = models.IntegerField(default=1, help_text="Total active faces in frame")
    fps = models.FloatField(default=0.0, help_text="Processing FPS at time of capture")
    latency_ms = models.FloatField(default=0.0, help_text="Pipeline latency in milliseconds")
    camera_ip = models.CharField(max_length=100, default="192.168.1.100")

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Emotion Record'
        verbose_name_plural = 'Emotion History Logs'

    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] Face #{self.face_id}: {self.emotion} ({self.confidence:.1f}%)"


class CameraConfig(models.Model):
    """
    Configurable ESP32-CAM parameters and runtime settings.
    """
    name = models.CharField(max_length=100, default="AI-Thinker ESP32-CAM")
    camera_url = models.CharField(max_length=255, default="http://192.168.1.100")
    is_active = models.BooleanField(default=True)
    target_fps = models.IntegerField(default=15)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Camera Configuration'
        verbose_name_plural = 'Camera Configurations'

    def __str__(self):
        return f"{self.name} ({self.camera_url})"
