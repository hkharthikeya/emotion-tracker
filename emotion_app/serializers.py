from rest_framework import serializers
from .models import EmotionLog, CameraConfig

class EmotionLogSerializer(serializers.ModelSerializer):
    timestamp_formatted = serializers.SerializerMethodField()

    class Meta:
        model = EmotionLog
        fields = [
            'id',
            'timestamp',
            'timestamp_formatted',
            'face_id',
            'emotion',
            'confidence',
            'face_count',
            'fps',
            'latency_ms',
            'camera_ip'
        ]

    def get_timestamp_formatted(self, obj):
        return obj.timestamp.strftime('%H:%M:%S')

class CameraConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = CameraConfig
        fields = ['id', 'name', 'camera_url', 'is_active', 'target_fps', 'updated_at']
