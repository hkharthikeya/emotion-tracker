from django.contrib import admin
from .models import EmotionLog, CameraConfig

@admin.register(EmotionLog)
class EmotionLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'face_id', 'emotion', 'confidence', 'face_count', 'fps', 'latency_ms', 'camera_ip')
    list_filter = ('emotion', 'timestamp')
    search_fields = ('emotion', 'camera_ip')
    readonly_fields = ('timestamp', 'face_id', 'emotion', 'confidence', 'face_count', 'fps', 'latency_ms', 'camera_ip')
    ordering = ('-timestamp',)

@admin.register(CameraConfig)
class CameraConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'camera_url', 'is_active', 'target_fps', 'updated_at')
    list_editable = ('camera_url', 'is_active', 'target_fps')
