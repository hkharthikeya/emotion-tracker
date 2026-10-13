"""
URL routing for emotion_app endpoints.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Dashboard HTML UI
    path('', views.dashboard_view, name='dashboard'),
    
    # REST API Endpoints
    path('api/status/', views.system_status_api, name='api_status'),
    path('api/camera/status/', views.camera_status_api, name='api_camera_status'),
    path('api/camera/config/', views.camera_config_api, name='api_camera_config'),
    path('api/emotions/', views.emotion_history_api, name='api_emotion_history'),
    path('api/emotions/stats/', views.emotion_stats_api, name='api_emotion_stats'),
    path('api/emotions/clear/', views.clear_emotion_history_api, name='api_clear_emotions'),
    path('api/nlp/analyze/', views.nlp_analyze_api, name='api_nlp_analyze'),
    path('api/multimodal/fuse/', views.multimodal_fuse_api, name='api_multimodal_fuse'),
    
    # MJPEG HTTP Stream Fallback
    path('api/stream/', views.video_stream_view, name='api_stream'),
]
