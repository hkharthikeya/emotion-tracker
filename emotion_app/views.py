"""
EmotiCam Django Views & REST API Endpoints.
"""

from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import Count, Avg
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import json
import logging
from .models import EmotionLog, CameraConfig
from .serializers import EmotionLogSerializer, CameraConfigSerializer
from ai.processor import StreamProcessor
from ai.emotion_detector import EmotionDetector
from ai.nlp_emotion import TextEmotionAnalyzer

nlp_analyzer = TextEmotionAnalyzer()

logger = logging.getLogger(__name__)

def dashboard_view(request):
    """
    Renders the EmotiCam Real-Time Facial Emotion Analysis Dashboard.
    """
    processor = StreamProcessor.get_instance()
    context = {
        'page_title': 'EmotiCam — Real-Time Facial Emotion Analysis',
        'camera_url': processor.camera_url,
        'camera_status': processor.camera._status,
        'ai_status': 'ACTIVE' if processor.emotion_detector.is_loaded else 'ERROR',
        'emotions_list': EmotionDetector.EMOTION_CLASSES,
    }
    return render(request, 'index.html', context)

@api_view(['GET', 'HEAD'])
def system_status_api(request):
    """
    GET /api/status/
    Returns full diagnostic status of Django, AI model, and Camera.
    """
    processor = StreamProcessor.get_instance()
    cam_status_info = processor.camera.check_camera_status()
    
    total_logs = EmotionLog.objects.count()
    
    return Response({
        "system": "EmotiCam",
        "version": "1.0.0",
        "django_status": "ONLINE",
        "ai_engine": "OpenCV DNN (ONNX)",
        "ai_status": "ONLINE" if processor.emotion_detector.is_loaded else "OFFLINE",
        "model_path": processor.model_path,
        "camera_configured_url": processor.camera_url,
        "camera_reachable": cam_status_info.get("reachable", False),
        "camera_details": cam_status_info,
        "current_fps": round(processor.fps, 1),
        "current_latency_ms": processor.latency_ms,
        "total_emotion_records": total_logs
    })

@api_view(['GET', 'HEAD'])
def camera_status_api(request):
    """
    GET /api/camera/status/
    Directly probes ESP32-CAM /status endpoint.
    """
    processor = StreamProcessor.get_instance()
    status_data = processor.camera.check_camera_status()
    return Response(status_data)

@api_view(['POST'])
def camera_config_api(request):
    """
    POST /api/camera/config/
    Updates ESP32-CAM target IP without restart.
    Payload: {"camera_url": "http://192.168.1.105"}
    """
    new_url = request.data.get('camera_url', '').strip()
    if not new_url:
        return Response({'error': 'camera_url is required'}, status=status.HTTP_400_BAD_REQUEST)

    is_webcam = new_url.lower() in ['0', '1', '2', 'webcam', 'local', 'cam', 'default'] or new_url.lower().startswith('webcam')
    if not is_webcam and not (new_url.startswith('http://') or new_url.startswith('https://')):
        new_url = f"http://{new_url}"

    processor = StreamProcessor.get_instance()
    processor.set_camera_url(new_url)

    # Persist in DB
    config, _ = CameraConfig.objects.get_or_create(id=1)
    config.camera_url = new_url
    config.save()

    return Response({
        'status': 'success',
        'message': f'Camera URL updated to {new_url}',
        'camera_url': new_url
    })

@api_view(['GET', 'HEAD'])
def emotion_history_api(request):
    """
    GET /api/emotions/
    Returns recent emotion records with pagination / limit.
    """
    limit = int(request.GET.get('limit', 25))
    emotion_filter = request.GET.get('emotion', None)
    
    queryset = EmotionLog.objects.all()
    if emotion_filter:
        queryset = queryset.filter(emotion__iexact=emotion_filter)
        
    logs = queryset[:limit]
    serializer = EmotionLogSerializer(logs, many=True)
    return Response({
        "count": queryset.count(),
        "results": serializer.data
    })

@api_view(['GET', 'HEAD'])
def emotion_stats_api(request):
    """
    GET /api/emotions/stats/
    Returns aggregated emotion frequency distribution for analytics & charts.
    """
    total_count = EmotionLog.objects.count()
    if total_count == 0:
        empty_dist = {e: 0 for e in EmotionDetector.EMOTION_CLASSES}
        return Response({
            "total_samples": 0,
            "counts": empty_dist,
            "percentages": empty_dist,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0
        })

    # Group by emotion
    counts_query = EmotionLog.objects.values('emotion').annotate(total=Count('id'))
    counts_dict = {e: 0 for e in EmotionDetector.EMOTION_CLASSES}
    for item in counts_query:
        if item['emotion'] in counts_dict:
            counts_dict[item['emotion']] = item['total']

    percentages_dict = {
        k: round((v / total_count) * 100.0, 1) for k, v in counts_dict.items()
    }

    avg_stats = EmotionLog.objects.aggregate(
        avg_conf=Avg('confidence'),
        avg_lat=Avg('latency_ms')
    )

    return Response({
        "total_samples": total_count,
        "counts": counts_dict,
        "percentages": percentages_dict,
        "avg_confidence": round(avg_stats.get('avg_conf') or 0.0, 1),
        "avg_latency_ms": round(avg_stats.get('avg_lat') or 0.0, 1)
    })

@api_view(['DELETE'])
def clear_emotion_history_api(request):
    """
    DELETE /api/emotions/clear/
    Clears all saved emotion history logs.
    """
    deleted_count, _ = EmotionLog.objects.all().delete()
    return Response({
        "status": "success",
        "message": f"Cleared {deleted_count} historical records."
    })

@api_view(['POST'])
def nlp_analyze_api(request):
    """
    POST /api/nlp/analyze/
    Analyzes emotion from submitted text payload.
    Payload: {"text": "I feel so grateful and happy today"}
    """
    text = request.data.get('text', '')
    result = nlp_analyzer.analyze(text)
    return Response(result)

@api_view(['POST'])
def multimodal_fuse_api(request):
    """
    POST /api/multimodal/fuse/
    Combines text sentiment with current or provided facial emotion data.
    Payload: {"text": "...", "face_weight": 0.6}
    """
    text = request.data.get('text', '')
    face_weight = float(request.data.get('face_weight', 0.6))
    
    text_result = nlp_analyzer.analyze(text)
    
    # Grab latest live frame telemetry if face data not supplied
    supplied_face = request.data.get('face_data')
    if supplied_face and isinstance(supplied_face, dict):
        face_result = supplied_face
    else:
        processor = StreamProcessor.get_instance()
        live_payload = processor.process_frame()
        face_result = {
            "primary_emotion": live_payload.get("primary_emotion", "Neutral"),
            "confidence": live_payload.get("confidence", 0.0),
            "distribution": live_payload.get("distribution", {})
        }
        
    fused_result = nlp_analyzer.fuse_multimodal(face_result, text_result, face_weight=face_weight)
    
    return Response({
        "status": "success",
        "multimodal": fused_result,
        "nlp_analysis": text_result,
        "face_analysis": face_result
    })

def video_stream_view(request):
    """
    GET /api/stream/
    Direct MJPEG video stream HTTP response.
    """
    processor = StreamProcessor.get_instance()
    return StreamingHttpResponse(
        processor.get_raw_jpeg_stream(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
