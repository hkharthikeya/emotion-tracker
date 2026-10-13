"""
EmotiCam Django Channels WebSocket Consumers.
Streams real-time processed video frames, HUD metadata, emotion analytics, and handles client commands.
"""

import json
import asyncio
import time
import base64
import logging
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from asgiref.sync import sync_to_async

from ai.processor import StreamProcessor
from ai.emotion_detector import EmotionDetector
from .models import EmotionLog

logger = logging.getLogger(__name__)

class EmotionStreamConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer managing continuous live frame streaming and bidirectional control commands.
    Endpoint: ws://<host>/ws/stream/
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.stream_task = None
        self._last_db_log_time = 0.0
        self._last_logged_emotion = None

    async def connect(self):
        await self.accept()
        self.is_streaming = True
        logger.info(f"WebSocket client connected: {self.channel_name}")
        
        # Send initial connection confirmation
        await self.send(text_data=json.dumps({
            "type": "connection_established",
            "message": "Connected to EmotiCam Real-Time AI Stream",
            "timestamp": time.strftime("%H:%M:%S")
        }))
        
        # Start streaming loop
        self.stream_task = asyncio.create_task(self._stream_loop())

    async def disconnect(self, close_code):
        self.is_streaming = False
        if self.stream_task:
            self.stream_task.cancel()
        logger.info(f"WebSocket client disconnected ({close_code}): {self.channel_name}")

    async def receive(self, text_data=None, bytes_data=None):
        """Handle incoming client commands."""
        if not text_data:
            return
            
        try:
            data = json.loads(text_data)
            command = data.get("command", "")
            
            if command == "start_processing":
                self.is_streaming = True
                if not self.stream_task or self.stream_task.done():
                    self.stream_task = asyncio.create_task(self._stream_loop())
                await self.send(text_data=json.dumps({
                    "type": "status_update",
                    "ai_status": "PROCESSING",
                    "message": "Stream processing resumed."
                }))
                
            elif command == "stop_processing":
                self.is_streaming = False
                await self.send(text_data=json.dumps({
                    "type": "status_update",
                    "ai_status": "STOPPED",
                    "message": "Stream processing paused by user."
                }))

            elif command == "reconnect_camera":
                processor = StreamProcessor.get_instance()
                processor.camera._status = "CONNECTING"
                await self.send(text_data=json.dumps({
                    "type": "status_update",
                    "camera_status": "CONNECTING",
                    "message": "Reconnecting to ESP32-CAM..."
                }))

            elif command == "set_camera_url":
                new_url = data.get("url", "").strip()
                if new_url:
                    processor = StreamProcessor.get_instance()
                    processor.set_camera_url(new_url)
                    await self.send(text_data=json.dumps({
                        "type": "status_update",
                        "camera_url": new_url,
                        "message": f"Target Camera URL changed to {new_url}"
                    }))

            elif command == "release_camera":
                processor = StreamProcessor.get_instance()
                processor.camera._release_webcam()

            elif command == "webcam_frame":
                # Browser-sent frame: base64 JPEG → decode → run AI pipeline
                frame_data = data.get("frame", "")
                if frame_data:
                    try:
                        # Strip data-URI prefix if present
                        if "," in frame_data:
                            frame_data = frame_data.split(",", 1)[1]
                        jpeg_bytes = base64.b64decode(frame_data)
                        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        import cv2
                        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame_bgr is not None:
                            processor = StreamProcessor.get_instance()
                            payload = await sync_to_async(
                                processor.process_raw_frame, thread_sensitive=False
                            )(frame_bgr)
                            await self._maybe_persist_log(payload)
                            await self.send(text_data=json.dumps(payload))
                    except Exception as e:
                        logger.error(f"webcam_frame processing error: {e}")

            elif command == "ping":
                await self.send(text_data=json.dumps({"type": "pong", "timestamp": time.time()}))
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _stream_loop(self):
        """Asynchronous streaming loop dispatching frames and AI telemetry."""
        processor = StreamProcessor.get_instance()
        
        while self.is_streaming:
            try:
                loop_start = time.perf_counter()
                
                # Run CPU-bound image processing in thread pool to avoid blocking asyncio event loop
                payload = await sync_to_async(processor.process_frame, thread_sensitive=False)()
                
                # Check if we should log to database (every 3 seconds when face detected or on emotion switch)
                await self._maybe_persist_log(payload)
                
                # Send JSON payload with base64 image over WebSocket
                await self.send(text_data=json.dumps(payload))
                
                # Pacing interval based on configured FPS (e.g. ~15-20 FPS)
                elapsed = time.perf_counter() - loop_start
                target_interval = 0.05  # ~20 FPS max refresh
                sleep_delay = max(0.01, target_interval - elapsed)
                await asyncio.sleep(sleep_delay)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket streaming loop: {e}")
                await asyncio.sleep(0.5)

    async def _maybe_persist_log(self, payload: dict):
        """Persists high-confidence face emotion detections periodically to SQLite."""
        face_count = payload.get("face_count", 0)
        primary_emotion = payload.get("primary_emotion", "")
        confidence = payload.get("confidence", 0.0)
        
        if face_count > 0 and primary_emotion not in ["No Face", "Unknown"] and confidence > 45.0:
            now = time.time()
            # Log if emotion changed OR if 4 seconds elapsed since last log
            if (primary_emotion != self._last_logged_emotion) or (now - self._last_db_log_time > 4.0):
                self._last_db_log_time = now
                self._last_logged_emotion = primary_emotion
                
                # Extract face ID safely
                faces = payload.get("faces", [])
                face_id = faces[0].get("track_id", 1) if (isinstance(faces, list) and len(faces) > 0 and isinstance(faces[0], dict)) else 1
                
                # Save asynchronously to DB
                processor = StreamProcessor.get_instance()
                await self._save_emotion_log(
                    face_id=face_id,
                    primary_emotion=primary_emotion,
                    confidence=confidence,
                    face_count=face_count,
                    fps=payload.get("fps", 0.0),
                    latency_ms=payload.get("latency_ms", 0.0),
                    camera_ip=processor.camera_url
                )

    @database_sync_to_async
    def _save_emotion_log(self, face_id, primary_emotion, confidence, face_count, fps, latency_ms, camera_ip):
        EmotionLog.objects.create(
            face_id=face_id,
            emotion=primary_emotion,
            confidence=confidence,
            face_count=face_count,
            fps=fps,
            latency_ms=latency_ms,
            camera_ip=camera_ip
        )

