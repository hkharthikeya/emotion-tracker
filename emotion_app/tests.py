"""
Comprehensive Unit Tests for EmotiCam System.
Tests models, REST APIs, AI detector, face tracker, and emotion classification.
"""

from django.test import TestCase, Client
from django.urls import reverse
import numpy as np
import cv2

from emotion_app.models import EmotionLog, CameraConfig
from ai.face_tracker import CentroidTracker
from ai.face_detector import FaceDetector
from ai.emotion_detector import EmotionDetector
from ai.utils import EMOTION_COLORS, EMOTION_EMOJIS

class EmotionModelTestCase(TestCase):
    def setUp(self):
        self.log_entry = EmotionLog.objects.create(
            face_id=1,
            emotion='Happy',
            confidence=92.5,
            face_count=1,
            fps=14.2,
            latency_ms=85.4,
            camera_ip="http://192.168.1.100"
        )
        self.config = CameraConfig.objects.create(
            name="AI-Thinker ESP32-CAM",
            camera_url="http://192.168.1.100",
            is_active=True,
            target_fps=15
        )

    def test_emotion_log_creation(self):
        self.assertEqual(self.log_entry.emotion, 'Happy')
        self.assertEqual(self.log_entry.confidence, 92.5)
        self.assertTrue('Happy' in str(self.log_entry))

    def test_camera_config_creation(self):
        self.assertEqual(self.config.target_fps, 15)
        self.assertTrue('AI-Thinker' in str(self.config))


class APITestCase(TestCase):
    def setUp(self):
        self.client = Client()
        EmotionLog.objects.create(
            face_id=1,
            emotion='Happy',
            confidence=95.0,
            face_count=1,
            fps=15.0,
            latency_ms=80.0,
            camera_ip="http://192.168.1.100"
        )

    def test_dashboard_renders(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'EmotiCam')

    def test_api_status(self):
        response = self.client.get(reverse('api_status'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('system'), 'EmotiCam')
        self.assertEqual(data.get('django_status'), 'ONLINE')

    def test_api_emotions_history(self):
        response = self.client.get(reverse('api_emotion_history'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data.get('count'), 1)

    def test_api_emotion_stats(self):
        response = self.client.get(reverse('api_emotion_stats'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('Happy', data.get('counts'))

    def test_camera_config_update(self):
        response = self.client.post(
            reverse('api_camera_config'),
            data={'camera_url': 'http://192.168.1.150'},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('camera_url'), 'http://192.168.1.150')

    def test_nlp_analyze_api(self):
        response = self.client.post(
            reverse('api_nlp_analyze'),
            data={'text': 'I am extremely happy and thrilled with the great progress!'},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('dominant_emotion'), 'Happy')
        self.assertGreaterEqual(data.get('confidence'), 20.0)

    def test_multimodal_fuse_api(self):
        response = self.client.post(
            reverse('api_multimodal_fuse'),
            data={
                'text': 'I feel sorrow and heartbroken.',
                'face_data': {
                    'primary_emotion': 'Sad',
                    'confidence': 88.0,
                    'distribution': {
                        'Happy': 2.0, 'Neutral': 5.0, 'Surprise': 1.0,
                        'Sad': 88.0, 'Angry': 2.0, 'Disgust': 1.0, 'Fear': 1.0
                    }
                }
            },
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'success')
        mm = data.get('multimodal', {})
        self.assertEqual(mm.get('multimodal_dominant_emotion'), 'Sad')
        self.assertTrue(mm.get('is_congruent'))


class AITrackerAndDetectorTestCase(TestCase):
    def test_centroid_tracker(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=50.0)
        # Frame 1: Detection at (100, 100, 50, 50)
        tracked = tracker.update([(100, 100, 50, 50)])
        self.assertEqual(len(tracked), 1)
        self.assertIn(1, tracked)

        # Frame 2: Face moved slightly to (105, 102, 50, 50)
        tracked = tracker.update([(105, 102, 50, 50)])
        self.assertEqual(len(tracked), 1)
        self.assertIn(1, tracked) # Should maintain same ID: 1

        # Frame 3: Disappeared for 6 frames -> should deregister
        for _ in range(6):
            tracked = tracker.update([])
        self.assertEqual(len(tracked), 0)

    def test_emotion_detector_dummy_inference(self):
        detector = EmotionDetector(model_path="models/emotion-ferplus-8.onnx")
        if detector.is_loaded:
            # Create a synthetic face crop (64, 64, 3)
            synthetic_face = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            emotion, conf, dist = detector.predict(synthetic_face, track_id=1)
            self.assertIn(emotion, EmotionDetector.EMOTION_CLASSES)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 100.0)
            self.assertEqual(len(dist), 7)
