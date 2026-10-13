"""
EmotiCam Camera Source Client.
Handles robust HTTP Wi-Fi frame retrieval from ESP32-CAM, with seamless fallback/support
for local Laptop Webcam testing without restarting the application.
"""

import cv2
import numpy as np
import requests
import time
import threading
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ESP32Camera:
    """
    Dual-mode Camera Client:
      - Mode A (Default): AI-Thinker ESP32-CAM via HTTP /capture over Wi-Fi
      - Mode B: Local Laptop Webcam (cv2.VideoCapture) for quick local validation
    """
    def __init__(self, base_url: str = "http://192.168.1.100", timeout: float = 1.8):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Internal state
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_fetch_latency_ms: float = 0.0
        self._last_successful_fetch_time: float = 0.0
        self._status: str = "CONNECTING"
        self._error_message: str = ""
        self._frame_count: int = 0
        self._is_webcam_mode: bool = False
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Thread safety & lifecycle
        self._lock = threading.Lock()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        
        # HTTP Session with connection pooling
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=0)
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

        self._check_and_configure_mode(self.base_url)

    def _check_and_configure_mode(self, url: str):
        """Determines if the requested source is a local webcam or remote ESP32."""
        clean = url.lower().strip()
        if clean in ['0', '1', '2', 'webcam', 'local', 'cam', 'default'] or clean.startswith('webcam'):
            self._is_webcam_mode = True
            cam_idx = 0
            if clean in ['0', '1', '2']:
                cam_idx = int(clean)
            elif ':' in clean:
                try:
                    cam_idx = int(clean.split(':')[1])
                except Exception:
                    cam_idx = 0
            self._init_webcam(cam_idx)
        else:
            self._is_webcam_mode = False
            self._release_webcam()

    def _init_webcam(self, index: int = 0):
        """Initializes OpenCV VideoCapture for local webcam."""
        try:
            self._release_webcam()
                
            # Try DirectShow first on Windows to avoid MSMF errors
            self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(index)
                
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, test_frame = self._cap.read()
                if ret and test_frame is not None:
                    self._status = "ONLINE"
                    self._error_message = ""
                    logger.info(f"Local webcam (Index {index}) opened and active.")
                    return
                else:
                    self._status = "OFFLINE"
                    self._error_message = f"Webcam opened but failed to capture frame."
            else:
                self._status = "OFFLINE"
                self._error_message = f"Failed to open local webcam (Index {index})."
        except Exception as e:
            self._status = "OFFLINE"
            self._error_message = f"Webcam init error: {e}"

    def _release_webcam(self):
        """Releases local webcam capture device if open."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def set_camera_url(self, new_url: str):
        """Dynamically update ESP32-CAM URL or switch to local webcam."""
        with self._lock:
            self.base_url = new_url.rstrip('/')
            self._status = "CONNECTING"
            self._error_message = ""
            self._check_and_configure_mode(self.base_url)
            logger.info(f"Camera source updated to: {self.base_url} (Webcam mode: {self._is_webcam_mode})")

    def get_capture_url(self) -> str:
        return f"{self.base_url}/capture"

    def get_status_url(self) -> str:
        return f"{self.base_url}/status"

    def check_camera_status(self) -> Dict[str, Any]:
        """Queries camera status."""
        if self._is_webcam_mode:
            is_active = bool(self._cap and self._cap.isOpened() and self._status == "ONLINE")
            return {
                "device": "Laptop Webcam",
                "camera": "online" if is_active else "offline",
                "source": "Local USB/Integrated Camera",
                "ping_ms": 1.0,
                "reachable": is_active
            }

        try:
            t0 = time.perf_counter()
            resp = self._session.get(self.get_status_url(), timeout=0.8)
            ping_ms = (time.perf_counter() - t0) * 1000.0
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    data['ping_ms'] = round(ping_ms, 1)
                    data['reachable'] = True
                    return data
                except Exception:
                    return {
                        "device": "ESP32-CAM",
                        "camera": "online",
                        "ip": self.base_url,
                        "ping_ms": round(ping_ms, 1),
                        "reachable": True
                    }
        except Exception as e:
            return {
                "device": "ESP32-CAM",
                "camera": "offline",
                "ip": self.base_url,
                "reachable": False,
                "error": str(e)
            }
        return {"reachable": False, "camera": "offline", "ip": self.base_url}

    def fetch_single_frame(self) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        """
        Fetches one raw frame:
          - If webcam mode: grabs directly from local camera device
          - If HTTP mode: fetches /capture JPEG from ESP32-CAM over Wi-Fi
        Returns: (frame_bgr, latency_ms, error_msg)
        """
        if self._is_webcam_mode:
            if self._cap is None or not self._cap.isOpened():
                self._init_webcam(0)
                if self._cap is None or not self._cap.isOpened() or self._status == "OFFLINE":
                    return None, 0.0, self._error_message or "Laptop webcam device is not available or busy."

            t_start = time.perf_counter()
            ret, frame = self._cap.read()
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            
            if not ret or frame is None:
                return None, 0.0, "Failed to read frame from laptop webcam."
            return frame, latency_ms, None

        # ESP32-CAM HTTP Fetch
        try:
            t_start = time.perf_counter()
            resp = self._session.get(self.get_capture_url(), timeout=self.timeout)
            
            if resp.status_code != 200:
                return None, 0.0, f"ESP32 HTTP {resp.status_code}: {resp.text[:50]}"
            
            jpeg_bytes = resp.content
            if len(jpeg_bytes) < 100:
                return None, 0.0, "Empty or truncated JPEG received from ESP32-CAM"
            
            np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None, 0.0, "OpenCV failed to decode corrupted JPEG frame"
            
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return frame, latency_ms, None
            
        except requests.exceptions.Timeout:
            return None, 0.0, "ESP32-CAM connection timed out (>1.8s)"
        except requests.exceptions.ConnectionError:
            return None, 0.0, f"Cannot reach ESP32 at {self.base_url}. Check Wi-Fi or switch to Webcam."
        except Exception as e:
            return None, 0.0, f"Camera fetch error: {str(e)}"

    def _worker_loop(self, target_fps: int = 15):
        """Continuous background thread fetching latest frames."""
        interval = 1.0 / max(1, target_fps)
        logger.info(f"Camera worker thread started at target {target_fps} FPS.")
        
        consecutive_failures = 0
        while self._is_running:
            loop_start = time.perf_counter()
            
            frame, latency_ms, err = self.fetch_single_frame()
            
            with self._lock:
                if frame is not None:
                    self._latest_frame = frame
                    self._latest_fetch_latency_ms = latency_ms
                    self._last_successful_fetch_time = time.time()
                    self._status = "ONLINE"
                    self._error_message = ""
                    self._frame_count += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    self._error_message = err or "Unknown fetch error"
                    if consecutive_failures >= 3:
                        self._status = "OFFLINE"
                        if self._is_webcam_mode and consecutive_failures >= 10:
                            self._release_webcam()
            
            # Rate limiting / backoff when offline
            elapsed = time.perf_counter() - loop_start
            if self._status == "OFFLINE":
                time.sleep(max(0.5, interval - elapsed))
            else:
                sleep_time = interval - elapsed
                if sleep_time > 0.002:
                    time.sleep(sleep_time)

    def start(self, target_fps: int = 15):
        """Starts background frame polling thread."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._thread = threading.Thread(target=self._worker_loop, args=(target_fps,), daemon=True)
            self._thread.start()
            logger.info("Camera frame grabber background worker started.")

    def stop(self):
        """Stops background frame polling thread."""
        with self._lock:
            self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._release_webcam()
        logger.info("Camera grabber stopped.")

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, str, str]:
        """
        Thread-safe retrieval of the most recently decoded frame.
        Returns: (frame, fetch_latency_ms, status, error_message)
        """
        with self._lock:
            if self._last_successful_fetch_time > 0 and (time.time() - self._last_successful_fetch_time > 3.0):
                current_status = "OFFLINE"
            else:
                current_status = self._status
                
            frame_copy = self._latest_frame.copy() if self._latest_frame is not None else None
            return frame_copy, self._latest_fetch_latency_ms, current_status, self._error_message
