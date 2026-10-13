# EmotiCam — Real-Time Facial Emotion Detection Using ESP32-CAM + Django + AI

![EmotiCam Banner](https://img.shields.io/badge/EmotiCam-Real--Time%20AI%20Vision-6366f1?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-Channels%20%2B%20Daphne-092E20?style=flat&logo=django&logoColor=white)
![ESP32](https://img.shields.io/badge/ESP32--CAM-OV2640-E7352C?style=flat&logo=espressif&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN%20ONNX%20Engine-5C3EE8?style=flat&logo=opencv&logoColor=white)

---

## 🌟 Executive Overview
**EmotiCam** is an end-to-end, edge-to-cloud facial emotion intelligence system. An **AI-Thinker ESP32-CAM** acts as the remote wireless optical sensor, continuously transmitting low-latency JPEG frames over local Wi-Fi. The **Django Channels** backend ingests the frames, processes them through an optimized **OpenCV & Deep Convolutional Neural Network (CNN)** pipeline, and streams real-time telemetry and annotated video feeds directly to a modern web dashboard accessible from laptops, tablets, and smartphones.

---

## 🏗️ System Architecture

```
                                    SAME Wi-Fi NETWORK
                   ┌──────────────────────────────────────────────────┐
                   │                                                  │
             ┌───────────┐                                            │
             │ ESP32-CAM │                                            │
             │  (OV2640) │                                            │
             └─────┬─────┘                                            │
                   │  HTTP GET /capture (Raw JPEG)                    │
                   ▼                                                  │
        ┌─────────────────────┐                                       │
        │   Django Backend    │                                       │
        │ ┌─────────────────┐ │                                       │
        │ │ OpenCV Decode   │ │                                       │
        │ └────────┬────────┘ │                                       │
        │          ▼          │                                       │
        │ ┌─────────────────┐ │                                       │
        │ │ Face Detection  │ │ (Haar Cascade + CLAHE)                │
        │ └────────┬────────┘ │                                       │
        │          ▼          │                                       │
        │ ┌─────────────────┐ │                                       │
        │ │ Face Tracking   │ │ (Centroid Tracker: Face #1, #2...)    │
        │ └────────┬────────┘ │                                       │
        │          ▼          │                                       │
        │ ┌─────────────────┐ │                                       │
        │ │ Emotion CNN     │ │ (Emotion-FERPlus ONNX via OpenCV DNN) │
        │ └────────┬────────┘ │                                       │
        │          ▼          │                                       │
        │ ┌─────────────────┐ │                                       │
        │ │ HUD Annotation  │ │ (Boxes, Badges, Probabilities)        │
        │ └────────┬────────┘ │                                       │
        └──────────┼──────────┘                                       │
                   │                                                  │
                   │  WebSocket (ws://<host>/ws/stream/)              │
                   │  Base64 Frame + JSON Telemetry                   │
                   ▼                                                  │
        ┌───────────────────────────────────┐                         │
        │     Modern Responsive Dashboard   │                         │
        │  ┌─────────────┐ ┌──────────────┐ │                         │
        │  │ Laptop      │ │ Mobile Phone │ │                         │
        │  │ Browser     │ │ Browser      │ │                         │
        │  └─────────────┘ └──────────────┘ │                         │
        └───────────────────────────────────┘                         │
                   │                                                  │
                   └──────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
EmotiCam/
├── manage.py                   # Django CLI management executable
├── requirements.txt            # Python dependencies (Django, Channels, OpenCV, etc.)
├── README.md                   # Complete architectural and setup manual
├── .gitignore                  # Git exclusions (credentials, caches, db)
├── .env.example                # Sample environment variables template
├── .env                        # Local runtime configuration (Do not commit)
│
├── config/                     # Django project core
│   ├── __init__.py
│   ├── settings.py             # ASGI/Channels, REST framework, and pipeline settings
│   ├── urls.py                 # Global URL routing
│   ├── asgi.py                 # ASGI protocol router (HTTP + WebSockets)
│   └── wsgi.py                 # WSGI web server entrypoint
│
├── emotion_app/                # Main Django application
│   ├── __init__.py
│   ├── admin.py                # Admin interface for emotion history logs
│   ├── apps.py                 # App configuration
│   ├── models.py               # EmotionLog and CameraConfig DB schemas
│   ├── urls.py                 # API and View routing
│   ├── views.py                # Dashboard and REST API controllers
│   ├── consumers.py            # Asynchronous WebSocket streaming consumer
│   ├── routing.py              # WebSocket URL router
│   ├── serializers.py          # DRF model serializers
│   └── tests.py                # Comprehensive unit and integration test suite
│
├── ai/                         # Computer Vision & Machine Learning Subsystem
│   ├── __init__.py
│   ├── camera.py               # Robust HTTP frame client with timeout and fault recovery
│   ├── face_detector.py        # Multi-scale face detector with CLAHE contrast enhancement
│   ├── face_tracker.py         # Persistent centroid & spatial tracking (Face #1, #2)
│   ├── emotion_detector.py     # Deep CNN ONNX classifier with EMA temporal smoothing
│   ├── processor.py            # End-to-end vision pipeline orchestrator
│   └── utils.py                # HUD drawing, BGR color palettes, emoji mappings
│
├── templates/
│   └── index.html              # Responsive dark glassmorphism dashboard
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern CSS3 with micro-animations & mobile grid
│   ├── js/
│   │   └── app.js              # WebSocket telemetry handler & Chart.js engine
│   └── assets/                 # Icons & static branding assets
│
├── models/
│   ├── README.md               # Model architecture & benchmark documentation
│   └── emotion-ferplus-8.onnx  # Pre-trained deep CNN model in ONNX format
│
└── esp32/
    └── esp32_cam.ino           # Complete Arduino C++ firmware for AI-Thinker ESP32-CAM
```

---

## ⚡ Emotion Classes Supported

| Emotion | Emoji | Color Code | Description |
| :--- | :---: | :---: | :--- |
| **Happy** | 😄 | `#22c55e` (Emerald) | Smile detected, raised lip corners |
| **Neutral** | 😐 | `#3b82f6` (Cyan Blue) | Relaxed facial baseline |
| **Sad** | 😢 | `#f97316` (Amber) | Downward mouth curvature, furrowed brow |
| **Angry** | 😠 | `#ef4444` (Crimson) | Lowered eyebrows, tightened lips |
| **Surprise** | 😲 | `#eab308` (Gold) | Widened eyes, dropped jaw |
| **Fear** | 😨 | `#a855f7` (Violet) | Tense eyelids, elevated brows |
| **Disgust** | 🤢 | `#10b981` (Teal) | Wrinkled nose, raised upper lip |

---

## 🛠️ Step-by-Step Setup Guide

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12** installed on Windows.
- **Arduino IDE** (with ESP32 board support package installed).
- **AI-Thinker ESP32-CAM** module with FTDI programmer or USB-to-UART adapter.
- Local Wi-Fi network (2.4 GHz required for ESP32).

---

### 2. ESP32-CAM Flashing Instructions
1. Open Arduino IDE.
2. Go to **File > Preferences** and add to *Additional Boards Manager URLs*:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools > Board > Boards Manager**, search for `esp32` and install version **2.0.14** or later.
4. Select board: **AI Thinker ESP32-CAM**.
5. Set Board Settings:
   - **CPU Frequency:** `240MHz (WiFi/BT)`
   - **Flash Frequency:** `80MHz`
   - **Flash Mode:** `QIO`
   - **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)`
   - **PSRAM:** `Enabled`
6. Connect FTDI to ESP32-CAM:
   - `5V` -> `5V`
   - `GND` -> `GND`
   - `TX` -> `U0R` (GPIO 3)
   - `RX` -> `U0T` (GPIO 1)
   - **Connect `GPIO 0` to `GND` during flashing!**
7. Open `esp32/esp32_cam.ino`.
8. Change Wi-Fi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
9. Click **Upload**. When flashing completes, disconnect `GPIO 0` from `GND` and press the **RST** button on the ESP32-CAM.
10. Open **Serial Monitor** at **115200 baud**. Note the printed IP address:
    ```
    [SUCCESS] Wi-Fi Connected!
    [INFO] ESP32-CAM IP Address : http://192.168.1.105
    [INFO] Capture Endpoint     : http://192.168.1.105/capture
    ```

---

### 3. Backend Setup & Configuration

1. Open PowerShell or Command Prompt in the project folder:
   ```powershell
   cd "c:\Users\HRUTHWIK KHARTHIKEYA\Desktop\feels_detector"
   ```

2. Install python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Configure `.env`:
   Ensure `ESP32_CAMERA_URL` matches your ESP32-CAM's IP address:
   ```ini
   ESP32_CAMERA_URL=http://192.168.1.105
   DEBUG=True
   ALLOWED_HOSTS=*
   ```

4. Run Database Migrations:
   ```powershell
   python manage.py makemigrations
   python manage.py migrate
   ```

5. Run Unit Tests to verify everything:
   ```powershell
   python manage.py test
   ```

6. Start Django Channels ASGI Server:
   ```powershell
   python manage.py runserver 0.0.0.0:8000
   ```

---

### 4. Accessing the Dashboard

- **On your Laptop:**
  Open your browser and navigate to:
  `http://localhost:8000` or `http://127.0.0.1:8000`

- **On your Mobile Phone (Same Wi-Fi Network):**
  1. Find your laptop's local IPv4 address by running `ipconfig` in PowerShell (e.g. `192.168.1.50`).
  2. On your phone's browser (Safari/Chrome), open:
     `http://192.168.1.50:8000`
  3. The responsive layout automatically adjusts to vertical mobile layout!

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web Dashboard user interface |
| `GET` | `/api/status/` | System, AI, and Camera health status |
| `GET` | `/api/camera/status/` | Probes ESP32 `/status` endpoint and measures ping |
| `POST` | `/api/camera/config/` | Updates ESP32 IP dynamically: `{"camera_url": "http://192.168.1.x"}` |
| `GET` | `/api/emotions/` | List historical emotion logs with pagination |
| `GET` | `/api/emotions/stats/` | Aggregated distribution and percentages for charts |
| `DELETE` | `/api/emotions/clear/` | Clears historical emotion database records |
| `GET` | `/api/stream/` | Raw MJPEG video stream fallback endpoint |
| `WS` | `/ws/stream/` | Bidirectional real-time telemetry & frame WebSocket |

---

## 🌐 Local vs Cloud Deployment Guide

### Local Mode (Recommended Default)
- **Topology:** ESP32-CAM and Django Server reside on the **same private Wi-Fi subnet** (e.g. `192.168.1.x`).
- **Data Path:** `ESP32-CAM -> Local Django Server -> Browser clients`.
- **Advantages:** Zero latency penalty, no public port forwarding required, zero cloud bandwidth costs.

### Cloud Mode (When Django is hosted on AWS/Heroku/DigitalOcean)
- **Constraint:** A cloud-hosted server cannot reach private internal IP addresses (`192.168.x.x`).
- **Solution Options:**
  1. **ESP32 Push Architecture:** Flash firmware that uses HTTP POST or WebSockets to send frames from the ESP32 directly to `https://your-domain.com/api/ingest/`.
  2. **Secure Tunnel (Cloudflare Tunnel / ngrok):** Create a secure outbound tunnel from the local network to expose the ESP32 endpoint to the cloud server without exposing open router ports.

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Camera Offline` on Dashboard | ESP32 IP incorrect or not on same Wi-Fi | Click **Camera IP Config**, enter the IP printed on Serial Monitor, and click Save. |
| `Brownout detector was triggered` on ESP32 | Insufficient USB current / poor USB cable | Connect ESP32 to a 5V 2A dedicated power adapter or add a 470uF capacitor between 5V and GND. |
| WebSocket Disconnected | Firewall blocking port 8000 | In Windows Defender Firewall, allow incoming TCP connections for Python on port 8000. |
| No face detected | Low lighting or face too far | Ensure good front-facing lighting and stand within 0.5m – 2m of OV2640 lens. Adjust lens focus ring if blurred. |

---

## 📄 License & Attribution
EmotiCam is distributed under the MIT Open Source License.
Built with Django, Daphne, Channels, OpenCV DNN, and Emotion-FERPlus.
