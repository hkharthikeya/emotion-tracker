/*
  EmotiCam - ESP32-CAM Firmware
  Real-Time Facial Emotion Detection Web System
  
  Target Board: AI-Thinker ESP32-CAM
  Camera Module: OV2640
  Protocol: HTTP over Wi-Fi
  
  Endpoints:
    - GET /        -> Diagnostic status web page
    - GET /capture -> Captures and returns latest JPEG frame
    - GET /status  -> Returns JSON telemetry (device, IP, camera state, free heap, uptime)
    
  Note: No SD card, No Bluetooth, No OLED display required.
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// =============================================================================
// WI-FI CONFIGURATION
// Replace with your local Wi-Fi SSID and Password before flashing
// =============================================================================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Web server listening on port 80
WebServer server(80);

// =============================================================================
// AI-THINKER ESP32-CAM PIN DEFINITIONS (OV2640)
// =============================================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Optional onboard Flash LED (GPIO 4) and Status LED (GPIO 33)
#define FLASH_LED_PIN      4
#define STATUS_LED_PIN    33

// =============================================================================
// HANDLER: GET / (Root Diagnostic Page)
// =============================================================================
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<title>EmotiCam ESP32-CAM Node</title>";
  html += "<style>body{font-family:sans-serif;background:#0d1117;color:#e6edf3;padding:2rem;text-align:center;}";
  html += ".card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:2rem;max-width:500px;margin:auto;box-shadow:0 8px 24px rgba(0,0,0,0.5);}";
  html += "h1{color:#58a6ff;margin-bottom:0.5rem;} .badge{background:#238636;color:white;padding:4px 12px;border-radius:20px;font-size:0.9rem;}";
  html += "a{color:#58a6ff;text-decoration:none;display:inline-block;margin-top:1rem;background:#21262d;padding:8px 16px;border-radius:6px;border:1px solid #30363d;}";
  html += "table{width:100%;margin-top:1.5rem;text-align:left;border-collapse:collapse;} td{padding:8px;border-bottom:1px solid #21262d;}</style></head><body>";
  html += "<div class='card'>";
  html += "<h1>EmotiCam Node</h1>";
  html += "<p><span class='badge'>CAMERA ONLINE</span></p>";
  html += "<table>";
  html += "<tr><td><b>Device:</b></td><td>AI-Thinker ESP32-CAM</td></tr>";
  html += "<tr><td><b>Sensor:</b></td><td>OmniVision OV2640</td></tr>";
  html += "<tr><td><b>IP Address:</b></td><td>" + WiFi.localIP().toString() + "</td></tr>";
  html += "<tr><td><b>Wi-Fi SSID:</b></td><td>" + String(WiFi.SSID()) + " (" + String(WiFi.RSSI()) + " dBm)</td></tr>";
  html += "<tr><td><b>Free Heap:</b></td><td>" + String(ESP.getFreeHeap()) + " bytes</td></tr>";
  html += "<tr><td><b>Uptime:</b></td><td>" + String(millis() / 1000) + "s</td></tr>";
  html += "</table>";
  html += "<a href='/capture' target='_blank'>Preview Single /capture Frame</a> ";
  html += "<a href='/status' target='_blank'>View /status JSON</a>";
  html += "</div></body></html>";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "text/html", html);
}

// =============================================================================
// HANDLER: GET /status (JSON Status Telemetry)
// =============================================================================
void handleStatus() {
  String json = "{";
  json += "\"device\":\"ESP32-CAM\",";
  json += "\"sensor\":\"OV2640\",";
  json += "\"camera\":\"online\",";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
  json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"free_heap\":" + String(ESP.getFreeHeap()) + ",";
  json += "\"uptime_ms\":" + String(millis());
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Content-Type", "application/json");
  server.send(200, "application/json", json);
}

// =============================================================================
// HANDLER: GET /capture (Real-Time JPEG Frame Capture)
// =============================================================================
void handleCapture() {
  camera_fb_t * fb = NULL;
  
  // Flash status LED indicator during capture
  digitalWrite(STATUS_LED_PIN, LOW); // Active LOW on ESP32-CAM
  
  // Capture frame from camera sensor
  fb = esp_camera_fb_get();
  
  digitalWrite(STATUS_LED_PIN, HIGH); // Turn off status LED

  if (!fb) {
    Serial.println("[ERROR] Camera capture failed!");
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  // Set HTTP headers for binary JPEG streaming
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.sendHeader("Pragma", "no-cache");
  server.sendHeader("Expires", "0");
  
  // Send the JPEG image buffer directly to client
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  
  // Return buffer back to camera driver pool (Crucial to prevent memory leaks!)
  esp_camera_fb_return(fb);
}

// =============================================================================
// INITIALIZE CAMERA HARDWARE
// =============================================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Frame size & quality configuration
  // PSRAM check allows higher resolution and double buffering
  if (psramFound()) {
    Serial.println("[INFO] PSRAM detected. Configuring VGA resolution (640x480).");
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 12;           // 10-63 (lower means higher quality)
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    Serial.println("[WARN] No PSRAM found. Falling back to QVGA (320x240).");
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 14;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[FATAL] Camera init failed with error 0x%x\n", err);
    return false;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    // Sensor adjustments for optimal face contrast
    s->set_brightness(s, 1);     // -2 to 2
    s->set_contrast(s, 1);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
    s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
    s->set_wb_mode(s, 0);        // 0 to 4 - 0: Auto, 1: Sunny, 2: Cloudy, 3: Office, 4: Home
  }

  Serial.println("[INFO] Camera hardware initialized successfully.");
  return true;
}

// =============================================================================
// SETUP
// =============================================================================
void setup() {
  // Disable brownout detector to prevent resets during high Wi-Fi/sensor current spikes
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  // Initialize Serial Monitor
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  delay(1000);

  Serial.println();
  Serial.println("==================================================");
  Serial.println("   EmotiCam - ESP32-CAM AI Node Starting Up...   ");
  Serial.println("==================================================");

  // Configure Status LED
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH); // Off

  // Initialize Camera
  if (!initCamera()) {
    Serial.println("[FATAL] Halting due to camera initialization error.");
    while (true) {
      digitalWrite(STATUS_LED_PIN, LOW);
      delay(200);
      digitalWrite(STATUS_LED_PIN, HIGH);
      delay(200);
    }
  }

  // Connect to Wi-Fi
  Serial.printf("[INFO] Connecting to Wi-Fi: %s", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("[SUCCESS] Wi-Fi Connected!");
    Serial.printf("[INFO] ESP32-CAM IP Address : http://%s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[INFO] Capture Endpoint     : http://%s/capture\n", WiFi.localIP().toString().c_str());
    Serial.printf("[INFO] Status Endpoint      : http://%s/status\n", WiFi.localIP().toString().c_str());
    Serial.println("==================================================");
  } else {
    Serial.println("[ERROR] Wi-Fi connection timed out. Please check SSID/Password.");
  }

  // Register Web Endpoints
  server.on("/", HTTP_GET, handleRoot);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/status", HTTP_GET, handleStatus);

  // Start HTTP Server
  server.begin();
  Serial.println("[INFO] HTTP REST Server started.");
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
  server.handleClient();
  
  // Auto-reconnect Wi-Fi if connection drops
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WARN] Wi-Fi connection lost. Attempting reconnection...");
    WiFi.reconnect();
    delay(2000);
  }
}
