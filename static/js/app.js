/**
 * EmotiCam Frontend Real-Time Orchestrator
 * Connects over WebSocket to Django Channels, updates UI/HUD, manages Chart.js analytics.
 */

document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // DOM Elements
    // =========================================================================
    const liveVideoFeed = document.getElementById('liveVideoFeed');
    const videoOverlayMessage = document.getElementById('videoOverlayMessage');
    const overlayMsgText = document.getElementById('overlayMsgText');

    // Header Status Pills & Theme Toggle
    const cameraStatusPill = document.getElementById('cameraStatusPill');
    const cameraStatusText = document.getElementById('cameraStatusText');
    const aiStatusPill = document.getElementById('aiStatusPill');
    const aiStatusText = document.getElementById('aiStatusText');
    const wsStatusPill = document.getElementById('wsStatusPill');
    const wsStatusText = document.getElementById('wsStatusText');
    const targetIpBadge = document.getElementById('targetIpBadge');
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeIcon = document.getElementById('themeIcon');
    const themeLabel = document.getElementById('themeLabel');
    const themeBtnLight = document.getElementById('themeBtnLight');
    const themeBtnDark = document.getElementById('themeBtnDark');

    // Viewport HUD Stats
    const hudFps = document.getElementById('hudFps');
    const hudLatency = document.getElementById('hudLatency');
    const hudFaces = document.getElementById('hudFaces');

    // Primary Emotion Hero
    const emotionHeroCard = document.getElementById('emotionHeroCard');
    const primaryEmoji = document.getElementById('primaryEmoji');
    const primaryEmotionName = document.getElementById('primaryEmotionName');
    const primaryConfidenceVal = document.getElementById('primaryConfidenceVal');
    const primaryConfidenceBar = document.getElementById('primaryConfidenceBar');
    const faceDetectedTag = document.getElementById('faceDetectedTag');

    // Metrics Box
    const metricFaceCount = document.getElementById('metricFaceCount');
    const metricFps = document.getElementById('metricFps');
    const metricLatency = document.getElementById('metricLatency');

    // Controls
    const toggleStreamBtn = document.getElementById('toggleStreamBtn');
    const toggleStreamBtnText = document.getElementById('toggleStreamBtnText');
    const reconnectCameraBtn = document.getElementById('reconnectCameraBtn');
    const configCameraBtn = document.getElementById('configCameraBtn');
    const ipConfigDrawer = document.getElementById('ipConfigDrawer');
    const cameraIpInput = document.getElementById('cameraIpInput');
    const saveIpBtn = document.getElementById('saveIpBtn');
    const presetWebcamBtn = document.getElementById('presetWebcamBtn');
    const presetEsp32Btn = document.getElementById('presetEsp32Btn');
    const snapshotBtn = document.getElementById('snapshotBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const quickWebcamBtn = document.getElementById('quickWebcamBtn');

    // Source selector tabs
    const tabBrowserCam = document.getElementById('tabBrowserCam');
    const tabServerCam  = document.getElementById('tabServerCam');
    const tabEsp32      = document.getElementById('tabEsp32');

    // Browser webcam elements
    const browserWebcamVideo  = document.getElementById('browserWebcamVideo');
    const browserWebcamCanvas = document.getElementById('browserWebcamCanvas');

    // History Table
    const historyTableBody = document.getElementById('historyTableBody');

    // =========================================================================
    // State Variables
    // =========================================================================
    let ws = null;
    let isProcessing = true;
    let lastEmotion = '';
    let reconnectTimeout = null;

    // Source mode: 'esp32' | 'server_webcam' | 'browser_webcam'
    let sourceMode = 'esp32';
    let browserCamStream = null;     // MediaStream from getUserMedia
    let browserCamInterval = null;   // setInterval handle for frame capture

    let emotionCounts = {
        Happy: 0,
        Neutral: 0,
        Surprise: 0,
        Sad: 0,
        Angry: 0,
        Fear: 0,
        Disgust: 0
    };

    // Emotion Color Mapping for Chart & Cards
    const EMOTION_THEME_COLORS = {
        Happy: '#22c55e',
        Neutral: '#3b82f6',
        Surprise: '#eab308',
        Sad: '#f97316',
        Angry: '#ef4444',
        Fear: '#a855f7',
        Disgust: '#10b981',
        Unknown: '#64748b'
    };

    const EMOTION_EMOJIS = {
        Happy: '😄',
        Neutral: '😐',
        Surprise: '😲',
        Sad: '😢',
        Angry: '😠',
        Fear: '😨',
        Disgust: '🤢',
        Unknown: '❓'
    };

    // =========================================================================
    // Chart.js Setup
    // =========================================================================
    const ctx = document.getElementById('emotionDistributionChart').getContext('2d');
    const distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Happy', 'Neutral', 'Surprise', 'Sad', 'Angry', 'Fear', 'Disgust'],
            datasets: [{
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: [
                    EMOTION_THEME_COLORS.Happy,
                    EMOTION_THEME_COLORS.Neutral,
                    EMOTION_THEME_COLORS.Surprise,
                    EMOTION_THEME_COLORS.Sad,
                    EMOTION_THEME_COLORS.Angry,
                    EMOTION_THEME_COLORS.Fear,
                    EMOTION_THEME_COLORS.Disgust
                ],
                borderWidth: 2,
                borderColor: '#111827'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        padding: 12
                    }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            },
            cutout: '68%'
        }
    });

    // =========================================================================
    // Theme Manager (Light & Dark Mode)
    // =========================================================================
    function applyTheme(theme) {
        if (theme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
            if (themeBtnLight) {
                themeBtnLight.classList.add('active');
                themeBtnLight.setAttribute('aria-pressed', 'true');
            }
            if (themeBtnDark) {
                themeBtnDark.classList.remove('active');
                themeBtnDark.setAttribute('aria-pressed', 'false');
            }
            if (themeIcon) themeIcon.className = 'fa-solid fa-moon';
            if (themeLabel) themeLabel.textContent = 'Dark';
            if (themeToggleBtn) {
                themeToggleBtn.title = 'Switch to Dark Mode';
                themeToggleBtn.style.color = '#334155';
            }
            // Update chart styling for crisp light mode
            distributionChart.data.datasets[0].borderColor = '#ffffff';
            distributionChart.options.plugins.legend.labels.color = '#475569';
            distributionChart.options.plugins.tooltip.backgroundColor = '#ffffff';
            distributionChart.options.plugins.tooltip.titleColor = '#0f172a';
            distributionChart.options.plugins.tooltip.bodyColor = '#334155';
            distributionChart.options.plugins.tooltip.borderColor = 'rgba(0,0,0,0.1)';
            distributionChart.update();
        } else {
            document.documentElement.removeAttribute('data-theme');
            if (themeBtnDark) {
                themeBtnDark.classList.add('active');
                themeBtnDark.setAttribute('aria-pressed', 'true');
            }
            if (themeBtnLight) {
                themeBtnLight.classList.remove('active');
                themeBtnLight.setAttribute('aria-pressed', 'false');
            }
            if (themeIcon) themeIcon.className = 'fa-solid fa-sun';
            if (themeLabel) themeLabel.textContent = 'Light';
            if (themeToggleBtn) {
                themeToggleBtn.title = 'Switch to Light Mode';
                themeToggleBtn.style.color = '#facc15';
            }
            // Update chart styling for dark mode
            distributionChart.data.datasets[0].borderColor = '#111827';
            distributionChart.options.plugins.legend.labels.color = '#94a3b8';
            distributionChart.options.plugins.tooltip.backgroundColor = '#1e293b';
            distributionChart.options.plugins.tooltip.titleColor = '#f8fafc';
            distributionChart.options.plugins.tooltip.bodyColor = '#cbd5e1';
            distributionChart.options.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
            distributionChart.update();
        }
        localStorage.setItem('emoticam_theme', theme);
    }

    function initTheme() {
        const savedTheme = localStorage.getItem('emoticam_theme');
        if (savedTheme) {
            applyTheme(savedTheme);
        } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            applyTheme('light');
        } else {
            applyTheme('dark');
        }
    }

    if (themeBtnLight) {
        themeBtnLight.addEventListener('click', () => applyTheme('light'));
    }

    if (themeBtnDark) {
        themeBtnDark.addEventListener('click', () => applyTheme('dark'));
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
            applyTheme(currentTheme === 'light' ? 'dark' : 'light');
        });
    }

    // Initialize Theme immediately
    initTheme();

    // =========================================================================
    // WebSocket Connection Manager
    // =========================================================================
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream/`;

        wsStatusPill.className = 'status-pill connecting';
        wsStatusText.textContent = 'CONNECTING';

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('[WS] Connected to EmotiCam server.');
            wsStatusPill.className = 'status-pill online';
            wsStatusText.textContent = 'ONLINE';
            
            // Sync button state
            isProcessing = true;
            toggleStreamBtn.className = 'btn btn-primary';
            toggleStreamBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span id="toggleStreamBtnText">PAUSE STREAM</span>';

            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
                reconnectTimeout = null;
            }
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerMessage(data);
            } catch (err) {
                console.error('[WS] Parse error:', err);
            }
        };

        ws.onclose = (event) => {
            console.warn('[WS] Disconnected from server. Code:', event.code);
            wsStatusPill.className = 'status-pill offline';
            wsStatusText.textContent = 'DISCONNECTED';
            
            // Auto-reconnect attempt after 3s
            if (!reconnectTimeout) {
                reconnectTimeout = setTimeout(() => {
                    console.log('[WS] Attempting reconnection...');
                    initWebSocket();
                }, 3000);
            }
        };

        ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            wsStatusPill.className = 'status-pill offline';
            wsStatusText.textContent = 'ERROR';
        };
    }

    // =========================================================================
    // Message Handler & UI Updaters
    // =========================================================================
    function handleServerMessage(data) {
        if (data.type === 'frame') {
            updateDashboard(data);
        } else if (data.type === 'status_update') {
            if (data.camera_status) updateCameraStatusPill(data.camera_status);
            if (data.ai_status) updateAiStatusPill(data.ai_status);
        }
    }

    function updateDashboard(data) {
        // 1. Update Video Frame
        if (data.image) {
            liveVideoFeed.src = data.image;
        }

        // 2. Camera Status
        updateCameraStatusPill(data.camera_status);

        // 3. AI Status
        updateAiStatusPill(data.ai_status);

        // 4. Update HUD Performance Badges
        hudFps.textContent = (data.fps || 0.0).toFixed(1);
        hudLatency.textContent = `${Math.round(data.latency_ms || 0)} ms`;
        hudFaces.textContent = data.face_count || 0;

        // 5. Update Metrics Grid
        metricFaceCount.textContent = data.face_count || 0;
        metricFps.textContent = (data.fps || 0.0).toFixed(1);
        metricLatency.textContent = `${Math.round(data.latency_ms || 0)} ms`;

        // 6. Update Primary Emotion Display
        if (data.face_count > 0 && data.primary_emotion && data.primary_emotion !== 'No Face') {
            faceDetectedTag.className = 'badge badge-accent';
            faceDetectedTag.textContent = `${data.face_count} FACE${data.face_count > 1 ? 'S' : ''} DETECTED`;

            primaryEmotionName.textContent = data.primary_emotion.toUpperCase();
            primaryConfidenceVal.textContent = `${Math.round(data.confidence || 0)}%`;
            primaryConfidenceBar.style.width = `${Math.min(100, Math.max(0, data.confidence || 0))}%`;

            const currentEmoji = data.primary_emoji || EMOTION_EMOJIS[data.primary_emotion] || '😐';
            if (lastEmotion !== data.primary_emotion) {
                primaryEmoji.textContent = currentEmoji;
                primaryEmoji.classList.remove('bounce');
                void primaryEmoji.offsetWidth; // Trigger reflow
                primaryEmoji.classList.add('bounce');
                
                // Update card glow
                emotionHeroCard.className = `emotion-hero-card glow-${data.primary_emotion.toLowerCase()}`;
                
                // Add event to history table & chart
                if (data.confidence > 45) {
                    addHistoryRow(data.faces[0]?.track_id || 1, data.primary_emotion, data.confidence, data.latency_ms);
                    recordEmotionForChart(data.primary_emotion);
                }
                lastEmotion = data.primary_emotion;
            }
        } else {
            faceDetectedTag.className = 'badge badge-neutral';
            faceDetectedTag.textContent = 'SEARCHING FACE';
            primaryEmotionName.textContent = 'NO FACE DETECTED';
            primaryConfidenceVal.textContent = '0%';
            primaryConfidenceBar.style.width = '0%';
            primaryEmoji.textContent = '🔍';
            emotionHeroCard.className = 'emotion-hero-card';
            lastEmotion = 'No Face';
        }

        // 7. Update Real-Time Probability Breakdown Bars
        if (data.distribution) {
            for (const [emotion, prob] of Object.entries(data.distribution)) {
                const bar = document.getElementById(`bar_${emotion}`);
                const val = document.getElementById(`val_${emotion}`);
                if (bar && val) {
                    const pct = Math.round(prob);
                    bar.style.width = `${pct}%`;
                    val.textContent = `${pct}%`;
                }
            }
        }
    }

    function updateCameraStatusPill(status) {
        if (!status) return;
        cameraStatusText.textContent = status.toUpperCase();
        if (status === 'ONLINE') {
            cameraStatusPill.className = 'status-pill online';
            videoOverlayMessage.classList.add('hidden');
        } else if (status === 'CONNECTING') {
            cameraStatusPill.className = 'status-pill standby';
            videoOverlayMessage.classList.remove('hidden');
            overlayMsgText.textContent = 'Connecting to ESP32-CAM...';
        } else {
            cameraStatusPill.className = 'status-pill offline';
            videoOverlayMessage.classList.remove('hidden');
            overlayMsgText.textContent = 'Camera Offline. Check Wi-Fi & ESP32 IP.';
        }
    }

    function updateAiStatusPill(status) {
        if (!status) return;
        aiStatusText.textContent = status.toUpperCase();
        if (status === 'PROCESSING' || status === 'ONLINE' || status === 'ACTIVE') {
            aiStatusPill.className = 'status-pill online';
        } else if (status === 'STANDBY') {
            aiStatusPill.className = 'status-pill standby';
        } else {
            aiStatusPill.className = 'status-pill offline';
        }
    }

    // =========================================================================
    // History & Chart Tracking
    // =========================================================================
    function addHistoryRow(trackId, emotion, confidence, latency) {
        const emptyRow = historyTableBody.querySelector('.empty-row');
        if (emptyRow) {
            emptyRow.remove();
        }

        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
        const emoji = EMOTION_EMOJIS[emotion] || '😐';
        const badgeClass = `badge-${emotion.toLowerCase()}`;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><code>${timeStr}</code></td>
            <td><strong>Face #${trackId}</strong></td>
            <td><span class="emotion-table-badge ${badgeClass}">${emoji} ${emotion}</span></td>
            <td><strong>${Math.round(confidence)}%</strong></td>
            <td><small>${Math.round(latency)} ms</small></td>
        `;

        // Prepend to top of table
        historyTableBody.insertBefore(tr, historyTableBody.firstChild);

        // Keep maximum 30 rows in DOM
        if (historyTableBody.children.length > 30) {
            historyTableBody.lastElementChild.remove();
        }
    }

    function recordEmotionForChart(emotion) {
        if (emotionCounts.hasOwnProperty(emotion)) {
            emotionCounts[emotion]++;
            updateChartData();
        }
    }

    function updateChartData() {
        const labels = ['Happy', 'Neutral', 'Surprise', 'Sad', 'Angry', 'Fear', 'Disgust'];
        distributionChart.data.datasets[0].data = labels.map(e => emotionCounts[e]);
        distributionChart.update();
    }

    // =========================================================================
    // Initial Stats Load via REST API
    // =========================================================================
    async function loadInitialStats() {
        try {
            const resp = await fetch('/api/emotions/stats/');
            if (resp.ok) {
                const data = await resp.json();
                if (data.counts) {
                    for (const [k, v] of Object.entries(data.counts)) {
                        if (emotionCounts.hasOwnProperty(k)) {
                            emotionCounts[k] = v;
                        }
                    }
                    updateChartData();
                }
            }
        } catch (e) {
            console.error('Failed to load initial emotion stats:', e);
        }

        try {
            const historyResp = await fetch('/api/emotions/?limit=10');
            if (historyResp.ok) {
                const histData = await historyResp.json();
                if (histData.results && histData.results.length > 0) {
                    const emptyRow = historyTableBody.querySelector('.empty-row');
                    if (emptyRow) emptyRow.remove();

                    histData.results.forEach(item => {
                        const tr = document.createElement('tr');
                        const emoji = EMOTION_EMOJIS[item.emotion] || '😐';
                        const badgeClass = `badge-${item.emotion.toLowerCase()}`;
                        tr.innerHTML = `
                            <td><code>${item.timestamp_formatted}</code></td>
                            <td><strong>Face #${item.face_id}</strong></td>
                            <td><span class="emotion-table-badge ${badgeClass}">${emoji} ${item.emotion}</span></td>
                            <td><strong>${Math.round(item.confidence)}%</strong></td>
                            <td><small>${Math.round(item.latency_ms)} ms</small></td>
                        `;
                        historyTableBody.appendChild(tr);
                    });
                }
            }
        } catch (e) {
            console.error('Failed to load initial history logs:', e);
        }
    }

    // =========================================================================
    // Event Listeners & Dashboard Controls
    // =========================================================================
    toggleStreamBtn.addEventListener('click', () => {
        isProcessing = !isProcessing;
        if (isProcessing) {
            toggleStreamBtn.className = 'btn btn-primary';
            toggleStreamBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>STOP PROCESSING</span>';
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'start_processing' }));
            }
        } else {
            toggleStreamBtn.className = 'btn btn-secondary';
            toggleStreamBtn.innerHTML = '<i class="fa-solid fa-play"></i> <span>START PROCESSING</span>';
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'stop_processing' }));
            }
            updateAiStatusPill('STOPPED');
        }
    });

    async function switchSource(newSource) {
        try {
            const resp = await fetch('/api/camera/config/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camera_url: newSource })
            });

            if (resp.ok) {
                const data = await resp.json();
                targetIpBadge.textContent = data.camera_url === '0' || data.camera_url === 'webcam' ? 'Laptop Webcam (0)' : data.camera_url;
                cameraIpInput.value = data.camera_url;
                ipConfigDrawer.classList.add('hidden');
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ command: 'set_camera_url', url: data.camera_url }));
                }

                if (quickWebcamBtn) {
                    if (data.camera_url === '0' || data.camera_url === 'webcam') {
                        quickWebcamBtn.innerHTML = '<i class="fa-solid fa-camera"></i> <span>Switch to ESP32-CAM</span>';
                    } else {
                        quickWebcamBtn.innerHTML = '<i class="fa-solid fa-laptop"></i> <span>Use Laptop Webcam</span>';
                    }
                }
            }
        } catch (e) {
            alert('Failed to update video source: ' + e);
        }
    }

    // =========================================================================
    // Browser Webcam (getUserMedia) — capture & send over WebSocket
    // =========================================================================
    async function startBrowserWebcam() {
        try {
            browserCamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false
            });
            browserWebcamVideo.srcObject = browserCamStream;
            await browserWebcamVideo.play();

            // Update UI
            updateCameraStatusPill('ONLINE');
            targetIpBadge.textContent = 'Browser Webcam 📸';

            // Capture and send frames at ~15 FPS
            browserCamInterval = setInterval(() => {
                if (!ws || ws.readyState !== WebSocket.OPEN || !isProcessing) return;
                const vw = browserWebcamVideo.videoWidth;
                const vh = browserWebcamVideo.videoHeight;
                if (!vw || !vh) return;

                browserWebcamCanvas.width  = vw;
                browserWebcamCanvas.height = vh;
                const ctx2d = browserWebcamCanvas.getContext('2d');
                ctx2d.drawImage(browserWebcamVideo, 0, 0, vw, vh);
                const frameDataUrl = browserWebcamCanvas.toDataURL('image/jpeg', 0.75);

                ws.send(JSON.stringify({
                    command: 'webcam_frame',
                    frame: frameDataUrl
                }));
            }, 67); // ~15 FPS

        } catch (err) {
            console.error('[BrowserCam] getUserMedia failed:', err);
            updateCameraStatusPill('OFFLINE');
            overlayMsgText.textContent = 'Camera permission denied. Please allow browser webcam access.';
            videoOverlayMessage.classList.remove('hidden');
            // Fall back to ESP32 tab
            setSourceTab('esp32');
        }
    }

    function stopBrowserWebcam() {
        if (browserCamInterval) {
            clearInterval(browserCamInterval);
            browserCamInterval = null;
        }
        if (browserCamStream) {
            browserCamStream.getTracks().forEach(t => t.stop());
            browserCamStream = null;
        }
        browserWebcamVideo.srcObject = null;
    }

    // =========================================================================
    // Source Tab Switcher
    // =========================================================================
    function setSourceTab(mode) {
        sourceMode = mode;
        [tabBrowserCam, tabServerCam, tabEsp32].forEach(t => t && t.classList.remove('active'));

        stopBrowserWebcam();

        if (mode === 'browser_webcam') {
            tabBrowserCam && tabBrowserCam.classList.add('active');
            // Release server-side OpenCV camera to avoid Windows exclusive lock conflict
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'release_camera' }));
                ws.send(JSON.stringify({ command: 'stop_processing' }));
            }
            startBrowserWebcam();
            isProcessing = true;
            toggleStreamBtn.className = 'btn btn-primary';
            toggleStreamBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>PAUSE STREAM</span>';

        } else if (mode === 'server_webcam') {
            tabServerCam && tabServerCam.classList.add('active');
            switchSource('0');
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'start_processing' }));
            }
            isProcessing = true;
            toggleStreamBtn.className = 'btn btn-primary';
            toggleStreamBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>PAUSE STREAM</span>';

        } else {
            tabEsp32 && tabEsp32.classList.add('active');
            const ip = cameraIpInput ? cameraIpInput.value.trim() : 'http://192.168.1.100';
            switchSource(ip || 'http://192.168.1.100');
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'start_processing' }));
            }
            isProcessing = true;
            toggleStreamBtn.className = 'btn btn-primary';
            toggleStreamBtn.innerHTML = '<i class="fa-solid fa-pause"></i> <span>PAUSE STREAM</span>';
        }
    }

    tabBrowserCam && tabBrowserCam.addEventListener('click', () => setSourceTab('browser_webcam'));
    tabServerCam  && tabServerCam.addEventListener('click',  () => setSourceTab('server_webcam'));
    tabEsp32      && tabEsp32.addEventListener('click',      () => setSourceTab('esp32'));

    if (quickWebcamBtn) {
        quickWebcamBtn.addEventListener('click', () => {
            const isCurrentlyWebcam = targetIpBadge.textContent.includes('Webcam') || (cameraIpInput && (cameraIpInput.value === '0' || cameraIpInput.value === 'webcam'));
            if (isCurrentlyWebcam) {
                setSourceTab('esp32');
            } else {
                setSourceTab('server_webcam');
            }
        });
    }

    if (presetWebcamBtn) {
        presetWebcamBtn.addEventListener('click', () => {
            cameraIpInput.value = '0';
        });
    }

    if (presetEsp32Btn) {
        presetEsp32Btn.addEventListener('click', () => {
            cameraIpInput.value = 'http://192.168.1.100';
        });
    }

    reconnectCameraBtn.addEventListener('click', () => {
        if (sourceMode === 'browser_webcam') {
            // Restart browser webcam
            stopBrowserWebcam();
            startBrowserWebcam();
            updateCameraStatusPill('CONNECTING');
        } else {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'reconnect_camera' }));
            }
            updateCameraStatusPill('CONNECTING');
        }
    });

    configCameraBtn.addEventListener('click', () => {
        ipConfigDrawer.classList.toggle('hidden');
    });

    saveIpBtn.addEventListener('click', () => {
        const newIp = cameraIpInput.value.trim();
        if (newIp) {
            switchSource(newIp);
        }
    });

    snapshotBtn.addEventListener('click', () => {
        if (!liveVideoFeed.src) return;
        const a = document.createElement('a');
        a.href = liveVideoFeed.src;
        a.download = `emoticam_capture_${Date.now()}.jpg`;
        a.click();
    });

    clearHistoryBtn.addEventListener('click', async () => {
        if (confirm('Clear all emotion logs and reset session chart?')) {
            try {
                await fetch('/api/emotions/clear/', { method: 'DELETE' });
                historyTableBody.innerHTML = `
                    <tr class="empty-row">
                        <td colspan="5">No emotion events logged yet. Stand in front of camera to begin.</td>
                    </tr>
                `;
                for (const k in emotionCounts) emotionCounts[k] = 0;
                updateChartData();
            } catch (e) {
                console.error('Failed to clear history:', e);
            }
        }
    });

    // =========================================================================
    // Multimodal NLP Analysis Controls
    // =========================================================================
    const nlpTextInput = document.getElementById('nlpTextInput');
    const analyzeNlpBtn = document.getElementById('analyzeNlpBtn');
    const clearNlpBtn = document.getElementById('clearNlpBtn');
    const nlpEmoji = document.getElementById('nlpEmoji');
    const nlpEmotionName = document.getElementById('nlpEmotionName');
    const nlpConfidenceVal = document.getElementById('nlpConfidenceVal');
    const multimodalEmoji = document.getElementById('multimodalEmoji');
    const multimodalEmotionName = document.getElementById('multimodalEmotionName');
    const multimodalConfidenceVal = document.getElementById('multimodalConfidenceVal');
    const congruenceBadge = document.getElementById('congruenceBadge');

    async function executeMultimodalAnalysis(text) {
        if (!text || !text.trim()) {
            nlpEmoji.textContent = '💬';
            nlpEmotionName.textContent = 'AWAITING TEXT';
            nlpConfidenceVal.textContent = '0%';
            multimodalEmoji.textContent = '🔮';
            multimodalEmotionName.textContent = 'STANDBY';
            multimodalConfidenceVal.textContent = '0%';
            congruenceBadge.className = 'badge badge-neutral';
            congruenceBadge.textContent = 'Congruency: --';
            return;
        }

        try {
            analyzeNlpBtn.disabled = true;
            analyzeNlpBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

            const resp = await fetch('/api/multimodal/fuse/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text.trim(), face_weight: 0.6 })
            });

            if (resp.ok) {
                const data = await resp.json();
                const nlp = data.nlp_analysis;
                const mm = data.multimodal;

                // Update NLP Card
                const nlpDom = nlp.dominant_emotion;
                nlpEmoji.textContent = EMOTION_EMOJIS[nlpDom] || '💬';
                nlpEmotionName.textContent = nlpDom.toUpperCase();
                nlpConfidenceVal.textContent = `${Math.round(nlp.confidence)}%`;

                // Update Multimodal Card
                const mmDom = mm.multimodal_dominant_emotion;
                multimodalEmoji.textContent = EMOTION_EMOJIS[mmDom] || '🔮';
                multimodalEmotionName.textContent = mmDom.toUpperCase();
                multimodalConfidenceVal.textContent = `${Math.round(mm.multimodal_confidence)}%`;

                if (mm.is_congruent) {
                    congruenceBadge.className = 'badge badge-accent';
                    congruenceBadge.textContent = 'Congruent (Face + Text Match)';
                } else {
                    congruenceBadge.className = 'badge badge-neutral';
                    congruenceBadge.textContent = `Mixed (Face: ${mm.face_dominant_emotion} | Text: ${mm.text_dominant_emotion})`;
                }
            }
        } catch (e) {
            console.error('Multimodal analysis failed:', e);
        } finally {
            analyzeNlpBtn.disabled = false;
            analyzeNlpBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze & Fuse Emotion';
        }
    }

    if (analyzeNlpBtn && nlpTextInput) {
        analyzeNlpBtn.addEventListener('click', () => {
            executeMultimodalAnalysis(nlpTextInput.value);
        });

        // Trigger analysis on Enter key (Ctrl+Enter or Enter without Shift)
        nlpTextInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                executeMultimodalAnalysis(nlpTextInput.value);
            }
        });
    }

    if (clearNlpBtn && nlpTextInput) {
        clearNlpBtn.addEventListener('click', () => {
            nlpTextInput.value = '';
            executeMultimodalAnalysis('');
        });
    }

    // Quick Preset Chips
    document.querySelectorAll('.preset-chips .chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const sampleText = chip.getAttribute('data-sample');
            if (sampleText && nlpTextInput) {
                nlpTextInput.value = sampleText;
                executeMultimodalAnalysis(sampleText);
            }
        });
    });

    // =========================================================================
    // Initialization
    // =========================================================================
    function initSourceSelection() {
        const rawBadge = targetIpBadge ? targetIpBadge.textContent.trim() : '';
        const rawInput = cameraIpInput ? cameraIpInput.value.trim() : '';
        const currentSource = rawBadge || rawInput || '0';
        
        if (currentSource === '0' || currentSource.toLowerCase().includes('webcam')) {
            sourceMode = 'server_webcam';
            [tabBrowserCam, tabServerCam, tabEsp32].forEach(t => t && t.classList.remove('active'));
            if (tabServerCam) tabServerCam.classList.add('active');
            if (targetIpBadge) targetIpBadge.textContent = 'Laptop Webcam (0)';
        } else {
            sourceMode = 'esp32';
            [tabBrowserCam, tabServerCam, tabEsp32].forEach(t => t && t.classList.remove('active'));
            if (tabEsp32) tabEsp32.classList.add('active');
        }
    }

    initSourceSelection();
    initWebSocket();
    loadInitialStats();
});
