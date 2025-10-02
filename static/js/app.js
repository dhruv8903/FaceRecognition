// Youth Mood Detection AI - JavaScript

class MoodDetectionApp {
    constructor() {
        this.currentStream = null;
        this.currentTab = 'text';
        this.combinedStream = null;
        this.capturedImage = null;
        this.combinedCapturedImage = null;
        
        this.init();
    }

    init() {
        this.setupTabNavigation();
        this.setupTextAnalysis();
        this.setupFacialAnalysis();
        this.setupCombinedAnalysis();
    }

    // Tab Navigation
    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const targetTab = e.target.dataset.tab;
                
                // Remove active class from all tabs and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                e.target.classList.add('active');
                document.getElementById(`${targetTab}-tab`).classList.add('active');
                
                this.currentTab = targetTab;
            });
        });
    }

    // Text Analysis Setup
    setupTextAnalysis() {
        const analyzeBtn = document.getElementById('analyzeTextBtn');
        const textInput = document.getElementById('textInput');
        const resultsDiv = document.getElementById('textResults');

        analyzeBtn.addEventListener('click', () => {
            const text = textInput.value.trim();
            if (!text) {
                this.showError(resultsDiv, 'Please enter some text to analyze.');
                return;
            }
            
            this.analyzeText(text, resultsDiv);
        });

        // Allow Enter key to trigger analysis
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyzeBtn.click();
            }
        });
    }

    // Facial Analysis Setup
    setupFacialAnalysis() {
        const startBtn = document.getElementById('startCameraBtn');
        const captureBtn = document.getElementById('captureBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        const video = document.getElementById('videoElement');
        const canvas = document.getElementById('canvasElement');
        const resultsDiv = document.getElementById('facialResults');

        startBtn.addEventListener('click', () => {
            this.startCamera(video, startBtn, captureBtn, stopBtn);
        });

        captureBtn.addEventListener('click', () => {
            this.captureAndAnalyze(video, canvas, resultsDiv);
        });

        stopBtn.addEventListener('click', () => {
            this.stopCamera(video, startBtn, captureBtn, stopBtn);
        });
    }

    // Combined Analysis Setup
    setupCombinedAnalysis() {
        const startBtn = document.getElementById('startCombinedCameraBtn');
        const captureBtn = document.getElementById('captureCombinedBtn');
        const stopBtn = document.getElementById('stopCombinedCameraBtn');
        const analyzeBtn = document.getElementById('analyzeCombinedBtn');
        const video = document.getElementById('combinedVideoElement');
        const canvas = document.getElementById('combinedCanvasElement');
        const textInput = document.getElementById('combinedTextInput');
        const resultsDiv = document.getElementById('combinedResults');

        startBtn.addEventListener('click', () => {
            this.startCombinedCamera(video, startBtn, captureBtn, stopBtn);
        });

        captureBtn.addEventListener('click', () => {
            this.captureForCombined(video, canvas);
        });

        stopBtn.addEventListener('click', () => {
            this.stopCombinedCamera(video, startBtn, captureBtn, stopBtn);
        });

        analyzeBtn.addEventListener('click', () => {
            this.analyzeCombined(textInput, resultsDiv);
        });
    }

    // Text Analysis
    async analyzeText(text, resultsDiv) {
        this.showLoading(resultsDiv, 'Analyzing text sentiment...');

        try {
            const response = await fetch('/analyze_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (data.success) {
                this.displayTextResults(data.sentiment, resultsDiv);
            } else {
                this.showError(resultsDiv, data.error || 'Analysis failed');
            }
        } catch (error) {
            this.showError(resultsDiv, `Error: ${error.message}`);
        }
    }

    // Camera Functions
    async startCamera(video, startBtn, captureBtn, stopBtn) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            video.srcObject = stream;
            this.currentStream = stream;

            startBtn.disabled = true;
            captureBtn.disabled = false;
            stopBtn.disabled = false;
        } catch (error) {
            alert('Error accessing camera: ' + error.message);
        }
    }

    stopCamera(video, startBtn, captureBtn, stopBtn) {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }
        
        video.srcObject = null;
        startBtn.disabled = false;
        captureBtn.disabled = true;
        stopBtn.disabled = true;
    }

    async captureAndAnalyze(video, canvas, resultsDiv) {
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        context.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        this.showLoading(resultsDiv, 'Analyzing facial emotions...');

        try {
            const response = await fetch('/analyze_facial', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (data.success) {
                this.displayFacialResults(data.emotions, resultsDiv);
            } else {
                this.showError(resultsDiv, data.error || 'Analysis failed');
            }
        } catch (error) {
            this.showError(resultsDiv, `Error: ${error.message}`);
        }
    }

    // Combined Analysis Functions
    async startCombinedCamera(video, startBtn, captureBtn, stopBtn) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            video.srcObject = stream;
            this.combinedStream = stream;

            startBtn.disabled = true;
            captureBtn.disabled = false;
            stopBtn.disabled = false;
        } catch (error) {
            alert('Error accessing camera: ' + error.message);
        }
    }

    stopCombinedCamera(video, startBtn, captureBtn, stopBtn) {
        if (this.combinedStream) {
            this.combinedStream.getTracks().forEach(track => track.stop());
            this.combinedStream = null;
        }
        
        video.srcObject = null;
        startBtn.disabled = false;
        captureBtn.disabled = true;
        stopBtn.disabled = true;
    }

    captureForCombined(video, canvas) {
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        context.drawImage(video, 0, 0);
        this.combinedCapturedImage = canvas.toDataURL('image/jpeg', 0.8);
        
        // Show confirmation
        const message = document.createElement('div');
        message.className = 'success';
        message.textContent = 'Image captured! Click "Analyze Combined Mood" to proceed.';
        
        const resultsDiv = document.getElementById('combinedResults');
        resultsDiv.innerHTML = '';
        resultsDiv.appendChild(message);
        resultsDiv.classList.remove('hidden');
    }

    async analyzeCombined(textInput, resultsDiv) {
        const text = textInput.value.trim();
        const image = this.combinedCapturedImage;

        if (!text && !image) {
            this.showError(resultsDiv, 'Please provide text input and/or capture an image.');
            return;
        }

        this.showLoading(resultsDiv, 'Analyzing combined mood...');

        try {
            const response = await fetch('/analyze_combined', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: text,
                    image: image
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayCombinedResults(data.results, resultsDiv);
            } else {
                this.showError(resultsDiv, data.error || 'Analysis failed');
            }
        } catch (error) {
            this.showError(resultsDiv, `Error: ${error.message}`);
        }
    }

    // Display Results Functions
    displayTextResults(sentiment, resultsDiv) {
        resultsDiv.innerHTML = `
            <h3><i class="fas fa-chart-bar"></i> Text Sentiment Analysis</h3>
            <div class="result-item">
                <div class="result-label">Sentiment</div>
                <div class="result-value">${sentiment.label.charAt(0).toUpperCase() + sentiment.label.slice(1)}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(sentiment.confidence * 100)}%"></div>
                </div>
                <small>Confidence: ${(sentiment.confidence * 100).toFixed(1)}%</small>
            </div>
            <div class="result-item">
                <div class="result-label">Score</div>
                <div class="result-value">${sentiment.score.toFixed(3)} (Range: -1 to 1)</div>
            </div>
            <div class="result-item">
                <div class="result-label">Magnitude</div>
                <div class="result-value">${sentiment.magnitude.toFixed(3)}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Analysis Source</div>
                <div class="result-value">${sentiment.source === 'google_cloud' ? 'Google Cloud Natural Language API' : 'Local Analysis'}</div>
            </div>
        `;
        resultsDiv.classList.remove('hidden');
    }

    displayFacialResults(emotions, resultsDiv) {
        if (!emotions.face_detected) {
            this.showError(resultsDiv, emotions.message || 'No face detected in image');
            return;
        }

        let emotionsList = '';
        if (emotions.emotions && Object.keys(emotions.emotions).length > 0) {
            for (const [emotion, confidence] of Object.entries(emotions.emotions)) {
                emotionsList += `
                    <div class="result-item">
                        <div class="result-label">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${(confidence * 100)}%"></div>
                        </div>
                        <small>${(confidence * 100).toFixed(1)}%</small>
                    </div>
                `;
            }
        }

        resultsDiv.innerHTML = `
            <h3><i class="fas fa-smile"></i> Facial Emotion Analysis</h3>
            <div class="result-item">
                <div class="result-label">Dominant Emotion</div>
                <div class="result-value">${emotions.dominant_emotion}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(emotions.confidence * 100)}%"></div>
                </div>
                <small>Overall Confidence: ${(emotions.confidence * 100).toFixed(1)}%</small>
            </div>
            ${emotionsList}
        `;
        resultsDiv.classList.remove('hidden');
    }

    displayCombinedResults(results, resultsDiv) {
        let html = '<h3><i class="fas fa-brain"></i> Combined Mood Analysis</h3>';

        if (results.text_sentiment) {
            html += `
                <div class="result-item">
                    <h4>Text Sentiment</h4>
                    <div class="result-label">Sentiment: ${results.text_sentiment.label}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${(results.text_sentiment.confidence * 100)}%"></div>
                    </div>
                    <small>Score: ${results.text_sentiment.score.toFixed(3)}</small>
                </div>
            `;
        }

        if (results.facial_emotions && results.facial_emotions.face_detected) {
            html += `
                <div class="result-item">
                    <h4>Facial Emotions</h4>
                    <div class="result-label">Dominant: ${results.facial_emotions.dominant_emotion}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${(results.facial_emotions.confidence * 100)}%"></div>
                    </div>
                    <small>Confidence: ${(results.facial_emotions.confidence * 100).toFixed(1)}%</small>
                </div>
            `;
        }

        if (results.overall_mood) {
            const mood = results.overall_mood;
            html += `
                <div class="result-item" style="border: 2px solid #667eea; background: #f0f4ff;">
                    <h4><i class="fas fa-chart-line"></i> Overall Mood Assessment</h4>
                    <div class="result-label">Mood Category</div>
                    <div class="result-value">${mood.mood_category.replace('_', ' ').toUpperCase()}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${(mood.confidence * 100)}%"></div>
                    </div>
                    <small>Confidence: ${(mood.confidence * 100).toFixed(1)}%</small>
                    <div class="result-label" style="margin-top: 10px;">Description</div>
                    <div class="result-value">${mood.description}</div>
                    <div class="result-label" style="margin-top: 10px;">Recommendations</div>
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        ${mood.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        resultsDiv.innerHTML = html;
        resultsDiv.classList.remove('hidden');
    }

    // Utility Functions
    showLoading(element, message) {
        element.innerHTML = `
            <div class="result-item">
                <div class="loading"></div>
                <span style="margin-left: 10px;">${message}</span>
            </div>
        `;
        element.classList.remove('hidden');
    }

    showError(element, message) {
        element.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i>
                ${message}
            </div>
        `;
        element.classList.remove('hidden');
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MoodDetectionApp();
});

// Handle page visibility change to stop cameras when tab is not visible
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Stop all camera streams when tab becomes hidden
        const streams = [
            document.getElementById('videoElement'),
            document.getElementById('combinedVideoElement')
        ];
        
        streams.forEach(video => {
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
        });
    }
});
