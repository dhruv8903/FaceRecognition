## Youth Mood Detection AI

An end-to-end Flask web app that analyzes mood from text and facial expressions using MediaPipe Face Mesh, TensorFlow-backed models, and a local keyword fallback for text sentiment. The UI lets you run Text, Facial, or Combined analysis in the browser.

### Features
- Text sentiment analysis via Google Cloud Natural Language API, with a privacy-first local fallback
- Facial emotion detection using MediaPipe Face Mesh and ML models (enhanced and original FER)
- Combined analysis aggregates text and facial signals
- Simple web UI with camera support

### Project Structure
```
app.py                         # Flask app entrypoint and API routes
text_sentiment.py              # Text sentiment analyzer (cloud + local fallback)
enhanced_facial_emotion_ml.py  # Enhanced facial emotion detector (MediaPipe + ML)
facial_emotion_ml.py           # Baseline facial emotion detector
mood_aggregator.py             # Aggregates text + facial into overall mood
templates/index.html           # Frontend page
static/css/style.css           # Styles
static/js/app.js               # Frontend logic (camera, requests)
fer_models/                    # Original FER models (pkl, scaler, metadata)
enhanced_fer_models/           # Enhanced models (pkl, scaler, metadata)
requirements.txt               # Python dependencies
```

### Prerequisites
- Python 3.12 (recommended)
- Windows PowerShell (commands below assume PowerShell)
- Optional: Google Cloud Natural Language credentials JSON and the env var `GOOGLE_APPLICATION_CREDENTIALS` pointing to it

### Quick Start
```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Configure Google Cloud for text sentiment
# Set to your JSON credentials path. If not set, local fallback is used.
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\\path\\to\\gcloud-creds.json"

# 4) Run the app (defaults to http://127.0.0.1:5000)
python app.py
```

Environment variables (optional):
- `FLASK_HOST` (default `127.0.0.1`)
- `FLASK_PORT` (default `5000`)
- `FLASK_DEBUG` (`true` to enable reload + debug)

Example:
```powershell
$env:FLASK_DEBUG = "true"
$env:FLASK_HOST = "0.0.0.0"
$env:FLASK_PORT = "8000"
python app.py
```

### API Endpoints
- `GET /` — Serves the web UI
- `POST /analyze_text` — Body: `{ "text": "..." }`
  - Response: `{ success, sentiment: { score, magnitude, label, confidence, source } }`
- `POST /analyze_facial` — Body: `{ "image": "data:image/jpeg;base64,..." }`
  - Response: `{ success, emotions: { angry, ... }, face_detected, confidence, dominant_emotion, model_type }`
- `POST /analyze_combined` — Body: `{ "text": "...", "image": "data:image/jpeg;base64,..." }` (either can be omitted)
  - Response: `{ success, results: { text_sentiment?, facial_emotions?, overall_mood? } }`

### Models
The app attempts to load enhanced models first from `enhanced_fer_models/` and falls back to original FER models from `fer_models/`.

If you see "No valid models found!", ensure the following files exist:
- Enhanced: `enhanced_fast_best_emotion_model.pkl`, `enhanced_fast_feature_scaler.pkl`, `enhanced_fast_model_metadata.json`
- Original: `best_emotion_model.pkl`, `feature_scaler.pkl`, `model_metadata.json`

Note: If you want to train or swap models, keep the expected filenames and update metadata as needed.

### Troubleshooting
- Slow first start: TensorFlow + MediaPipe initialization can take 10–30 seconds.
- No Google credentials: Text analysis will automatically use the local keyword fallback; this is expected.
- Missing scikit-learn: If model `.pkl` or scaler loading fails with `No module named 'sklearn'`, install:
  ```powershell
  pip install scikit-learn
  ```
- Camera issues in browser: Ensure you grant camera permission and use a modern browser (Chrome/Edge). If using HTTPS is required by the browser for camera, run locally via `http://localhost`.

### Development
Run tests (if present):
```powershell
pytest -q
```

Format/lint suggestions:
- Keep code readable and avoid large inline comments
- Prefer explicit names and early returns; match project style

### License
MIT License. See `LICENSE`.

### Repository
GitHub: `https://github.com/dhruv8903/FaceRecognition`


