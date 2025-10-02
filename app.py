"""
Enhanced Flask Application with FaceSDK Integration
Provides superior emotion detection performance using FaceSDK with intelligent fallbacks
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from text_sentiment import TextSentimentAnalyzer
import base64
import io
from PIL import Image
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import emotion detection systems with fallback hierarchy
try:
    from facesdk_emotion_detector import FaceSDKEmotionDetector
    USE_FACESDK = True
    logger.info("✓ FaceSDK detector available")
except ImportError as e:
    USE_FACESDK = False
    logger.warning(f"✗ FaceSDK detector not available: {e}")

try:
    from enhanced_facial_emotion_ml import EnhancedFacialEmotionDetector
    USE_ENHANCED = True
    logger.info("✓ Enhanced detector available")
except ImportError as e:
    USE_ENHANCED = False
    logger.warning(f"✗ Enhanced detector not available: {e}")

try:
    from facial_emotion_ml import FacialEmotionDetector
    USE_STANDARD = True
    logger.info("✓ Standard detector available")
except ImportError as e:
    USE_STANDARD = False
    logger.warning(f"✗ Standard detector not available: {e}")

from mood_aggregator import MoodAggregator

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class EmotionDetectionService:
    """
    Intelligent emotion detection service with multiple detector support and fallback logic.
    """
    
    def __init__(self):
        """Initialize the emotion detection service with available detectors."""
        self.detectors = {}
        self.primary_detector = None
        self.detection_stats = {
            'total_requests': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'detector_usage': {}
        }
        
        # Initialize detectors in order of preference
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize available detectors in order of preference."""
        
        # 1. Try to initialize FaceSDK (highest priority)
        if USE_FACESDK:
            try:
                # Check for API key in environment
                facesdk_api_key = os.getenv('FACESDK_API_KEY', os.getenv('RAPIDAPI_KEY'))
                
                if facesdk_api_key:
                    self.detectors['FaceSDK_API'] = FaceSDKEmotionDetector(
                        api_key=facesdk_api_key
                    )
                    logger.info("✓ FaceSDK API detector initialized")
                    
                # Also initialize fallback version
                self.detectors['FaceSDK_Fallback'] = FaceSDKEmotionDetector()
                logger.info("✓ FaceSDK fallback detector initialized")
                
                # Set primary detector
                if not self.primary_detector:
                    self.primary_detector = self.detectors.get('FaceSDK_API') or self.detectors['FaceSDK_Fallback']
                    
            except Exception as e:
                logger.error(f"Failed to initialize FaceSDK: {e}")
        
        # 2. Enhanced detector (medium priority)
        if USE_ENHANCED and not self.primary_detector:
            try:
                self.detectors['Enhanced'] = EnhancedFacialEmotionDetector()
                self.primary_detector = self.detectors['Enhanced']
                logger.info("✓ Enhanced detector initialized as primary")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced detector: {e}")
        
        # 3. Standard detector (lowest priority, but essential fallback)
        if USE_STANDARD:
            try:
                self.detectors['Standard'] = FacialEmotionDetector()
                if not self.primary_detector:
                    self.primary_detector = self.detectors['Standard']
                    logger.info("✓ Standard detector initialized as primary")
                else:
                    logger.info("✓ Standard detector initialized as fallback")
            except Exception as e:
                logger.error(f"Failed to initialize Standard detector: {e}")
        
        # Summary
        if self.primary_detector:
            primary_name = self._get_detector_name(self.primary_detector)
            logger.info(f"🎯 Primary detector: {primary_name}")
            logger.info(f"📊 Total detectors available: {len(self.detectors)}")
        else:
            logger.error("❌ No emotion detectors available!")
            
    def _get_detector_name(self, detector) -> str:
        """Get the name of a detector instance."""
        for name, det in self.detectors.items():
            if det is detector:
                return name
        return "Unknown"
    
    def detect_emotions(self, image: Image.Image, prefer_speed: bool = False) -> dict:
        """
        Detect emotions from an image using the best available detector.
        
        Args:
            image: PIL Image object
            prefer_speed: If True, prioritize speed over accuracy
            
        Returns:
            Dict containing emotion predictions and metadata
        """
        start_time = time.time()
        self.detection_stats['total_requests'] += 1
        
        # Determine which detector to use
        detector_to_use = self._select_detector(prefer_speed)
        detector_name = self._get_detector_name(detector_to_use)
        
        # Track detector usage
        self.detection_stats['detector_usage'][detector_name] = \
            self.detection_stats['detector_usage'].get(detector_name, 0) + 1
        try:
            # Run emotion detection
            result = detector_to_use.detect_emotions(image)
            
            # Add metadata
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['detector_used'] = detector_name
            result['service_stats'] = self.get_stats()
            
            # Update stats
            if result.get('face_detected', False):
                self.detection_stats['successful_detections'] += 1
            else:
                self.detection_stats['failed_detections'] += 1
                # Try fallback if primary detector failed and fallbacks available
                if len(self.detectors) > 1 and detector_to_use == self.primary_detector:
                    fallback_result = self._try_fallback_detection(image, exclude=detector_name)
                    if fallback_result and fallback_result.get('face_detected', False):
                        fallback_result['processing_time'] = time.time() - start_time
                        fallback_result['service_stats'] = self.get_stats()
                        return fallback_result
            
            return result
            
        except Exception as e:
            logger.error(f"Emotion detection failed with {detector_name}: {e}")
            self.detection_stats['failed_detections'] += 1
            
            # Try fallback
            if len(self.detectors) > 1:
                fallback_result = self._try_fallback_detection(image, exclude=detector_name)
                if fallback_result:
                    fallback_result['processing_time'] = time.time() - start_time
                    fallback_result['service_stats'] = self.get_stats()
                    return fallback_result
            
            # Return error result
            return {
                'emotions': {'neutral': 1.0},
                'confidence': 0.0,
                'face_detected': False,
                'model_type': 'Error',
                'processing_time': time.time() - start_time,
                'detector_used': detector_name,
                'additional_info': {'error': str(e)},
                'service_stats': self.get_stats()
            }
    
    def _select_detector(self, prefer_speed: bool = False):
        """Select the best detector based on preferences and availability."""
        
        if prefer_speed:
            # For speed, prefer detectors in this order: Standard -> Enhanced -> FaceSDK
            for detector_name in ['Standard', 'Enhanced', 'FaceSDK_Fallback', 'FaceSDK_API']:
                if detector_name in self.detectors:
                    return self.detectors[detector_name]
        else:
            # For accuracy, prefer detectors in this order: FaceSDK_API -> FaceSDK_Fallback -> Enhanced -> Standard
            for detector_name in ['FaceSDK_API', 'FaceSDK_Fallback', 'Enhanced', 'Standard']:
                if detector_name in self.detectors:
                    return self.detectors[detector_name]
        
        # Fallback to primary detector
        return self.primary_detector
    
    def _try_fallback_detection(self, image: Image.Image, exclude: str = None) -> dict:
        """Try fallback detectors when primary detection fails."""
        
        for detector_name, detector in self.detectors.items():
            if detector_name == exclude:
                continue
                
            try:
                logger.info(f"Trying fallback detector: {detector_name}")
                result = detector.detect_emotions(image)
                
                if result.get('face_detected', False):
                    result['detector_used'] = f"{detector_name} (fallback)"
                    self.detection_stats['successful_detections'] += 1
                    logger.info(f"✓ Fallback detection successful with {detector_name}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Fallback detector {detector_name} also failed: {e}")
                continue
        
        return None
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        total = self.detection_stats['total_requests']
        return {
            'total_requests': total,
            'success_rate': self.detection_stats['successful_detections'] / total if total > 0 else 0,
            'detectors_available': list(self.detectors.keys()),
            'primary_detector': self._get_detector_name(self.primary_detector) if self.primary_detector else None,
            'detector_usage': self.detection_stats['detector_usage']
        }
    
    def get_model_info(self) -> dict:
        """Get information about all available models."""
        info = {
            'available_detectors': {},
            'primary_detector': self._get_detector_name(self.primary_detector) if self.primary_detector else None,
            'total_detectors': len(self.detectors)
        }
        
        for detector_name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'get_model_info'):
                    info['available_detectors'][detector_name] = detector.get_model_info()
                else:
                    info['available_detectors'][detector_name] = {'type': type(detector).__name__}
            except Exception as e:
                info['available_detectors'][detector_name] = {'error': str(e)}
        
        return info

# Initialize services
logger.info("🚀 Initializing Emotion Detection Service...")
emotion_service = EmotionDetectionService()
text_analyzer = TextSentimentAnalyzer()
mood_aggregator = MoodAggregator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get API and service status."""
    return jsonify({
        'status': 'active',
        'service_stats': emotion_service.get_stats(),
        'model_info': emotion_service.get_model_info(),
        'timestamp': time.time()
    })

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze text sentiment
        sentiment_result = text_analyzer.analyze(text)
        
        return jsonify({
            'success': True,
            'sentiment': sentiment_result,
            'processing_time': 0,  # Text analysis is very fast
            'service_stats': emotion_service.get_stats()
        })
    
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_facial', methods=['POST'])
def analyze_facial():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        prefer_speed = data.get('prefer_speed', False)
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        try:
            image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {e}'}), 400
        
        # Analyze facial emotions using the service
        emotion_result = emotion_service.detect_emotions(image, prefer_speed=prefer_speed)
        
        return jsonify({
            'success': True,
            'emotions': emotion_result
        })
    
    except Exception as e:
        logger.error(f"Facial analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_combined', methods=['POST'])
def analyze_combined():
    try:
        data = request.get_json()
        text = data.get('text', '')
        image_data = data.get('image', '')
        prefer_speed = data.get('prefer_speed', False)
        
        results = {}
        
        # Analyze text if provided
        if text:
            text_sentiment = text_analyzer.analyze(text)
            results['text_sentiment'] = text_sentiment
        
        # Analyze facial emotions if image provided
        if image_data:
            try:
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                facial_emotions = emotion_service.detect_emotions(image, prefer_speed=prefer_speed)
                results['facial_emotions'] = facial_emotions
            except Exception as e:
                results['facial_emotions'] = {
                    'error': f'Image processing failed: {e}',
                    'face_detected': False
                }
        
        # Aggregate mood if both are available
        if text and image_data and results.get('facial_emotions', {}).get('face_detected', False):
            try:
                overall_mood = mood_aggregator.aggregate_mood(
                    results['text_sentiment'], 
                    results['facial_emotions']
                )
                results['overall_mood'] = overall_mood
            except Exception as e:
                results['overall_mood'] = {'error': f'Mood aggregation failed: {e}'}
        
        # Add service information
        results['service_info'] = {
            'stats': emotion_service.get_stats(),
            'timestamp': time.time()
        }
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Combined analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Benchmark endpoint to test detector performance."""
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        iterations = min(data.get('iterations', 3), 10)  # Limit to max 10 iterations
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
            
        # Decode image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run benchmark
        results = {}
        
        for detector_name, detector in emotion_service.detectors.items():
            detector_results = {
                'name': detector_name,
                'times': [],
                'successful_detections': 0,
                'failed_detections': 0
            }
            
            for i in range(iterations):
                start_time = time.time()
                try:
                    result = detector.detect_emotions(image)
                    end_time = time.time()
                    
                    detector_results['times'].append(end_time - start_time)
                    
                    if result.get('face_detected', False):
                        detector_results['successful_detections'] += 1
                    else:
                        detector_results['failed_detections'] += 1
                        
                except Exception as e:
                    detector_results['failed_detections'] += 1
                    detector_results['times'].append(time.time() - start_time)
            
            # Calculate statistics
            if detector_results['times']:
                detector_results['avg_time'] = sum(detector_results['times']) / len(detector_results['times'])
                detector_results['min_time'] = min(detector_results['times'])
                detector_results['max_time'] = max(detector_results['times'])
            
            results[detector_name] = detector_results
        
        return jsonify({
            'success': True,
            'benchmark_results': results,
            'iterations': iterations,
            'service_stats': emotion_service.get_stats()
        })
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configuration from environment variables
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5000'))  # Default port
    
    logger.info(f"🌟 Starting Enhanced Emotion Detection Server...")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"📊 API Status: http://{host}:{port}/api/status")
    
    app.run(debug=debug_mode, host=host, port=port)