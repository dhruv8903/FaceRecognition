"""
FaceSDK-Enhanced Emotion Detection Module
Provides superior performance compared to MediaPipe-based detection
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
import requests
import base64
from PIL import Image
import io
import logging

class FaceSDKEmotionDetector:
    """
    Enhanced facial emotion detector using FaceSDK for superior accuracy and performance.
    Falls back to MediaPipe if FaceSDK is not available.
    """
    
    def __init__(self, api_endpoint: str = None, api_key: str = None):
        """
        Initialize the FaceSDK emotion detector.
        
        Args:
            api_endpoint: FaceSDK API endpoint URL
            api_key: API key for FaceSDK service
        """
        self.api_endpoint = api_endpoint or "https://facerecognition.rapidapi.com"
        self.api_key = api_key
        self.use_facesdk = self._check_facesdk_availability()
        
        # Fallback to MediaPipe if FaceSDK not available
        if not self.use_facesdk:
            self._init_mediapipe_fallback()
            
        self.emotion_mapping = {
            'angry': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful', 
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprised',
            'neutral': 'neutral'
        }
        
        logging.info(f"✓ FaceSDK Emotion Detector initialized (Using {'FaceSDK' if self.use_facesdk else 'MediaPipe fallback'})")

    def _check_facesdk_availability(self) -> bool:
        """Check if FaceSDK API is available and accessible."""
        if not self.api_key:
            return False
            
        try:
            # Test API connectivity
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "facerecognition.rapidapi.com"
            }
            
            # Make a simple test request
            response = requests.get(f"{self.api_endpoint}/status", headers=headers, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            logging.warning(f"FaceSDK not available: {e}")
            return False

    def _init_mediapipe_fallback(self):
        """Initialize MediaPipe as fallback when FaceSDK is not available."""
        try:
            import mediapipe as mp
            
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.7
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7
            )
            
            # Load enhanced models if available
            try:
                import pickle
                import os
                
                model_path = "enhanced_fer_models/enhanced_fast_best_emotion_model.pkl"
                scaler_path = "enhanced_fer_models/enhanced_fast_feature_scaler.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.emotion_model = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.feature_scaler = pickle.load(f)
                    self.has_ml_model = True
                    logging.info("✓ Loaded enhanced ML models for fallback")
                else:
                    self.has_ml_model = False
                    logging.warning("No enhanced ML models found, using basic detection")
                    
            except Exception as e:
                self.has_ml_model = False
                logging.warning(f"Could not load ML models: {e}")
                
        except ImportError:
            logging.error("MediaPipe not available for fallback")
            self.face_detection = None
            self.face_mesh = None
            self.has_ml_model = False

    def detect_emotions_facesdk(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect emotions using FaceSDK API for superior accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing emotion predictions and metadata
        """
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare API request
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "facerecognition.rapidapi.com",
                "Content-Type": "application/json"
            }
            
            payload = {
                "image": f"data:image/jpeg;base64,{img_base64}",
                "detect_emotions": True,
                "detect_age": True,
                "detect_gender": True
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_endpoint}/detect",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._process_facesdk_result(result)
            else:
                logging.warning(f"FaceSDK API error: {response.status_code}")
                return self._fallback_detection(image)
                
        except Exception as e:
            logging.error(f"FaceSDK detection failed: {e}")
            return self._fallback_detection(image)

    def _process_facesdk_result(self, result: Dict) -> Dict[str, Any]:
        """Process FaceSDK API response into standardized format."""
        
        if not result.get('faces') or len(result['faces']) == 0:
            return {
                'emotions': {'neutral': 1.0},
                'confidence': 0.0,
                'face_detected': False,
                'model_type': 'FaceSDK',
                'processing_time': 0,
                'additional_info': {'message': 'No face detected'}
            }
            
        face_data = result['faces'][0]  # Take the first face
        emotions = face_data.get('emotions', {})
        
        # Normalize emotion names and values
        normalized_emotions = {}
        total_confidence = 0
        
        for emotion, confidence in emotions.items():
            emotion_key = self.emotion_mapping.get(emotion.lower(), emotion.lower())
            normalized_emotions[emotion_key] = float(confidence)
            total_confidence += float(confidence)
            
        # Ensure probabilities sum to 1
        if total_confidence > 0:
            for emotion in normalized_emotions:
                normalized_emotions[emotion] /= total_confidence
        
        # Get dominant emotion
        dominant_emotion = max(normalized_emotions, key=normalized_emotions.get)
        max_confidence = normalized_emotions[dominant_emotion]
        
        return {
            'emotions': normalized_emotions,
            'dominant_emotion': dominant_emotion,
            'confidence': max_confidence,
            'face_detected': True,
            'model_type': 'FaceSDK',
            'processing_time': result.get('processing_time', 0),
            'additional_info': {
                'age': face_data.get('age'),
                'gender': face_data.get('gender'),
                'face_quality': face_data.get('quality', 'unknown')
            }
        }

    def _fallback_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback to MediaPipe-based detection."""
        if not self.face_detection or not self.face_mesh:
            return self._create_error_result("No detection method available")
            
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_results = self.face_detection.process(rgb_image)
            
            if not face_results.detections:
                return self._create_no_face_result()
                
            # Extract features using face mesh
            mesh_results = self.face_mesh.process(rgb_image)
            
            if not mesh_results.multi_face_landmarks:
                return self._create_no_face_result()
                
            # Extract enhanced features
            features = self._extract_enhanced_features(mesh_results.multi_face_landmarks[0], rgb_image.shape)
            
            # Predict emotions using ML model
            if self.has_ml_model and features is not None:
                emotions = self._predict_emotions_ml(features)
            else:
                emotions = self._basic_emotion_estimation(features)
                
            dominant_emotion = max(emotions, key=emotions.get)
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': emotions[dominant_emotion],
                'face_detected': True,
                'model_type': 'MediaPipe_Enhanced_Fallback',
                'processing_time': 0,
                'additional_info': {'fallback_reason': 'FaceSDK unavailable'}
            }
            
        except Exception as e:
            logging.error(f"Fallback detection failed: {e}")
            return self._create_error_result(str(e))

    def _extract_enhanced_features(self, landmarks, image_shape) -> Optional[List[float]]:
        """Extract enhanced 31-feature set from MediaPipe landmarks."""
        try:
            height, width = image_shape[:2]
            
            # Convert landmarks to pixel coordinates
            points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append([x, y])
            
            points = np.array(points)
            
            # Extract 31 enhanced features (same as training)
            features = []
            
            # Eye features (6 features)
            left_eye = points[[33, 7, 163, 144, 145, 153]]
            right_eye = points[[362, 382, 381, 380, 374, 373]]
            
            features.extend(self._calculate_eye_features(left_eye, right_eye))
            
            # Mouth features (6 features) 
            mouth_points = points[[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]]
            features.extend(self._calculate_mouth_features(mouth_points))
            
            # Eyebrow features (7 features)
            left_brow = points[[70, 63, 105, 66, 107, 55, 65]]
            right_brow = points[[296, 334, 293, 300, 276, 283, 295]]
            features.extend(self._calculate_eyebrow_features(left_brow, right_brow, left_eye, right_eye))
            
            # Nose features (3 features)
            nose_points = points[[1, 2, 5, 4, 6, 19, 20, 125, 142, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 307, 375, 321, 308, 324, 318]]
            features.extend(self._calculate_nose_features(nose_points))
            
            # Cheek/Jaw features (6 features)
            face_oval = points[[10, 151, 9, 8, 168, 6, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]]
            features.extend(self._calculate_cheek_jaw_features(face_oval, points))
            
            # Facial geometry (3 features)
            features.extend(self._calculate_facial_geometry(points))
            
            return features[:31]  # Ensure exactly 31 features
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return None

    def _calculate_eye_features(self, left_eye, right_eye) -> List[float]:
        """Calculate eye-related features."""
        features = []
        
        try:
            # Left eye aspect ratio
            left_height = np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])
            left_width = np.linalg.norm(left_eye[0] - left_eye[3])
            left_ear = left_height / (2.0 * left_width) if left_width > 0 else 0
            features.append(left_ear)
            
            # Right eye aspect ratio
            right_height = np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])
            right_width = np.linalg.norm(right_eye[0] - right_eye[3])
            right_ear = right_height / (2.0 * right_width) if right_width > 0 else 0
            features.append(right_ear)
            
            # Eye width difference
            features.append(abs(left_width - right_width))
            
            # Eye height difference  
            features.append(abs(left_height - right_height))
            
            # Eye center distance
            left_center = np.mean(left_eye, axis=0)
            right_center = np.mean(right_eye, axis=0)
            eye_distance = np.linalg.norm(left_center - right_center)
            features.append(eye_distance)
            
            # Eye asymmetry
            features.append(abs(left_ear - right_ear))
            
        except Exception:
            features.extend([0.0] * 6)
            
        return features[:6]

    def _calculate_mouth_features(self, mouth_points) -> List[float]:
        """Calculate mouth-related features."""
        features = []
        
        try:
            # Mouth aspect ratio
            mouth_height = np.linalg.norm(mouth_points[3] - mouth_points[9])
            mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[6])
            mouth_ar = mouth_height / mouth_width if mouth_width > 0 else 0
            features.append(mouth_ar)
            
            # Mouth width
            features.append(mouth_width)
            
            # Mouth curvature (smile detection)
            left_corner = mouth_points[0]
            right_corner = mouth_points[6] 
            mouth_center = mouth_points[3]
            
            left_curve = left_corner[1] - mouth_center[1]  # Y difference
            right_curve = right_corner[1] - mouth_center[1]
            avg_curve = (left_curve + right_curve) / 2.0
            features.append(avg_curve)
            
            # Mouth corner asymmetry
            features.append(abs(left_curve - right_curve))
            
            # Upper lip curvature
            upper_lip_curve = mouth_points[1][1] - mouth_points[2][1]
            features.append(upper_lip_curve)
            
            # Lower lip curvature  
            lower_lip_curve = mouth_points[7][1] - mouth_points[8][1]
            features.append(lower_lip_curve)
            
        except Exception:
            features.extend([0.0] * 6)
            
        return features[:6]

    def _calculate_eyebrow_features(self, left_brow, right_brow, left_eye, right_eye) -> List[float]:
        """Calculate eyebrow-related features."""
        features = []
        
        try:
            # Eyebrow heights
            left_brow_height = np.mean([p[1] for p in left_brow])
            right_brow_height = np.mean([p[1] for p in right_brow])
            features.append(left_brow_height)
            features.append(right_brow_height)
            
            # Eyebrow slopes
            left_slope = (left_brow[-1][1] - left_brow[0][1]) / (left_brow[-1][0] - left_brow[0][0]) if left_brow[-1][0] != left_brow[0][0] else 0
            right_slope = (right_brow[-1][1] - right_brow[0][1]) / (right_brow[-1][0] - right_brow[0][0]) if right_brow[-1][0] != right_brow[0][0] else 0
            features.append(left_slope)
            features.append(right_slope)
            
            # Eyebrow-eye distances
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            left_brow_center = np.mean(left_brow, axis=0)
            right_brow_center = np.mean(right_brow, axis=0)
            
            left_brow_eye_dist = np.linalg.norm(left_brow_center - left_eye_center)
            right_brow_eye_dist = np.linalg.norm(right_brow_center - right_eye_center)
            features.append(left_brow_eye_dist)
            features.append(right_brow_eye_dist)
            
            # Eyebrow asymmetry
            features.append(abs(left_brow_height - right_brow_height))
            
        except Exception:
            features.extend([0.0] * 7)
            
        return features[:7]

    def _calculate_nose_features(self, nose_points) -> List[float]:
        """Calculate nose-related features."""
        features = []
        
        try:
            # Nose width (nostril distance)
            if len(nose_points) >= 20:
                nose_width = np.linalg.norm(nose_points[15] - nose_points[19])
                features.append(nose_width)
                
                # Nose height
                nose_height = np.linalg.norm(nose_points[0] - nose_points[3])
                features.append(nose_height)
                
                # Nose bridge curvature
                bridge_curve = abs(nose_points[1][0] - nose_points[2][0])
                features.append(bridge_curve)
            else:
                features.extend([0.0] * 3)
                
        except Exception:
            features.extend([0.0] * 3)
            
        return features[:3]

    def _calculate_cheek_jaw_features(self, face_oval, all_points) -> List[float]:
        """Calculate cheek and jaw features."""
        features = []
        
        try:
            # Face width at cheekbones
            if len(face_oval) >= 20:
                cheek_width = np.linalg.norm(face_oval[8] - face_oval[16])
                features.append(cheek_width)
                
                # Jaw width
                jaw_width = np.linalg.norm(face_oval[3] - face_oval[20])
                features.append(jaw_width)
                
                # Jaw angle (approximate)
                jaw_angle = abs(face_oval[3][1] - face_oval[1][1])
                features.append(jaw_angle)
                
                # Cheek fullness (distance from face center)
                face_center = np.mean(face_oval, axis=0)
                left_cheek_dist = np.linalg.norm(face_oval[8] - face_center)
                right_cheek_dist = np.linalg.norm(face_oval[16] - face_center)
                features.append(left_cheek_dist)
                features.append(right_cheek_dist)
                
                # Cheek asymmetry
                features.append(abs(left_cheek_dist - right_cheek_dist))
            else:
                features.extend([0.0] * 6)
                
        except Exception:
            features.extend([0.0] * 6)
            
        return features[:6]

    def _calculate_facial_geometry(self, points) -> List[float]:
        """Calculate overall facial geometry features."""
        features = []
        
        try:
            # Face symmetry (left vs right side comparison)
            face_center_x = np.mean([p[0] for p in points])
            left_points = [p for p in points if p[0] < face_center_x]
            right_points = [p for p in points if p[0] > face_center_x]
            
            if left_points and right_points:
                left_centroid = np.mean(left_points, axis=0)
                right_centroid = np.mean(right_points, axis=0)
                symmetry = np.linalg.norm(left_centroid - right_centroid)
                features.append(symmetry)
            else:
                features.append(0.0)
                
            # Face proportions (height/width ratio)
            face_height = max([p[1] for p in points]) - min([p[1] for p in points])
            face_width = max([p[0] for p in points]) - min([p[0] for p in points])
            face_ratio = face_height / face_width if face_width > 0 else 0
            features.append(face_ratio)
            
            # Triangle area (eyes and mouth)
            if len(points) >= 300:
                eye_left = points[33]
                eye_right = points[362] 
                mouth_center = points[13]
                
                # Calculate triangle area
                area = abs((eye_left[0] * (eye_right[1] - mouth_center[1]) + 
                           eye_right[0] * (mouth_center[1] - eye_left[1]) + 
                           mouth_center[0] * (eye_left[1] - eye_right[1])) / 2.0)
                features.append(area)
            else:
                features.append(0.0)
                
        except Exception:
            features.extend([0.0] * 3)
            
        return features[:3]

    def _predict_emotions_ml(self, features: List[float]) -> Dict[str, float]:
        """Predict emotions using trained ML model."""
        try:
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features_array)
            
            # Predict probabilities
            probabilities = self.emotion_model.predict_proba(features_scaled)[0]
            
            # Map to emotion names (assuming standard order)
            emotion_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
            
            emotions = {}
            for i, emotion in enumerate(emotion_names):
                if i < len(probabilities):
                    emotions[emotion] = float(probabilities[i])
                else:
                    emotions[emotion] = 0.0
                    
            return emotions
            
        except Exception as e:
            logging.error(f"ML prediction failed: {e}")
            return self._basic_emotion_estimation(features)

    def _basic_emotion_estimation(self, features: Optional[List[float]]) -> Dict[str, float]:
        """Basic emotion estimation based on facial features when ML model not available."""
        
        if not features or len(features) < 6:
            return {
                'neutral': 0.7,
                'happy': 0.1,
                'sad': 0.1,
                'surprised': 0.05,
                'angry': 0.03,
                'fearful': 0.01,
                'disgusted': 0.01
            }
            
        try:
            emotions = {'neutral': 0.4}  # Base neutral
            
            # Eye features (first 6)
            eye_openness = (features[0] + features[1]) / 2.0  # Average eye aspect ratio
            
            # Mouth features (next 6) 
            mouth_ar = features[6] if len(features) > 6 else 0
            mouth_curve = features[8] if len(features) > 8 else 0
            
            # Basic rules for emotion estimation
            if mouth_curve > 5:  # Upward curve suggests smile
                emotions['happy'] = 0.6
                emotions['neutral'] = 0.2
            elif mouth_curve < -5:  # Downward curve suggests frown
                emotions['sad'] = 0.5
                emotions['neutral'] = 0.25
                
            if mouth_ar > 0.05:  # Open mouth
                emotions['surprised'] = emotions.get('surprised', 0) + 0.3
                
            if eye_openness < 0.2:  # Squinting
                emotions['angry'] = emotions.get('angry', 0) + 0.3
                
            # Normalize probabilities
            total = sum(emotions.values())
            if total > 0:
                for emotion in emotions:
                    emotions[emotion] /= total
                    
            # Fill missing emotions
            all_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
            for emotion in all_emotions:
                if emotion not in emotions:
                    emotions[emotion] = 0.01
                    
            return emotions
            
        except Exception:
            # Return default neutral emotion
            return {
                'neutral': 0.7,
                'happy': 0.1,
                'sad': 0.1,
                'surprised': 0.05,
                'angry': 0.03,
                'fearful': 0.01,
                'disgusted': 0.01
            }

    def _create_no_face_result(self) -> Dict[str, Any]:
        """Create result for when no face is detected."""
        return {
            'emotions': {'neutral': 1.0},
            'confidence': 0.0,
            'face_detected': False,
            'model_type': 'MediaPipe_Fallback',
            'processing_time': 0,
            'additional_info': {'message': 'No face detected'}
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create result for when an error occurs."""
        return {
            'emotions': {'neutral': 1.0},
            'confidence': 0.0,
            'face_detected': False,
            'model_type': 'Error',
            'processing_time': 0,
            'additional_info': {'error': error_message}
        }

    def detect_emotions(self, image: Image.Image) -> Dict[str, Any]:
        """
        Main method to detect emotions from an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing emotion predictions and metadata
        """
        if self.use_facesdk:
            return self.detect_emotions_facesdk(image)
        else:
            return self._fallback_detection(image)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model being used."""
        return {
            'using_facesdk': self.use_facesdk,
            'api_endpoint': self.api_endpoint if self.use_facesdk else None,
            'has_api_key': bool(self.api_key),
            'fallback_model': 'MediaPipe + Enhanced ML' if not self.use_facesdk else None,
            'has_enhanced_ml_model': getattr(self, 'has_ml_model', False)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector (will use fallback if no API key provided)
    detector = FaceSDKEmotionDetector()
    
    # Test with a sample image
    try:
        # You would load your test image here
        # image = Image.open("test_image.jpg")
        # result = detector.detect_emotions(image)
        # print(json.dumps(result, indent=2))
        
        model_info = detector.get_model_info()
        print("Model Information:")
        print(json.dumps(model_info, indent=2))
        
    except Exception as e:
        print(f"Test failed: {e}")