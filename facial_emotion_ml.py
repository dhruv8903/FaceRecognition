import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import os
import pickle
import json
from typing import Dict

class FacialEmotionDetector:
    """
    Advanced ML-based facial emotion detector using MediaPipe Face Mesh 468 landmarks
    Trained on real FER dataset from Kaggle
    """
    
    def __init__(self, model_dir: str = 'fer_models'):
        """Initialize MediaPipe Face Mesh and load trained ML models."""
        
        # Initialize MediaPipe Face Mesh (using 468 landmarks)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key facial regions for emotion detection (from face-mesh project)
        self.facial_regions = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            'nose_bridge': [6, 168, 8, 9, 10, 151, 195, 3, 51, 48, 115, 131, 134, 102],
            'nose_tip': [1, 2, 5, 4, 19, 20, 94, 125],
            'mouth_outer': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 324],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427],
            'jaw_line': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
        
        # Initialize model components
        self.best_model = None
        self.scaler = None
        self.emotion_labels = {
            0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
            4: 'sad', 5: 'surprised', 6: 'neutral'
        }
        
        # Load trained models
        self._load_trained_models(model_dir)
        
        print("Facial Emotion Detector initialized with trained ML models (FER dataset + 468 landmarks)")

    def _load_trained_models(self, model_dir: str):
        """Load the trained ML models and components"""
        try:
            # Load the best model
            best_model_path = os.path.join(model_dir, 'best_emotion_model.pkl')
            if os.path.exists(best_model_path):
                with open(best_model_path, 'rb') as f:
                    self.best_model = pickle.load(f)
                print(f"Loaded best emotion model from {best_model_path}")
            else:
                print(f"Best model file not found at {best_model_path}")
                return False
                
            # Load the feature scaler
            scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Loaded feature scaler")
            else:
                print(f"Scaler file not found at {scaler_path}")
                return False
                
            # Load metadata if available
            metadata_path = os.path.join(model_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'emotion_labels' in metadata:
                        # Convert string keys back to integers
                        self.emotion_labels = {int(k): v for k, v in metadata['emotion_labels'].items()}
                    if 'facial_regions' in metadata:
                        self.facial_regions.update(metadata['facial_regions'])
                print("Loaded model metadata")
                
            return True
            
        except Exception as e:
            print(f"Error loading trained models: {e}")
            return False

    def detect_emotions(self, image):
        """
        Detect emotions from facial image using trained ML models
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            
        Returns:
            dict: Emotion detection results
        """
        try:
            # Convert PIL image to OpenCV format if needed
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image
            
            # Check if models are loaded
            if self.best_model is None or self.scaler is None:
                return {
                    'emotions': {},
                    'face_detected': False,
                    'error': 'ML models not loaded properly'
                }
            
            # Extract facial landmarks and calculate features
            features = self.extract_landmarks(cv_image)
            
            if features is None:
                return {
                    'emotions': {},
                    'face_detected': False,
                    'message': 'No face detected in the image'
                }
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction and probabilities
            prediction = self.best_model.predict(features_scaled)[0]
            probabilities = self.best_model.predict_proba(features_scaled)[0]
            
            # Create emotion probability dictionary
            emotions = {}
            for i, prob in enumerate(probabilities):
                if i in self.emotion_labels:
                    emotions[self.emotion_labels[i]] = float(prob)
            
            # Get dominant emotion and confidence
            dominant_emotion = self.emotion_labels[prediction]
            confidence = float(max(probabilities))
            
            return {
                'emotions': emotions,
                'face_detected': True,
                'confidence': confidence,
                'dominant_emotion': dominant_emotion,
                'model_type': 'ML_trained_on_FER_dataset'
            }
            
        except Exception as e:
            return {
                'emotions': {},
                'face_detected': False,
                'error': str(e)
            }

    def extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all 468 face mesh landmarks from an image and calculate features
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            numpy array of geometric features, or None if no face detected
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            # Extract landmarks
            landmarks = results.multi_face_landmarks[0]
            landmark_points = []
            
            # Extract all 468 landmarks (x, y, z coordinates)
            for landmark in landmarks.landmark:
                landmark_points.extend([landmark.x, landmark.y, landmark.z])
            
            # Calculate geometric features from landmarks
            features = self.calculate_geometric_features(np.array(landmark_points))
            
            return features
            
        except Exception as e:
            return None
    
    def calculate_geometric_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate advanced geometric features from 468 landmarks for emotion detection
        Same as in the training script to ensure consistency
        """
        try:
            # Reshape landmarks to (468, 3)
            points = landmarks.reshape(-1, 3)
            
            features = []
            
            # 1. Eye Aspect Ratios (EAR)
            left_eye_points = np.array([points[i] for i in self.facial_regions['left_eye'][:6] if i < len(points)])
            right_eye_points = np.array([points[i] for i in self.facial_regions['right_eye'][:6] if i < len(points)])
            
            if len(left_eye_points) >= 6 and len(right_eye_points) >= 6:
                left_ear = self._calculate_eye_aspect_ratio(left_eye_points)
                right_ear = self._calculate_eye_aspect_ratio(right_eye_points)
                features.extend([left_ear, right_ear, abs(left_ear - right_ear)])  # Eye asymmetry
            else:
                features.extend([0.3, 0.3, 0.0])
            
            # 2. Mouth Features
            mouth_outer = np.array([points[i] for i in self.facial_regions['mouth_outer'] if i < len(points)])
            
            if len(mouth_outer) >= 6:
                mar = self._calculate_mouth_aspect_ratio(mouth_outer[:6])
                mouth_width = np.linalg.norm(mouth_outer[0] - mouth_outer[6]) if len(mouth_outer) > 6 else 0
                mouth_curvature = self._calculate_mouth_curvature(mouth_outer)
                features.extend([mar, mouth_width, mouth_curvature])
            else:
                features.extend([0.1, 0.1, 0.0])
            
            # 3. Eyebrow Features
            left_brow = np.array([points[i] for i in self.facial_regions['left_eyebrow'] if i < len(points)])
            right_brow = np.array([points[i] for i in self.facial_regions['right_eyebrow'] if i < len(points)])
            
            if len(left_brow) >= 5 and len(right_brow) >= 5:
                left_brow_height = np.mean(left_brow[:, 1])
                right_brow_height = np.mean(right_brow[:, 1])
                brow_asymmetry = abs(left_brow_height - right_brow_height)
                brow_slope_left = self._calculate_slope(left_brow[:3])
                brow_slope_right = self._calculate_slope(right_brow[:3])
                features.extend([left_brow_height, right_brow_height, brow_asymmetry, brow_slope_left, brow_slope_right])
            else:
                features.extend([0.4, 0.4, 0.0, 0.0, 0.0])
            
            # 4. Nose Features
            nose_bridge = np.array([points[i] for i in self.facial_regions['nose_bridge'] if i < len(points)])
            nose_tip = np.array([points[i] for i in self.facial_regions['nose_tip'] if i < len(points)])
            
            if len(nose_bridge) >= 4 and len(nose_tip) >= 4:
                nose_width = np.linalg.norm(nose_tip[0] - nose_tip[2]) if len(nose_tip) > 2 else 0
                nose_height = np.linalg.norm(nose_bridge[0] - nose_tip[0]) if len(nose_bridge) > 0 and len(nose_tip) > 0 else 0
                features.extend([nose_width, nose_height])
            else:
                features.extend([0.05, 0.1])
            
            # 5. Cheek Features
            left_cheek = np.array([points[i] for i in self.facial_regions['left_cheek'] if i < len(points)])
            right_cheek = np.array([points[i] for i in self.facial_regions['right_cheek'] if i < len(points)])
            
            if len(left_cheek) >= 3 and len(right_cheek) >= 3:
                cheek_puffiness_left = np.std(left_cheek[:, 2])  # Z-coordinate variation
                cheek_puffiness_right = np.std(right_cheek[:, 2])
                features.extend([cheek_puffiness_left, cheek_puffiness_right])
            else:
                features.extend([0.0, 0.0])
            
            # 6. Jaw Features
            jaw_points = np.array([points[i] for i in self.facial_regions['jaw_line'] if i < len(points)])
            
            if len(jaw_points) >= 6:
                jaw_width = np.linalg.norm(jaw_points[0] - jaw_points[-1]) if len(jaw_points) > 1 else 0
                jaw_angle = self._calculate_jaw_angle(jaw_points)
                features.extend([jaw_width, jaw_angle])
            else:
                features.extend([0.2, 0.0])
            
            # 7. Overall Face Features
            if len(points) >= 400:
                # Face symmetry
                left_face = points[:len(points)//2]
                right_face = points[len(points)//2:]
                symmetry_score = np.mean(np.abs(left_face[:, 0] - (1 - right_face[:, 0])))
                
                # Face compactness
                face_center = np.mean(points, axis=0)
                distances = [np.linalg.norm(point - face_center) for point in points]
                face_compactness = np.std(distances)
                
                # Eye-mouth triangle area
                eye_center = np.mean([points[33], points[362]], axis=0) if len(points) > 362 else np.array([0.5, 0.4, 0])
                mouth_center = points[13] if len(points) > 13 else np.array([0.5, 0.7, 0])
                triangle_area = self._calculate_triangle_area(points[33], points[362], mouth_center) if len(points) > 362 else 0
                
                features.extend([symmetry_score, face_compactness, triangle_area])
            else:
                features.extend([0.02, 0.1, 0.05])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return np.zeros(20)  # Return default features

    # Helper methods (same as in training script)
    def _calculate_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        try:
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C) if C > 0 else 0.3
        except:
            return 0.3
    
    def _calculate_mouth_aspect_ratio(self, mouth_points: np.ndarray) -> float:
        try:
            A = np.linalg.norm(mouth_points[1] - mouth_points[5])
            B = np.linalg.norm(mouth_points[2] - mouth_points[4])
            C = np.linalg.norm(mouth_points[0] - mouth_points[3])
            return (A + B) / (2.0 * C) if C > 0 else 0.1
        except:
            return 0.1
    
    def _calculate_mouth_curvature(self, mouth_points: np.ndarray) -> float:
        try:
            if len(mouth_points) < 12:
                return 0.0
            left_corner = mouth_points[0]
            center = mouth_points[6]
            right_corner = mouth_points[6]
            left_curve = left_corner[1] - center[1]
            right_curve = right_corner[1] - center[1]
            return (left_curve + right_curve) / 2
        except:
            return 0.0
    
    def _calculate_slope(self, points: np.ndarray) -> float:
        try:
            if len(points) < 2:
                return 0.0
            x1, y1 = points[0][:2]
            x2, y2 = points[-1][:2]
            return (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
        except:
            return 0.0
    
    def _calculate_jaw_angle(self, jaw_points: np.ndarray) -> float:
        try:
            if len(jaw_points) < 3:
                return 0.0
            p1 = jaw_points[0]
            p2 = jaw_points[len(jaw_points)//2]
            p3 = jaw_points[-1]
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            return angle
        except:
            return 0.0
    
    def _calculate_triangle_area(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        try:
            return 0.5 * abs((p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])))
        except:
            return 0.0