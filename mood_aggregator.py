import numpy as np
from typing import Dict, Any, Optional

class MoodAggregator:
    def __init__(self):
        """Initialize the mood aggregation system."""
        # Define mood categories and their characteristics
        self.mood_categories = {
            'very_positive': {
                'sentiment_range': (0.5, 1.0),
                'emotions': ['happy', 'excited', 'joyful'],
                'description': 'Very positive and energetic mood'
            },
            'positive': {
                'sentiment_range': (0.1, 0.5),
                'emotions': ['happy', 'content', 'pleased'],
                'description': 'Generally positive mood'
            },
            'neutral': {
                'sentiment_range': (-0.1, 0.1),
                'emotions': ['neutral', 'calm'],
                'description': 'Neutral or balanced mood'
            },
            'negative': {
                'sentiment_range': (-0.5, -0.1),
                'emotions': ['sad', 'disappointed', 'frustrated'],
                'description': 'Generally negative mood'
            },
            'very_negative': {
                'sentiment_range': (-1.0, -0.5),
                'emotions': ['angry', 'depressed', 'anxious'],
                'description': 'Very negative or distressed mood'
            }
        }
        
        # Weights for different factors
        self.weights = {
            'text_sentiment': 0.4,
            'facial_emotions': 0.6
        }
    
    def aggregate_mood(self, text_sentiment: Dict[str, Any], facial_emotions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate text sentiment and facial emotions to determine overall mood.
        
        Args:
            text_sentiment: Results from text sentiment analysis
            facial_emotions: Results from facial emotion detection
            
        Returns:
            dict: Aggregated mood analysis
        """
        try:
            # Extract sentiment score
            text_score = text_sentiment.get('score', 0.0)
            text_confidence = text_sentiment.get('confidence', 0.0)
            
            # Extract facial emotion scores
            facial_scores = self._extract_facial_scores(facial_emotions)
            facial_confidence = facial_emotions.get('confidence', 0.0)
            
            # Calculate weighted mood score
            mood_score = self._calculate_mood_score(
                text_score, text_confidence,
                facial_scores, facial_confidence
            )
            
            # Determine mood category
            mood_category = self._categorize_mood(mood_score, facial_scores)
            
            # Calculate confidence in the assessment
            overall_confidence = self._calculate_overall_confidence(
                text_confidence, facial_confidence
            )
            
            # Generate recommendations based on mood
            recommendations = self._generate_recommendations(mood_category, mood_score)
            
            return {
                'mood_score': mood_score,
                'mood_category': mood_category,
                'confidence': overall_confidence,
                'description': self.mood_categories[mood_category]['description'],
                'recommendations': recommendations,
                'breakdown': {
                    'text_contribution': {
                        'score': text_score,
                        'confidence': text_confidence,
                        'weight': self.weights['text_sentiment']
                    },
                    'facial_contribution': {
                        'scores': facial_scores,
                        'confidence': facial_confidence,
                        'weight': self.weights['facial_emotions']
                    }
                }
            }
            
        except Exception as e:
            return {
                'mood_score': 0.0,
                'mood_category': 'neutral',
                'confidence': 0.0,
                'description': 'Unable to determine mood',
                'error': str(e),
                'recommendations': ['Please try again with clearer input']
            }
    
    def _extract_facial_scores(self, facial_emotions: Dict[str, Any]) -> Dict[str, float]:
        """Extract emotion scores from facial analysis results."""
        emotions = facial_emotions.get('emotions', {})
        
        # Map facial emotions to sentiment-like scores
        emotion_scores = {
            'happy': 0.8,
            'excited': 0.9,
            'joyful': 0.9,
            'content': 0.6,
            'pleased': 0.7,
            'neutral': 0.0,
            'calm': 0.1,
            'sad': -0.7,
            'disappointed': -0.6,
            'frustrated': -0.5,
            'angry': -0.8,
            'depressed': -0.9,
            'anxious': -0.6,
            'fearful': -0.7,
            'surprised': 0.3,
            'disgusted': -0.5
        }
        
        # Calculate weighted facial sentiment score
        facial_sentiment = 0.0
        total_weight = 0.0
        
        for emotion, confidence in emotions.items():
            if emotion in emotion_scores:
                facial_sentiment += emotion_scores[emotion] * confidence
                total_weight += confidence
        
        if total_weight > 0:
            facial_sentiment /= total_weight
        
        return {
            'sentiment': facial_sentiment,
            'emotions': emotions
        }
    
    def _calculate_mood_score(self, text_score: float, text_confidence: float,
                            facial_scores: Dict[str, Any], facial_confidence: float) -> float:
        """Calculate overall mood score from text and facial data."""
        facial_sentiment = facial_scores.get('sentiment', 0.0)
        
        # Weight the scores by confidence and predefined weights
        text_weighted = text_score * text_confidence * self.weights['text_sentiment']
        facial_weighted = facial_sentiment * facial_confidence * self.weights['facial_emotions']
        
        # Calculate total weight for normalization
        total_weight = (text_confidence * self.weights['text_sentiment'] + 
                       facial_confidence * self.weights['facial_emotions'])
        
        if total_weight > 0:
            mood_score = (text_weighted + facial_weighted) / total_weight
        else:
            # Fallback to simple average if no confidence
            mood_score = (text_score + facial_sentiment) / 2
        
        # Clamp between -1 and 1
        return max(-1.0, min(1.0, mood_score))
    
    def _categorize_mood(self, mood_score: float, facial_scores: Dict[str, Any]) -> str:
        """Categorize mood score into predefined categories."""
        # Check for specific emotional indicators that might override score
        emotions = facial_scores.get('emotions', {})
        
        # Check for extreme emotions that should override score
        if emotions.get('angry', 0) > 0.7 or emotions.get('depressed', 0) > 0.7:
            return 'very_negative'
        elif emotions.get('happy', 0) > 0.7 or emotions.get('excited', 0) > 0.7:
            return 'very_positive'
        
        # Use score-based categorization
        for category, info in self.mood_categories.items():
            min_score, max_score = info['sentiment_range']
            if min_score <= mood_score <= max_score:
                return category
        
        # Default to neutral if no category matches
        return 'neutral'
    
    def _calculate_overall_confidence(self, text_confidence: float, facial_confidence: float) -> float:
        """Calculate overall confidence in the mood assessment."""
        # Weighted average of confidences
        total_weight = self.weights['text_sentiment'] + self.weights['facial_emotions']
        
        if total_weight > 0:
            overall_confidence = (
                text_confidence * self.weights['text_sentiment'] +
                facial_confidence * self.weights['facial_emotions']
            ) / total_weight
        else:
            overall_confidence = (text_confidence + facial_confidence) / 2
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _generate_recommendations(self, mood_category: str, mood_score: float) -> list:
        """Generate recommendations based on detected mood."""
        recommendations = []
        
        if mood_category in ['very_negative', 'negative']:
            recommendations.extend([
                "Consider taking a break or doing something you enjoy",
                "Try deep breathing exercises or meditation",
                "Talk to someone you trust about how you're feeling",
                "Consider professional support if these feelings persist"
            ])
        elif mood_category == 'neutral':
            recommendations.extend([
                "Your mood seems balanced - maintain your current routine",
                "Consider trying something new to add excitement",
                "Stay connected with friends and family"
            ])
        elif mood_category in ['positive', 'very_positive']:
            recommendations.extend([
                "Great! Keep doing what makes you happy",
                "Share your positive energy with others",
                "Consider documenting what's working well for you"
            ])
        
        # Add general recommendations
        recommendations.extend([
            "Remember that moods can change and that's normal",
            "Take care of your physical health with good sleep and nutrition",
            "Don't hesitate to seek help if you need it"
        ])
        
        return recommendations[:4]  # Return top 4 recommendations

