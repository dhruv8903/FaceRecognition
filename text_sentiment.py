import os
from google.cloud import language_v1
from google.cloud.language_v1 import types
import json

class TextSentimentAnalyzer:
    def __init__(self):
        """Initialize the Google Cloud Natural Language API client."""
        # Check if credentials are available
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set. Using local fallback.")
            self.use_local_fallback = True
        else:
            self.use_local_fallback = False
            try:
                self.client = language_v1.LanguageServiceClient()
            except Exception as e:
                print(f"Failed to initialize Google Cloud client: {e}")
                self.use_local_fallback = True
    
    def analyze(self, text):
        """
        Analyze text sentiment using Google Cloud Natural Language API.
        Falls back to local analysis if Google Cloud is not available.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if self.use_local_fallback:
            return self._local_sentiment_analysis(text)
        
        try:
            # Use Google Cloud Natural Language API
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            # Analyze sentiment
            response = self.client.analyze_sentiment(
                request={'document': document}
            )
            
            sentiment = response.document_sentiment
            
            return {
                'score': sentiment.score,
                'magnitude': sentiment.magnitude,
                'label': self._get_sentiment_label(sentiment.score),
                'confidence': abs(sentiment.score),
                'source': 'google_cloud'
            }
            
        except Exception as e:
            print(f"Google Cloud API error: {e}")
            return self._local_sentiment_analysis(text)
    
    def _local_sentiment_analysis(self, text):
        """
        Fallback local sentiment analysis using simple keyword matching.
        This is a basic implementation for privacy-focused local processing.
        """
        # Simple keyword-based sentiment analysis
        positive_words = [
            'happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic',
            'love', 'like', 'good', 'excellent', 'awesome', 'brilliant', 'perfect',
            'smile', 'laugh', 'fun', 'enjoy', 'pleased', 'satisfied', 'content'
        ]
        
        negative_words = [
            'sad', 'angry', 'mad', 'upset', 'disappointed', 'frustrated', 'annoyed',
            'hate', 'dislike', 'bad', 'terrible', 'awful', 'horrible', 'worst',
            'cry', 'tears', 'depressed', 'anxious', 'worried', 'stressed', 'tired'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total_words = len(text.split())
        if total_words == 0:
            score = 0
        else:
            score = (positive_count - negative_count) / max(total_words, 1)
            score = max(-1, min(1, score))  # Clamp between -1 and 1
        
        # Calculate magnitude (strength of sentiment)
        magnitude = (positive_count + negative_count) / max(total_words, 1)
        
        return {
            'score': score,
            'magnitude': magnitude,
            'label': self._get_sentiment_label(score),
            'confidence': min(magnitude, 1.0),
            'source': 'local_fallback'
        }
    
    def _get_sentiment_label(self, score):
        """Convert sentiment score to human-readable label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'

