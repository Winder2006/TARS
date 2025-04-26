import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import librosa
import pickle
import os
from pathlib import Path

class SentimentAnalyzer:
    """Analyzes sentiment from both text and voice audio"""
    
    def __init__(self):
        # Initialize the VADER sentiment analyzer for text
        self.text_analyzer = SentimentIntensityAnalyzer()
        
        # Create paths for storing sentiment models
        self.models_dir = Path("memory/sentiment")
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Path for voice sentiment model
        self.voice_model_path = self.models_dir / "voice_sentiment_model.pkl"
        
        # Initialize voice sentiment features
        self.voice_features = {
            "pitch_mean": None,
            "pitch_std": None, 
            "energy_mean": None,
            "energy_std": None,
            "speech_rate": None
        }
        
        # Load voice model if available
        self.voice_model = self._load_voice_model()
        
    def _load_voice_model(self):
        """Load the voice sentiment model if available"""
        if os.path.exists(self.voice_model_path):
            try:
                with open(self.voice_model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading voice sentiment model: {e}")
        return None
        
    def analyze_text(self, text):
        """Analyze sentiment from text using VADER"""
        if not text:
            return {
                "compound": 0.0,
                "pos": 0.0,
                "neu": 1.0,
                "neg": 0.0,
                "emotion": "neutral"
            }
            
        # Get VADER sentiment scores
        sentiment = self.text_analyzer.polarity_scores(text)
        
        # Add emotion label
        if sentiment["compound"] >= 0.05:
            emotion = "positive"
        elif sentiment["compound"] <= -0.05:
            emotion = "negative"
        else:
            emotion = "neutral"
            
        # Add intensity
        intensity = abs(sentiment["compound"])
        if intensity > 0.75:
            intensity_label = "very"
        elif intensity > 0.5:
            intensity_label = "moderately"
        elif intensity > 0.25:
            intensity_label = "slightly"
        else:
            intensity_label = "mildly"
            
        # Enhance with more specific emotion detection
        emotion_details = self._detect_specific_emotions(text)
            
        # Return comprehensive results
        return {
            "compound": sentiment["compound"],
            "pos": sentiment["pos"],
            "neu": sentiment["neu"],
            "neg": sentiment["neg"],
            "emotion": emotion,
            "intensity": intensity,
            "intensity_label": intensity_label,
            "emotion_details": emotion_details
        }
        
    def _detect_specific_emotions(self, text):
        """Detect more specific emotions from text"""
        text = text.lower()
        
        emotion_keywords = {
            "joy": ["happy", "joy", "delighted", "excited", "pleased", "glad", "thrilled", "wonderful"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "outraged"],
            "sadness": ["sad", "unhappy", "depressed", "down", "blue", "gloomy", "heartbroken"],
            "fear": ["afraid", "scared", "fearful", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["surprised", "shocked", "astonished", "amazed", "stunned", "unexpected"],
            "disgust": ["disgusted", "repulsed", "revolted", "gross", "yuck", "ew"],
            "confusion": ["confused", "puzzled", "perplexed", "uncertain", "unsure", "don't understand"]
        }
        
        # Count matches for each emotion
        emotion_counts = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            if count > 0:
                emotion_counts[emotion] = count
                
        # Return the emotions found, sorted by count
        return sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
    def analyze_voice(self, audio_data):
        """Extract emotional features from voice audio"""
        try:
            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                if hasattr(audio_data, 'read'):
                    audio_data.seek(0)
                    audio, sr = librosa.load(audio_data, sr=None)
                else:
                    return {"error": "Unsupported audio format"}
            else:
                audio = audio_data
                sr = 44100  # Assume default sample rate
                
            # Extract pitch features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Extract energy/volume features
            rms = librosa.feature.rms(y=audio)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            # Calculate speech rate (approximate)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onsets) > 1:
                # Roughly estimate syllables per second
                speech_rate = len(onsets) / (len(audio) / sr)
            else:
                speech_rate = 0
                
            # Store for potential model training
            self.voice_features = {
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "speech_rate": speech_rate
            }
            
            # Simple heuristic model if no trained model available
            if self.voice_model is None:
                # High energy + pitch variation often indicates excitement or anger
                if energy_mean > 0.1 and pitch_std > 20:
                    if pitch_mean > 200:
                        emotion = "excited"
                    else:
                        emotion = "angry"
                # Low energy, low pitch often indicates sadness
                elif energy_mean < 0.05 and pitch_mean < 180:
                    emotion = "sad"
                # Fast speech can indicate nervousness
                elif speech_rate > 4:
                    emotion = "nervous"
                # Default to neutral
                else:
                    emotion = "neutral"
                    
                confidence = 0.6  # Heuristic model has moderate confidence
            else:
                # Use trained model (not implemented yet)
                emotion = "neutral"
                confidence = 0.5
                
            return {
                "emotion": emotion,
                "confidence": confidence,
                "features": self.voice_features
            }
            
        except Exception as e:
            print(f"Error in voice sentiment analysis: {e}")
            return {
                "emotion": "unknown",
                "error": str(e),
                "confidence": 0
            }
    
    def combine_analysis(self, text_sentiment, voice_sentiment):
        """Combine text and voice sentiment for a more complete picture"""
        # Start with text sentiment as base
        combined = {
            "primary_emotion": text_sentiment["emotion"],
            "confidence": 0.7,  # Default confidence
            "text_emotion": text_sentiment["emotion"],
            "voice_emotion": voice_sentiment.get("emotion", "unknown"),
            "mismatch": False
        }
        
        # Handle cases where voice and text emotions differ
        if voice_sentiment.get("emotion") != "unknown" and voice_sentiment.get("emotion") != text_sentiment["emotion"]:
            # Flag potential sarcasm or mixed emotions
            combined["mismatch"] = True
            
            # When voice emotion is strong, give it more weight
            voice_confidence = voice_sentiment.get("confidence", 0)
            if voice_confidence > 0.7:
                combined["primary_emotion"] = voice_sentiment["emotion"]
                combined["confidence"] = voice_confidence
                
        return combined

    def get_response_style(self, sentiment_result):
        """Recommend response style based on sentiment analysis"""
        emotion = sentiment_result.get("primary_emotion", "neutral")
        mismatch = sentiment_result.get("mismatch", False)
        
        # Default style
        style = {
            "voice": "default",
            "tone": "neutral",
            "approach": "informative"
        }
        
        # Adjust based on detected emotion
        if emotion == "positive":
            style["voice"] = "friendly"
            style["tone"] = "upbeat"
            style["approach"] = "enthusiastic"
        elif emotion == "negative":
            style["voice"] = "professional"
            style["tone"] = "supportive"
            style["approach"] = "helpful"
        elif emotion == "angry":
            style["voice"] = "professional"
            style["tone"] = "calm"
            style["approach"] = "diplomatic"
        elif emotion == "sad":
            style["voice"] = "friendly"
            style["tone"] = "gentle"
            style["approach"] = "supportive"
        elif emotion == "nervous" or emotion == "fear":
            style["voice"] = "professional"
            style["tone"] = "reassuring"
            style["approach"] = "clear"
            
        # Adjust for sarcasm/mismatch
        if mismatch:
            style["approach"] = "careful"
            
        return style

# Simple test function
def test_sentiment_analyzer():
    analyzer = SentimentAnalyzer()
    
    # Test with different texts
    test_texts = [
        "I'm having a great day today!",
        "This is absolutely terrible.",
        "I'm feeling a bit anxious about the meeting.",
        "I don't really care either way.",
        "I'm both excited and nervous about the new job."
    ]
    
    print("Testing Sentiment Analyzer:")
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: '{text}'")
        print(f"Emotion: {result['emotion']} ({result['intensity_label']})")
        print(f"Score: {result['compound']:.2f}")
        if result.get('emotion_details'):
            print(f"Details: {result['emotion_details']}")
        
        # Get response style
        style = analyzer.get_response_style({"primary_emotion": result["emotion"]})
        print(f"Recommended response: Voice={style['voice']}, Tone={style['tone']}, Approach={style['approach']}")
    
if __name__ == "__main__":
    test_sentiment_analyzer() 