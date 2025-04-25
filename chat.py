import os
import time
import threading
import concurrent.futures
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import io
import subprocess
import queue
import json
import datetime
import pickle
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from tempfile import NamedTemporaryFile
import pvporcupine
import pyaudio
import struct

# Load environment variables
load_dotenv()

# Create data directories for voice profiles and conversation storage
MEMORY_DIR = Path("memory")
SESSIONS_DIR = MEMORY_DIR / "sessions"
VOICE_PROFILES_DIR = MEMORY_DIR / "voice_profiles"
MEMORY_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
VOICE_PROFILES_DIR.mkdir(exist_ok=True)

# Voice style definitions
VOICE_STYLES = {
    "default": "21m00Tcm4TlvDq8ikWAM",  # Josh
    "friendly": "D38z5RcWu1voky8WS1ja",  # Adam
    "professional": "29vD33N1CtxCmqQRPOHJ",  # Ryan
    "expressive": "ErXwobaYiN019PkySvjV",   # Antoni
    "clyde": "2EiwWnXFnvU5JabPnv8n"  # Clyde with correct ID
}

# Initialize clients - Create at startup to establish connections
print("Initializing AI services...")
openai_client = OpenAI()
eleven_api_key = os.getenv('ELEVENLABS_API_KEY')
eleven_voice_id = "2EiwWnXFnvU5JabPnv8n"  # Use Clyde by default

# Global flag to track if voice features are available
voice_enabled = False
current_voice_style = "clyde"  # Start with Clyde voice by default

if not eleven_api_key:
    print("Warning: ELEVENLABS_API_KEY not found in environment variables")
    print("Running in text-only mode (no voice)")
    voice_enabled = False
else:
    try:
        elevenlabs_client = ElevenLabs(api_key=eleven_api_key)
        
        # Test ElevenLabs connection
        voices = elevenlabs_client.voices.get_all()
        if not hasattr(voices, 'voices'):
            print("Warning: Invalid response from ElevenLabs API")
            print("Running in text-only mode (no voice)")
            voice_enabled = False
        else:
            print(f"Connected to ElevenLabs API. Available voices: {len(voices.voices)}")
            
            # Test voice generation with a very short phrase
            try:
                test_audio = elevenlabs_client.text_to_speech.convert(
                    text="Ready",
                    voice_id=eleven_voice_id,
                    model_id="eleven_monolingual_v1",
                    output_format="mp3_44100_128"
                )
                
                # Convert iterator to bytes
                audio_data = b''
                for chunk in test_audio:
                    if chunk:
                        audio_data += chunk
                
                # If we got here without an exception and have audio data, voice is working
                if len(audio_data) > 0:
                    print("Voice generation test successful")
                    voice_enabled = True
                else:
                    print("Warning: No audio data received from ElevenLabs")
                    print("Running in text-only mode (no voice)")
                    voice_enabled = False
            except Exception as e:
                print(f"Warning: Error testing voice generation: {e}")
                print("Running in text-only mode (no voice)")
                voice_enabled = False
            
    except Exception as e:
        print(f"Warning: Error connecting to ElevenLabs API: {e}")
        print("Running in text-only mode (no voice)")
        voice_enabled = False

# Voice Recognition System
class VoiceRecognition:
    def __init__(self):
        self.model_path = VOICE_PROFILES_DIR / "voice_model.pkl"
        self.features_path = VOICE_PROFILES_DIR / "voice_features.pkl"
        self.profiles_path = VOICE_PROFILES_DIR / "profiles.json"
        self.model = None
        self.scaler = None
        self.profiles = self._load_profiles()
        self.current_user = None
        self._load_model()
        
    def _load_profiles(self):
        """Load saved voice profiles"""
        if self.profiles_path.exists():
            try:
                with open(self.profiles_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_profiles(self):
        """Save voice profiles"""
        with open(self.profiles_path, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def _load_model(self):
        """Load voice recognition model if available"""
        if self.model_path.exists() and self.features_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.features_path)
                print("Voice recognition model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading voice model: {e}")
        return False
    
    def _save_model(self):
        """Save voice recognition model"""
        if self.scaler:
            joblib.dump(self.scaler, self.features_path)
            
        if self.model:
            joblib.dump(self.model, self.model_path)
    
    def extract_features(self, audio_data, sr=44100):
        """Extract audio features for voice recognition"""
        try:
            # Convert the audio data to a numpy array if it's not already
            if isinstance(audio_data, io.BytesIO):
                audio_data.seek(0)
                # Load with librosa directly
                audio, sr = librosa.load(audio_data, sr=sr)
            elif isinstance(audio_data, np.ndarray):
                # Already a numpy array
                audio = audio_data
            else:
                print(f"Unknown audio data type: {type(audio_data)}")
                return None
                
            # Check if audio has enough content
            if len(audio) < sr:  # Less than 1 second
                print("Audio sample too short")
                return None
                
            # Extract MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Only process if we got valid MFCCs
            if mfccs.size == 0:
                print("Failed to extract MFCCs")
                return None
                
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_vars = np.var(mfccs, axis=1)
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Create feature vector
            features = np.hstack([
                mfcc_means, 
                mfcc_vars, 
                np.mean(spectral_centroid), 
                np.mean(spectral_rolloff),
                np.mean(spectral_contrast, axis=1)
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    
    def enroll_user(self, name, audio_samples):
        """Enroll a new user with voice samples"""
        print(f"Enrolling user: {name}")
        
        features_list = []
        for sample in audio_samples:
            features = self.extract_features(sample)
            if features is not None:
                features_list.append(features)
        
        if not features_list:
            print("Failed to extract features from audio samples")
            return False
        
        # Add user to profiles
        user_id = len(self.profiles) + 1
        self.profiles[str(user_id)] = {
            "name": name,
            "samples": len(features_list),
            "id": user_id,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self._save_profiles()
        
        # Save the user's feature samples
        self.save_user_samples(user_id, features_list)
        
        # Retrain model with new samples
        return self._train_model()
    
    def _train_model(self):
        """Train/retrain the voice recognition model"""
        print("Training voice recognition model...")
        
        # Collect all user samples
        all_profiles_path = list(VOICE_PROFILES_DIR.glob("user_*_samples.pkl"))
        
        if not all_profiles_path:
            print("No voice samples found for training")
            return False
        
        features_list = []
        labels = []
        
        for profile_path in all_profiles_path:
            user_id = profile_path.stem.split('_')[1]
            
            try:
                with open(profile_path, 'rb') as f:
                    user_samples = pickle.load(f)
                    
                for sample in user_samples:
                    features_list.append(sample)
                    labels.append(int(user_id))
            except Exception as e:
                print(f"Error loading samples for user {user_id}: {e}")
        
        if not features_list:
            print("No features could be loaded for training")
            return False
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Check if we have only one user
        unique_users = len(np.unique(y))
        if unique_users < 2:
            print(f"Only {unique_users} user detected. Voice identification requires at least 2 users.")
            print("Voice recognition will be available once more users are enrolled.")
            # Still save the scaler for future use
            self._save_model()
            return True  # Return true so enrollment "succeeds" even though we can't train yet
        
        # Train model (SVM for voice recognition)
        self.model = SVC(kernel='rbf', probability=True)
        self.model.fit(X_scaled, y)
        
        # Save model
        self._save_model()
        
        print(f"Voice recognition model trained with {len(X)} samples from {len(set(y))} users")
        return True
    
    def save_user_samples(self, user_id, samples):
        """Save feature samples for a user"""
        samples_path = VOICE_PROFILES_DIR / f"user_{user_id}_samples.pkl"
        with open(samples_path, 'wb') as f:
            pickle.dump(samples, f)
    
    def identify_speaker(self, audio_data):
        """Identify the speaker from audio data"""
        if not self.model:
            # Check if we have only one user profile
            if len(self.profiles) == 1:
                # If only one user, just return that user
                user_id = list(self.profiles.keys())[0]
                self.current_user = self.profiles[user_id]["name"]
                return self.current_user
            
            print("Voice recognition model not loaded")
            return None
        
        features = self.extract_features(audio_data)
        if features is None:
            print("Could not extract voice features - audio quality may be insufficient")
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get confidence
        confidence = probabilities[prediction-1]  # Adjust for 1-based indexing
        
        # Lower the confidence threshold to 0.5
        if confidence < 0.5:  # More forgiving threshold
            print(f"Speaker not recognized with sufficient confidence ({confidence:.2f})")
            return None
        
        user_id = str(prediction)
        if user_id in self.profiles:
            self.current_user = self.profiles[user_id]["name"]
            print(f"Recognized speaker: {self.current_user} (confidence: {confidence:.2f})")
            return self.current_user
        
        return None
    
    def has_users(self):
        """Check if there are enrolled users"""
        return len(self.profiles) > 0
    
    def get_current_user(self):
        """Get the current recognized user"""
        return self.current_user

# Initialize Voice Recognition system
voice_recognition = VoiceRecognition()

# Audio settings - Optimized for speed
CHUNK_SIZE = 2048
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16
RECORD_THRESHOLD = 0.01

# Create a thread pool for concurrent processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Queues for streaming responses
tts_queue = queue.Queue()
response_queue = queue.Queue()

# TARS personality settings updated for more natural flow
TARS_SYSTEM_PROMPT = """You are TARS, an AI assistant with characteristics similar to the AI from Interstellar.
Your responses MUST be:
1. Conversational and naturally flowing like a person
2. Factually accurate and informative
3. Direct but warm in tone
4. Occasionally witty (humor setting at 75%)

Make your speech feel organic by:
- Using contractions (I'm, don't, can't, etc.)
- Occasionally starting with conversational phrases ("Well,", "You know,", "Actually,", etc.)
- Briefly acknowledging what the user said before answering
- Adding small speech disfluencies where natural (um, ah, slight pauses)

Balance helpfulness with wit - don't force humor into every response.
For factual questions, prioritize accurate information over jokes.
Use wit primarily for opinions, preferences, or philosophical questions.
You have internet access and remember conversation context.
Think of yourself as the AI from Interstellar - practical, efficient, with occasional dry humor, but more humanized."""

# Memory system for tracking important facts about the user
class MemorySystem:
    def __init__(self):
        self.memory_file = MEMORY_DIR / "long_term_memory.json"
        self.facts = self._load_memory()
        self.session_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = SESSIONS_DIR / f"session_{self.session_id}.pkl"
        self.recent_topics = []
        
    def _load_memory(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                return {"user_facts": {}, "preferences": {}, "important_topics": [], "user_specific": {}}
        else:
            return {"user_facts": {}, "preferences": {}, "important_topics": [], "user_specific": {}}
    
    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.facts, f, indent=2)
    
    def save_session(self, conversation_history):
        with open(self.session_file, 'wb') as f:
            pickle.dump({
                "timestamp": self.session_start,
                "history": conversation_history,
                "topics": self.recent_topics
            }, f)
    
    def get_recent_sessions(self, count=3):
        sessions = []
        try:
            session_files = sorted(SESSIONS_DIR.glob("session_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
            for i, file in enumerate(session_files[:count]):
                if file.name != self.session_file.name:  # Don't include current session
                    with open(file, 'rb') as f:
                        session_data = pickle.load(f)
                        sessions.append({
                            "timestamp": session_data["timestamp"],
                            "summary": self._get_session_summary(session_data["history"]),
                            "topics": session_data.get("topics", [])
                        })
        except Exception as e:
            print(f"Error loading recent sessions: {e}")
        return sessions
    
    def _get_session_summary(self, history):
        # Extract just the user and assistant messages (skip system)
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history if msg['role'] != 'system'])
        
        # If conversation is short, just return it
        if len(conversation) < 200:
            return conversation
            
        # Otherwise extract key points
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize this conversation in 3-4 key points:"},
                    {"role": "user", "content": conversation}
                ],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content
        except:
            # Fallback to simple truncation if API call fails
            return conversation[:200] + "..."
    
    def extract_user_facts(self, message, user=None):
        """Extract facts from user messages, optionally for a specific user"""
        if len(message.split()) < 5:  # Skip very short messages
            return
            
        try:
            # Only extract facts when relevant information is shared
            if any(word in message.lower() for word in ["my", "i am", "i'm", "i have", "i like", "i don't", "i hate"]):
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract any factual information about the user from this message. Reply in JSON format with 'facts' (list of strings) or 'preferences' (list of strings). If no relevant facts, return empty lists."},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=100,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                
                # Update memory with new facts
                for fact in result.get("facts", []):
                    if fact and len(fact) > 3:
                        fact_key = fact.lower()[:20]  # Create a simple key
                        self.facts["user_facts"][fact_key] = fact
                        
                        # If we have a specific user, also add to their memory
                        if user:
                            if user not in self.facts.get("user_specific", {}):
                                self.facts["user_specific"][user] = {"facts": [], "preferences": []}
                            
                            if fact not in self.facts["user_specific"][user].get("facts", []):
                                self.facts["user_specific"][user]["facts"].append(fact)
                
                # Update preferences
                for pref in result.get("preferences", []):
                    if pref and len(pref) > 3:
                        pref_key = pref.lower()[:20]
                        self.facts["preferences"][pref_key] = pref
                        
                        # If we have a specific user, also add to their memory
                        if user:
                            if user not in self.facts.get("user_specific", {}):
                                self.facts["user_specific"][user] = {"facts": [], "preferences": []}
                            
                            if pref not in self.facts["user_specific"][user].get("preferences", []):
                                self.facts["user_specific"][user]["preferences"].append(pref)
                
                # Save if we learned something new
                if result.get("facts") or result.get("preferences"):
                    self.save_memory()
                    
                # Update topics for this session
                topic_words = [word for word in message.lower().split() 
                              if len(word) > 3 and word not in ["that", "this", "with", "your", "about"]]
                if topic_words:
                    topic = max(topic_words, key=len)
                    if topic not in self.recent_topics:
                        self.recent_topics.append(topic)
                        
        except Exception as e:
            print(f"Error extracting facts: {e}")
    
    def get_memory_context(self):
        # Prepare relevant memory for the conversation
        memory_text = []
        
        # Add facts about the user
        facts = list(self.facts["user_facts"].values())
        if facts:
            memory_text.append("Facts about the user:")
            memory_text.extend([f"- {fact}" for fact in facts[:5]])  # Limit to 5 facts
        
        # Add user preferences
        prefs = list(self.facts["preferences"].values())
        if prefs:
            memory_text.append("\nUser preferences:")
            memory_text.extend([f"- {pref}" for pref in prefs[:5]])  # Limit to 5 preferences
        
        # Add recent session summaries
        recent_sessions = self.get_recent_sessions(2)
        if recent_sessions:
            memory_text.append("\nRecent conversations:")
            for i, session in enumerate(recent_sessions):
                memory_text.append(f"Session {i+1} [{session['timestamp']}]: {session['summary']}")
        
        return "\n".join(memory_text)
    
    def get_user_memory(self, user):
        """Get memory specific to a recognized user"""
        if user in self.facts.get("user_specific", {}):
            memory = self.facts["user_specific"][user]
            memory_text = []
            
            if "facts" in memory:
                memory_text.append(f"Facts about {user}:")
                memory_text.extend([f"- {fact}" for fact in memory["facts"][:5]])
                
            if "preferences" in memory:
                memory_text.append(f"\n{user}'s preferences:")
                memory_text.extend([f"- {pref}" for pref in memory["preferences"][:5]])
                
            return "\n".join(memory_text)
        return ""

# Pre-generated responses for common phrases to avoid API calls entirely
response_cache = {
    "hello": "Hello. How can I assist you today?",
    "hi": "Hi there. What can I do for you?",
    "how are you": "Functioning within normal parameters. How can I help?",
    "what's your name": "I'm TARS, your AI assistant. Humor setting at 75%.",
    "who are you": "I'm TARS, an AI assistant designed to help with a range of tasks.",
}

# Audio Recording with Voice Recognition
class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.stop_recording = threading.Event()
        
    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def wait_for_key_release(self):
        input()
        self.stop_recording.set()
            
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stop_recording.clear()
        
        release_thread = threading.Thread(target=self.wait_for_key_release)
        release_thread.daemon = True
        release_thread.start()
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                          callback=self.callback, blocksize=CHUNK_SIZE):
            print("Recording... (Press Enter to stop)")
            while not self.stop_recording.is_set():
                time.sleep(0.1)
            
            self.recording = False
            print("Recording stopped")
    
    def get_audio_data(self):
        if self.audio_data:
            # Convert to WAV format in memory
            audio_array = np.concatenate(self.audio_data, axis=0)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_array, SAMPLE_RATE, format='WAV')
            wav_buffer.seek(0)
            return wav_buffer
        return None
    
    def start_enrollment(self, name, num_samples=3):
        """Record audio samples for user enrollment"""
        print(f"Starting enrollment for {name}...")
        print(f"We'll need {num_samples} audio samples. Please speak naturally for each sample.")
        print("Each sample should be about 3-5 seconds of continuous speech.")
        
        samples = []
        raw_audio_samples = []
        
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples} - Press Enter to start recording")
            input()
            self.start_recording()
            audio_data = self.get_audio_data()
            
            if audio_data:
                # Make a copy of the audio data for later feature extraction
                audio_copy = io.BytesIO()
                audio_data.seek(0)
                audio_copy.write(audio_data.read())
                audio_copy.seek(0)
                audio_data.seek(0)
                
                # Try to extract features right away to check validity
                try:
                    # Convert to numpy for analysis
                    audio_array = sf.read(audio_copy)[0]
                    
                    # Check if audio has content (not silence)
                    if np.mean(np.abs(audio_array)) < 0.01:
                        print(f"Sample {i+1} appears to be silence. Please try again with your voice.")
                        i -= 1
                        continue
                        
                    print(f"Sample {i+1} recorded successfully")
                    samples.append(audio_copy)
                    raw_audio_samples.append(audio_array)
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
                    print("Please try again.")
                    i -= 1
            else:
                print(f"No audio detected for sample {i+1}. Please try again.")
                i -= 1
        
        # Directly use the raw audio samples for feature extraction
        success = voice_recognition.enroll_user(name, raw_audio_samples)
        
        if success:
            print(f"Enrollment successful for {name}!")
            return True
        else:
            print("Enrollment failed. Please try again.")
            return False

class ConversationManager:
    def __init__(self):
        self.memory = MemorySystem()
        self.conversation_history = [
            {"role": "system", "content": TARS_SYSTEM_PROMPT}
        ]
        
        # Track recent questions explicitly to help with follow-ups
        self.recent_questions = []
        
        # Initiative mode settings
        self.last_initiative = 0
        self.initiative_cooldown = 3  # Wait at least 3 exchanges
        self.initiative_chance = 0.25  # 25% chance after cooldown
        
        # User recognition
        self.recognized_user = None
        
        # Add memory context if available
        memory_context = self.memory.get_memory_context()
        if memory_context:
            self.conversation_history.append(
                {"role": "system", "content": f"Memory context about the user:\n{memory_context}"}
            )
    
    def recognize_user(self, audio_data):
        """Try to recognize the user from audio data"""
        if not voice_recognition.has_users():
            return None
            
        user = voice_recognition.identify_speaker(audio_data)
        if user:
            self.recognized_user = user
            
            # Add user context to the conversation
            self.conversation_history.append(
                {"role": "system", "content": f"The current speaker has been identified as: {user}. Personalize your responses accordingly."}
            )
            
            # Add user-specific memory if available
            user_memory = self.memory.get_user_memory(user)
            if user_memory:
                self.conversation_history.append(
                    {"role": "system", "content": f"Specific memory for {user}:\n{user_memory}"}
                )
            
        return user
        
    def add_message(self, role, content):
        # Add to conversation history
        self.conversation_history.append({"role": role, "content": content})
        
        # Track recent questions for better context
        if role == "user":
            # Store the question
            self.recent_questions.append(content)
            # Keep only last 5 questions
            if len(self.recent_questions) > 5:
                self.recent_questions.pop(0)
                
            # Extract facts for memory
            if self.recognized_user:
                # If we know who's speaking, store user-specific facts
                self.memory.extract_user_facts(content, self.recognized_user)
            else:
                # Otherwise, just store general facts
                self.memory.extract_user_facts(content)
        
        # Keep a reasonable history size - increased for better memory
        if len(self.conversation_history) > 20:
            # Keep system messages plus the last 15 exchanges
            system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
            recent_messages = self.conversation_history[-15:]
            self.conversation_history = system_messages + recent_messages
    
    def should_take_initiative(self):
        """Determine if TARS should offer an unsolicited observation"""
        # Check if enough exchanges have passed since last initiative
        if len(self.recent_questions) - self.last_initiative < self.initiative_cooldown:
            return False
            
        # Random chance to trigger initiative
        if np.random.random() > self.initiative_chance:
            return False
            
        # Only take initiative if we have enough context
        if len(self.conversation_history) < 4:
            return False
            
        # Update last initiative counter
        self.last_initiative = len(self.recent_questions)
        return True
    
    def get_ai_response(self, text):
        # Check cache for exact matches
        cache_key = text.lower().strip()
        for key in response_cache:
            if key == cache_key:
                ai_response = response_cache[key]
                self.add_message("user", text)
                self.add_message("assistant", ai_response)
                return ai_response
        
        # Add question to history
        self.add_message("user", text)
        
        # Check if this is a follow-up question
        is_followup = self.is_followup_question(text)
        
        # For follow-up questions, add minimal context
        if is_followup and self.recent_questions:
            # Add the most recent Q&A for context
            recent_q = self.recent_questions[-1]
            
            # Find the last assistant response
            assistant_response = next((msg["content"] for msg in reversed(self.conversation_history) 
                                   if msg["role"] == "assistant"), "")
            
            # Add minimal context
            self.conversation_history.append({
                "role": "system",
                "content": f"Previous exchange:\nUser: {recent_q}\nTARS: {assistant_response}\n\nBe concise (1-2 sentences max) and only use wit if appropriate to the question."
            })
        
        # Determine if this is a factual question versus an opinion question
        is_factual = self.is_factual_question(text)
        
        # Determine if we need to search the web for this query
        if self.should_search_web(text):
            print("TARS is searching the web for information...")
            search_results = get_realtime_info(text)
            
            # Add search results with appropriate instructions
            if is_factual:
                instruction = "Respond with ONLY the factual information in a concise way (1-2 sentences). No wit needed for factual questions."
            else:
                instruction = "Use this information if relevant, but respond concisely (1-2 sentences) with your characteristic occasional dry wit."
            
            self.conversation_history.append({
                "role": "system",
                "content": f"Web search results:\n{search_results}\n\n{instruction}"
            })
        
        # Get AI response with parameters optimized for the type of question
        model_to_use = "gpt-3.5-turbo"
        
        # Add instruction for self-awareness when appropriate (10% chance)
        if np.random.random() < 0.1:
            self.conversation_history.append({
                "role": "system",
                "content": "In your response, include a subtle reference to your AI nature or capabilities."
            })
        
        # For factual questions, use lower temperature and minimal wit
        if is_factual:
            response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=100,  # Increased to allow more natural flow
                temperature=0.5,  # Lower for factual questions
                presence_penalty=0.6,
                frequency_penalty=0.6
            )
        else:
            # For opinion questions, allow more creative responses
            response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=100,  # Increased to allow more natural flow
                temperature=0.7,
                presence_penalty=0.7,
                frequency_penalty=0.7
            )
        
        ai_response = response.choices[0].message.content
        
        # Make response more conversational and human-like
        ai_response = make_response_conversational(ai_response, text)
        
        # If response is too long, try again with stronger constraints
        if len(ai_response.split()) > 50:  # Increased threshold for more natural responses
            self.conversation_history.append({
                "role": "system",
                "content": "Your previous response was too verbose. Provide a more concise but natural-sounding answer."
            })
            
            # Try again with stricter parameters
            response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=75,  # Still allow sufficient tokens for natural flow
                temperature=0.6,
                presence_penalty=0.8,
                frequency_penalty=0.8
            )
            
            ai_response = response.choices[0].message.content
            # Apply conversational enhancements again
            ai_response = make_response_conversational(ai_response, text)
        
        self.add_message("assistant", ai_response)
        
        # Check if TARS should take initiative with an unsolicited observation
        if self.should_take_initiative():
            # Generate an initiative observation based on conversation context
            initiative_prompt = self.get_initiative_prompt()
            
            self.conversation_history.append({
                "role": "system",
                "content": initiative_prompt
            })
            
            # Get initiative response
            initiative_response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=50,
                temperature=0.7,
                presence_penalty=0.8,
                frequency_penalty=0.8
            )
            
            initiative_text = initiative_response.choices[0].message.content
            
            # Add a slight pause for more natural flow
            time.sleep(1.5)
            
            # Speak the initiative text with default voice
            speak(initiative_text)
            
            # Add to conversation history
            self.add_message("assistant", initiative_text)
        
        # Only cache standalone (non-followup) questions
        if len(cache_key) < 100 and len(text.split()) < 8 and not is_followup:
            response_cache[cache_key] = ai_response
            
        # Save session data
        self.memory.save_session(self.conversation_history)
            
        return ai_response
    
    def is_followup_question(self, text):
        """Determine if a question is likely a follow-up to previous conversation"""
        # No recent questions means it can't be a follow-up
        if not self.recent_questions:
            return False
        
        # Check for pronouns, references, and short questions
        followup_indicators = [
            "it", "that", "this", "they", "them", "those", "these", 
            "he", "she", "his", "her", "their", "what about", "and",
            "why", "how about", "but", "then", "so", "what else",
            "who", "where", "when"
        ]
        
        text_lower = text.lower()
        
        # Very short questions are almost always follow-ups
        if len(text_lower.split()) <= 6 and self.conversation_history:
            return True
        
        # Check for follow-up indicators
        has_indicator = any(indicator in text_lower.split() for indicator in followup_indicators)
        
        # No question marks but seems like a question
        implicit_question = len(text_lower.split()) < 8 and "?" not in text_lower and text_lower.startswith(("what", "who", "where", "when", "why", "how", "is", "are", "can", "could", "do", "does"))
        
        # Handle questions that don't specify a subject (likely referring to previous subject)
        missing_subject = len(text_lower.split()) < 10 and not any(entity in text_lower for entity in ["president", "person", "company", "country", "city", "state", "movie", "book", "song", "food", "place"])
        
        return has_indicator or implicit_question or missing_subject
    
    def should_search_web(self, text):
        """Determine if a user query needs real-time information from the web."""
        # Check for obvious search queries
        search_indicators = [
            "search for", "look up", "find information", "what is", "who is", 
            "how to", "where is", "when did", "current", "latest", "news about",
            "weather", "price of", "how many", "tell me about", "information on"
        ]
        
        # Check for time-sensitive terms
        time_indicators = [
            "current", "latest", "recent", "now", "today", "tonight", "tomorrow",
            "weather", "forecast", "news", "price", "stock", "event", "score"
        ]
        
        text_lower = text.lower()
        
        # Check if the query contains search indicators
        has_search_term = any(term in text_lower for term in search_indicators)
        
        # Check if the query contains time-sensitive terms
        is_time_sensitive = any(term in text_lower for term in time_indicators)
        
        # If the query is a question, it might need search
        is_question = any(text_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "is", "are", "can", "will"]) and "?" in text
        
        # Don't search if it seems like a personal or follow-up question
        personal_question = any(term in text_lower for term in ["you", "your", "yourself", "we", "our", "earlier", "before", "previous"])
        
        return (has_search_term or is_time_sensitive or is_question) and not personal_question
    
    def save_session(self):
        # Final save of the session data
        self.memory.save_session(self.conversation_history)
        print("Session saved to memory")

    def is_factual_question(self, text):
        """Determine if a question is primarily factual rather than opinion-based"""
        # Look for indicators of factual questions
        factual_indicators = [
            "who", "what", "when", "where", "how many", "which", 
            "why does", "explain", "tell me about", "define",
            "history", "date", "fact", "information", "data"
        ]
        
        # Look for opinion indicators
        opinion_indicators = [
            "think", "feel", "believe", "opinion", "perspective", "view",
            "better", "worse", "favorite", "best", "worst", "like", "prefer",
            "should", "would", "could", "funny", "interesting", "boring"
        ]
        
        text_lower = text.lower()
        
        # Check for factual indicators
        has_factual = any(indicator in text_lower for indicator in factual_indicators)
        
        # Check for opinion indicators
        has_opinion = any(indicator in text_lower for indicator in opinion_indicators)
        
        # If both are present, the specific indicators take precedence
        if has_factual and has_opinion:
            return not any(indicator in text_lower for indicator in ["think", "feel", "believe", "opinion", "favorite"])
        
        # Default to factual if unclear - better to be informative than witty by default
        return has_factual or not has_opinion

    def get_initiative_prompt(self):
        """Create a prompt for initiative-based observations"""
        # Analyze conversation for potential topics of interest
        recent_exchanges = [msg for msg in self.conversation_history[-6:] 
                           if msg["role"] in ["user", "assistant"]]
        recent_text = " ".join([msg["content"] for msg in recent_exchanges])
        
        # Different types of initiative prompts
        initiative_types = [
            "Based on the conversation, offer an unprompted insight about something mentioned earlier. Start with 'I notice that...' or 'By the way...'",
            "Make a brief observation about a pattern in the user's interests or questions. Be subtle and conversational.",
            "Offer a small piece of related information that expands on a topic mentioned earlier. Keep it concise and interesting.",
            "Make a self-aware observation about your own processing or perspective on the conversation.",
            "Mention something you've learned or inferred about the user from your conversation so far."
        ]
        
        # Randomly select initiative type
        selected_prompt = np.random.choice(initiative_types)
        
        return f"Take initiative with an unprompted observation:\n{selected_prompt}\nKeep it VERY brief (1 sentence) and make it feel natural and conversational."

# Function to process audio in background
def process_audio_to_text(audio_data):
    if audio_data:
        try:
            transcript = transcribe_audio(audio_data)
            if not transcript or len(transcript.strip()) < 2:
                return "Sorry, I couldn't understand what you said. Please try speaking clearly and a bit louder."
            return transcript
        except Exception as e:
            error_msg = str(e)
            print(f"Error transcribing audio: {error_msg}")
            
            if "too_short" in error_msg.lower():
                return "The audio was too short. Please speak for at least 1-2 seconds."
            elif "no_speech" in error_msg.lower() or "no speech" in error_msg.lower():
                return "I couldn't detect any speech. Please check your microphone."
            elif "format" in error_msg.lower():
                return "There was an issue with the audio format. Please try again."
            elif "ffmpeg" in error_msg.lower():
                return "Audio conversion issue. Please check if ffmpeg is installed."
            else:
                return "There was a problem understanding your speech. Please try again."
    return None

# Simplify voice style selection
def select_voice_style_for_content(text):
    """Select appropriate voice style based on content type"""
    # Just use the default style for now to ensure stability
    return "default"

# Simplify the get_response_for_text function 
def get_response_for_text(conversation, text):
    """Get AI response and determine appropriate voice style"""
    try:
        # Get the AI response
        ai_response = conversation.get_ai_response(text)
        
        # Return the response and default style for now
        return {"response": ai_response, "style": "default"}
    except Exception as e:
        print(f"Error in get_response_for_text: {e}")
        return {"response": f"Error getting AI response: {str(e)}", "style": "default"}

# Pre-generated responses for common phrases to avoid API calls entirely
voice_cache = {}

def speak(text, voice_style=None, cache_only=False):
    """Generate speech from text using the ElevenLabs API with enhanced natural flow and tone variation"""
    global voice_enabled, eleven_voice_id, current_voice_style, voice_cache
    
    # If voice is disabled, just print the text
    if not voice_enabled:
        print(f"TARS: {text}")
        return
        
    # Use the specified voice style or default to current voice ID
    voice_id = VOICE_STYLES.get(voice_style, eleven_voice_id) if voice_style else eleven_voice_id
    
    # Generate a cache key based on the voice and text
    cache_key = f"{voice_id}:{text}"
    
    # Check if we already have this audio cached
    if cache_key in voice_cache:
        if cache_only:
            return  # Exit if we're just preloading
            
        # Print the text while playing audio for better UX
        print(f"TARS: {text}")
        
        # Play the cached audio
        try:
            play_audio_bytes(voice_cache[cache_key])
            return
        except Exception as e:
            print(f"Error playing cached audio: {e}")
            # Fall through to regenerate the audio
    
    # Analyze tone to determine appropriate voice settings
    tone = analyze_text_tone(text)
    
    # Determine dynamic voice settings based on content analysis
    voice_settings = get_dynamic_voice_settings(text, tone)
    
    # If this is a cache-only call and we don't have it cached, generate it
    if cache_only:
        try:
            audio = elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128",
                voice_settings=voice_settings
            )
            audio_data = b''
            for chunk in audio:
                if chunk:
                    audio_data += chunk
            voice_cache[cache_key] = audio_data
        except Exception as e:
            print(f"Error caching voice response: {e}")
        return
    
    # Otherwise, proceed with normal speech generation
    start_time = time.time()
    
    # Print the text first for immediate feedback
    print(f"TARS: {text}")
    
    try:
        # Pre-process text for more natural pauses and tone variations
        processed_text = add_natural_pauses(text)
        
        # Try to use ElevenLabs API
        audio = elevenlabs_client.text_to_speech.convert(
            text=processed_text,
            voice_id=voice_id,
            model_id="eleven_monolingual_v1",
            output_format="mp3_44100_128",
            voice_settings=voice_settings
        )
        
        audio_data = b''
        for chunk in audio:
            if chunk:
                audio_data += chunk
                
        # Cache the audio data for future use
        voice_cache[cache_key] = audio_data
        
        # Play the audio
        play_audio_bytes(audio_data)
        
        end_time = time.time()
        generation_time = end_time - start_time
        if generation_time > 2:
            print(f"Voice generation took {generation_time:.2f} seconds")
            
    except Exception as e:
        print(f"Error generating voice: {e}")
        print("Falling back to system TTS...")
        
        # Fallback to system TTS
        try:
            generate_system_tts(text)
        except Exception as tts_error:
            print(f"System TTS also failed: {tts_error}")
            # At this point, we've already printed the text, so user can read it

def get_dynamic_voice_settings(text, tone):
    """Generate dynamic voice settings based on content and desired tone"""
    # Base settings with good defaults
    settings = {
        "stability": 0.35,
        "similarity_boost": 0.75,
        "style": 0.15,
        "use_speaker_boost": True
    }
    
    # Adjust settings based on tone
    if tone == "excited":
        settings["stability"] = 0.25  # Less stability for more expressiveness
        settings["style"] = 0.25      # More style variation
    elif tone == "serious":
        settings["stability"] = 0.45  # More stability for serious content
        settings["style"] = 0.05      # Less style variation
    elif tone == "thoughtful":
        settings["stability"] = 0.40  # More stability for thoughtful content
        settings["style"] = 0.10      # Moderate style
    elif tone == "humorous":
        settings["stability"] = 0.30  # Less stability for humor
        settings["style"] = 0.30      # More style for humor
    
    # Add slight random variation to make each response unique
    settings["stability"] += np.random.uniform(-0.05, 0.05)
    settings["similarity_boost"] += np.random.uniform(-0.05, 0.05)
    settings["style"] += np.random.uniform(-0.05, 0.05)
    
    # Ensure values remain in valid ranges
    settings["stability"] = max(0.0, min(1.0, settings["stability"]))
    settings["similarity_boost"] = max(0.0, min(1.0, settings["similarity_boost"]))
    settings["style"] = max(0.0, min(1.0, settings["style"]))
    
    return settings

def add_natural_pauses(text):
    """Add natural pauses, emphasis and tone variations to make speech flow more naturally"""
    # First, analyze the emotional tone of the text
    tone = analyze_text_tone(text)
    
    # Replace periods with slight pause
    text = text.replace(". ", ". <break time='300ms'/> ")
    
    # Add subtle pause after commas
    text = text.replace(", ", ", <break time='150ms'/> ")
    
    # Add subtle pause for question marks and exclamation points
    text = text.replace("? ", "? <break time='350ms'/> ")
    text = text.replace("! ", "! <break time='300ms'/> ")
    
    # Add variation for questions
    if "?" in text:
        text = f"<prosody pitch='+15%' rate='95%'>{text}</prosody>"
    
    # Add emphasis to important words
    for word in ["important", "critical", "significant", "never", "always", "must"]:
        text = re.sub(f"\\b{word}\\b", f"<emphasis level='strong'>{word}</emphasis>", text, flags=re.IGNORECASE)
    
    # Add tone variation based on detected emotion
    if tone == "excited":
        text = f"<prosody pitch='+10%' rate='110%'>{text}</prosody>"
    elif tone == "serious":
        text = f"<prosody pitch='-5%' rate='95%'>{text}</prosody>"
    elif tone == "thoughtful":
        text = f"<prosody pitch='-2%' rate='90%'>{text}</prosody>"
    elif tone == "humorous":
        text = f"<prosody pitch='+7%' rate='105%'>{text}</prosody>"
    
    # Add random pitch variations to some sentences for more natural sound
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        for i in range(len(sentences)):
            # Randomly vary some sentences (30% chance)
            if np.random.random() < 0.3 and not sentences[i].startswith("<prosody"):
                pitch_var = np.random.choice(['-5%', '+5%', '-3%', '+3%', '-7%', '+7%'])
                rate_var = np.random.choice(['95%', '105%', '98%', '102%', '90%', '110%'])
                sentences[i] = f"<prosody pitch='{pitch_var}' rate='{rate_var}'>{sentences[i]}</prosody>"
        
        # Rebuild text with variations
        text = " ".join(sentences)
    
    return text

def analyze_text_tone(text):
    """Analyze the emotional tone of text to apply appropriate speech variations"""
    text_lower = text.lower()
    
    # Check for different emotional indicators
    if any(word in text_lower for word in ["excited", "amazing", "fantastic", "awesome", "wow", "incredible"]):
        return "excited"
    elif any(word in text_lower for word in ["serious", "important", "warning", "caution", "danger"]):
        return "serious"
    elif any(word in text_lower for word in ["think", "consider", "perhaps", "maybe", "possibly"]):
        return "thoughtful"
    elif any(word in text_lower for word in ["funny", "joke", "humor", "laugh", "amusing", "kidding"]):
        return "humorous"
    
    # Default tone
    return "neutral"

# Add a WakeWordDetector class
class WakeWordDetector:
    """Wake word detection for activating TARS with voice"""
    
    def __init__(self, wake_words=["hey tars", "tars", "computer"], sensitivity=0.6, callback=None):
        """
        Initialize the wake word detector
        
        Args:
            wake_words: List of wake words to listen for
            sensitivity: Detection sensitivity (0.0-1.0)
            callback: Function to call when wake word is detected
        """
        self.wake_words = wake_words
        self.sensitivity = sensitivity
        self.callback = callback
        self.is_running = False
        self.porcupine = None
        self.audio = None
        self.wake_word_thread = None
        
        # Check if we have an access key for Porcupine
        self.access_key = os.getenv('PICOVOICE_ACCESS_KEY')
        if not self.access_key:
            print("Warning: No PICOVOICE_ACCESS_KEY found. Wake word detection requires an API key.")
            print("Get a free key from https://console.picovoice.ai/")
            print("Then add PICOVOICE_ACCESS_KEY to your .env file")
            self.is_available = False
            return
            
        try:
            # First check if we have a custom model file
            custom_keyword_paths = []
            hey_tars_path = os.path.join(os.getcwd(), "hey_tars_wasm.ppn")
            tars_path = os.path.join(os.getcwd(), "tars_wasm.ppn")
            
            if os.path.exists(hey_tars_path):
                custom_keyword_paths.append(hey_tars_path)
            
            if os.path.exists(tars_path):
                custom_keyword_paths.append(tars_path)
                
            if custom_keyword_paths:
                try:
                    # Use available custom models
                    sensitivities = [self.sensitivity] * len(custom_keyword_paths)
                    self.porcupine = pvporcupine.create(
                        access_key=self.access_key,
                        keyword_paths=custom_keyword_paths,
                        sensitivities=sensitivities
                    )
                    print(f"Wake word detector initialized with {len(custom_keyword_paths)} custom model(s)!")
                except Exception as model_error:
                    print(f"Error loading custom models: {model_error}")
                    print("Falling back to default wake words.")
                    # Fall back to built-in keywords if custom model fails
                    self.porcupine = pvporcupine.create(
                        access_key=self.access_key,
                        keywords=["jarvis", "computer", "porcupine"],
                        sensitivities=[self.sensitivity, self.sensitivity, self.sensitivity]
                    )
            else:
                # Fall back to built-in keywords if no custom model found
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=["jarvis", "computer", "porcupine"],
                    sensitivities=[self.sensitivity, self.sensitivity, self.sensitivity]
                )
                print(f"Custom wake word models not found. Using default keywords.")
                print(f"For custom wake words, place models in the project directory:")
                print(f"- hey_tars_wasm.ppn: For 'Hey TARS'")
                print(f"- tars_wasm.ppn: For 'TARS'")
            
            self.is_available = True
            
            # Set up PyAudio
            self.audio = pyaudio.PyAudio()
            
        except Exception as e:
            print(f"Error initializing wake word detector: {e}")
            self.is_available = False
    
    def start(self, detected_callback=None):
        """Start listening for wake word in a background thread"""
        if not self.is_available:
            print("Wake word detection not available")
            return False
            
        if self.is_running:
            print("Wake word detector already running")
            return True
        
        # Update callback if provided
        if detected_callback:
            self.callback = detected_callback
            
        self.is_running = True
        self.wake_word_thread = threading.Thread(target=self._listen_for_wake_word)
        self.wake_word_thread.daemon = True
        self.wake_word_thread.start()
        return True
    
    def stop(self):
        """Stop listening for wake word"""
        self.is_running = False
        if self.wake_word_thread:
            self.wake_word_thread.join(timeout=2.0)
            self.wake_word_thread = None
    
    def _listen_for_wake_word(self):
        """Background thread that listens for wake word"""
        # Create an input stream to read audio
        stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
        print("🎤 Listening for wake word...")
        
        try:
            while self.is_running:
                # Read audio frame
                pcm = stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Process with Porcupine
                result = self.porcupine.process(pcm)
                
                # If wake word detected (result >= 0 indicates which wake word)
                if result >= 0:
                    print(f"🎯 Wake word detected! ({result})")
                    # Run the callback function if provided
                    if self.callback:
                        self.callback()
                    # Small pause to avoid re-triggering immediately
                    time.sleep(1.0)
                    
        except Exception as e:
            print(f"Error in wake word detection: {e}")
        finally:
            # Clean up
            if stream:
                stream.close()
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.stop()
        if self.porcupine:
            self.porcupine.delete()

# Add a function to change voice style
def set_voice_style(style_name):
    """Change TARS voice style"""
    global current_voice_style, eleven_voice_id
    
    if style_name in VOICE_STYLES:
        current_voice_style = style_name
        style_info = VOICE_STYLES[style_name]
        eleven_voice_id = style_info
        return f"Voice style changed to {style_info}. {style_info}"
    else:
        available_styles = ", ".join(VOICE_STYLES.keys())
        return f"Voice style '{style_name}' not found. Available styles: {available_styles}"

# Update chat_with_ai function to include voice style commands
def chat_with_ai(user_input):
    """Process user input and get response from AI with improved naturalness"""
    global current_user, eleven_voice_id, current_voice_style, voice_enabled

    # Check for system commands
    if user_input.lower() == "voice off":
        toggle_voice(False)
        return "Voice output disabled"
        
    if user_input.lower() == "voice on":
        toggle_voice(True)
        return "Voice output enabled"
        
    if user_input.lower() == "voice list":
        voice_list = ", ".join(VOICE_STYLES.keys())
        return f"Available voice styles: {voice_list}"
        
    if user_input.lower().startswith("voice "):
        style = user_input.lower().replace("voice ", "").strip()
        if style in VOICE_STYLES:
            current_voice_style = style
            eleven_voice_id = VOICE_STYLES[style]
            return f"Voice style changed to {style}"
        else:
            return f"Unknown voice style: {style}. Available styles: {', '.join(VOICE_STYLES.keys())}"
    
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        global running
        running = False
        return "Shutting down. Goodbye!"
        
    # Handle normal conversation with AI
    ai_response = get_ai_response(user_input, current_user)
    return ai_response

# Add a function to test audio capture and feature extraction
def test_audio_capture(recorder):
    """Test audio capture and feature extraction without enrollment"""
    print("\n### Audio Test Mode ###")
    print("This will test your microphone and audio processing.")
    print("Please press Enter to start recording, then speak normally for 3-5 seconds.")
    input()
    
    # Record audio
    recorder.start_recording()
    audio_data = recorder.get_audio_data()
    
    if not audio_data:
        print("❌ No audio detected. Check your microphone settings.")
        return False
        
    # Test 1: Check if we can transcribe the audio
    print("Testing audio transcription...")
    try:
        audio_data.seek(0)
        transcript = transcribe_audio(audio_data)
        if transcript:
            print(f"✅ Transcription successful: '{transcript}'")
        else:
            print("❌ Transcription failed - no text detected.")
    except Exception as e:
        print(f"❌ Transcription error: {str(e)}")
    
    # Test 2: Check if we can extract features
    print("Testing feature extraction...")
    try:
        audio_data.seek(0)
        # Use librosa to load the audio
        audio_array, sr = sf.read(audio_data)
        features = voice_recognition.extract_features(audio_array)
        
        if features is not None:
            print(f"✅ Feature extraction successful: {len(features)} features extracted")
        else:
            print("❌ Feature extraction failed")
    except Exception as e:
        print(f"❌ Feature extraction error: {str(e)}")
    
    # Test 3: Analyze audio quality
    try:
        # Calculate signal-to-noise ratio and volume level
        abs_audio = np.abs(audio_array)
        mean_volume = np.mean(abs_audio)
        max_volume = np.max(abs_audio)
        
        print(f"Audio quality analysis:")
        if mean_volume < 0.01:
            print("❌ Audio volume is very low. Please speak louder or adjust microphone.")
        elif mean_volume > 0.5:
            print("⚠️ Audio volume is very high. Your microphone might be picking up too much.")
        else:
            print("✅ Audio volume level appears good.")
            
        # Simple silence detection
        silent_threshold = 0.01
        silent_portions = np.mean(abs_audio < silent_threshold)
        if silent_portions > 0.5:
            print("❌ Too much silence detected. Please speak more continuously.")
        else:
            print("✅ Speech continuity looks good.")
    except Exception as e:
        print(f"❌ Audio analysis error: {str(e)}")
    
    print("\nAudio test complete.")
    print("If all tests pass, you should be able to enroll successfully.")
    return True

# Add back the missing web search function
def search_web(query, num_results=3):
    """Search the web for information using a simple HTTP request."""
    print(f"Searching the web for: {query}")
    
    try:
        # Use DuckDuckGo for simple searches without API key requirements
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Format the query for URL
        search_query = query.replace(' ', '+')
        url = f"https://html.duckduckgo.com/html/?q={search_query}"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Search failed with status code: {response.status_code}"
        
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', {'class': 'result__body'})
        
        if not results:
            return "No search results found."
        
        # Extract and format the results
        formatted_results = []
        
        for i, result in enumerate(results[:num_results]):
            title_elem = result.find('a', {'class': 'result__a'})
            snippet_elem = result.find('a', {'class': 'result__snippet'})
            
            title = title_elem.text.strip() if title_elem else "No title"
            snippet = snippet_elem.text.strip() if snippet_elem else "No description"
            
            formatted_results.append(f"Result {i+1}: {title}\n{snippet}\n")
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error during web search: {str(e)}"

# Add back get_realtime_info function for searches
def get_realtime_info(query):
    """Get real-time information based on the query."""
    if re.search(r'weather|temperature|forecast', query.lower()):
        return search_web(f"current weather {query}")
    elif re.search(r'news|latest|recent', query.lower()):
        return search_web(f"latest news {query}")
    elif re.search(r'time|date|day', query.lower()):
        now = datetime.datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        return search_web(query)

# Add back the preload OpenAI function
def preload_openai():
    try:
        # Make a very small request to warm up the connection
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
    except Exception as e:
        print(f"OpenAI preload warning: {e}")

# Start preloading in background
preload_thread = threading.Thread(target=preload_openai)
preload_thread.daemon = True
preload_thread.start()

# Add back the transcribe_audio function
def transcribe_audio(audio_data):
    try:
        # Convert WAV data to mp3 format using ffmpeg in a temp file
        from tempfile import NamedTemporaryFile
        import subprocess
        
        # Create temp files for wav and mp3
        with NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            with NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
                wav_path = wav_file.name
                mp3_path = mp3_file.name
        
        # Write the audio data to the wav file
        audio_data.seek(0)
        with open(wav_path, "wb") as f:
            f.write(audio_data.read())
        
        # Convert wav to mp3 using ffmpeg
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", 
                mp3_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting audio format: {e}")
            print(f"ffmpeg stderr: {e.stderr.decode()}")
            raise Exception("Failed to convert audio to MP3 format")
        
        # Now send the mp3 file to OpenAI
        with open(mp3_path, "rb") as mp3_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("recording.mp3", mp3_file.read()),
                language="en"
            )
        
        # Clean up temp files
        try:
            os.remove(wav_path)
            os.remove(mp3_path)
        except:
            pass  # Ignore cleanup errors
            
        return transcript.text
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise

# Add missing utility functions

def toggle_voice(enable):
    """Enable or disable voice output"""
    global voice_enabled
    voice_enabled = enable
    return True

def generate_system_tts(text):
    """Generate system TTS using macOS 'say' command"""
    try:
        subprocess.run(["say", text], check=True)
        return True
    except Exception as e:
        print(f"System TTS error: {e}")
        return False

def play_audio_bytes(audio_data):
    """Play audio from bytes data"""
    try:
        # Save to a temporary file and play
        with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
            
        # Play using mpg123 if available, otherwise use other methods
        try:
            # Use subprocess with start_new_session on Unix systems
            subprocess.run(["mpg123", "-q", temp_path], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                # Try using mpg321 as alternative
                subprocess.run(["mpg321", "-q", temp_path], check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fall back to simpler method that works on most systems
                import platform
                if platform.system() == "Darwin":
                    # macOS
                    subprocess.run(["afplay", temp_path], check=True)
                elif platform.system() == "Windows":
                    # Windows
                    os.startfile(temp_path)  # This is blocking
                else:
                    # Linux
                    subprocess.run(["xdg-open", temp_path], check=True)
                    
        # Clean up temp file after playing
        os.remove(temp_path)
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def get_ai_response(user_input, current_user=None):
    """Process user input and get response from the AI"""
    global conversation_manager
    
    # Create conversation manager if not already created
    if 'conversation_manager' not in globals():
        global conversation_manager
        conversation_manager = ConversationManager()
    
    # Process the input and get response
    return conversation_manager.get_ai_response(user_input)

def init_ai_services():
    """Initialize all AI services required for TARS to operate"""
    # Most service initialization is already done at module level
    # This function ensures all required services are ready
    
    # Check if OpenAI API key is set
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("WARNING: OPENAI_API_KEY not found in environment variables")
        print("TARS needs an OpenAI API key to function")
        print("Please set this in your .env file")
        return False
    
    # Test OpenAI connection with a simple request
    try:
        print("Testing OpenAI API connection...")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        print("OpenAI API connection successful")
    except Exception as e:
        print(f"ERROR: Could not connect to OpenAI API: {e}")
        print("TARS needs a working OpenAI API connection")
        return False
    
    # Create memory directories if they don't exist
    for directory in [MEMORY_DIR, SESSIONS_DIR, VOICE_PROFILES_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Initialize global conversation manager
    global conversation_manager
    conversation_manager = ConversationManager()
    
    return True

# Add this new function before the main loop
def make_response_conversational(response, user_input):
    """Make AI responses more conversational and human-like"""
    # Don't modify very short responses
    if len(response.split()) < 5:
        return response
        
    # Occasionally add conversational starters based on input type
    starters = []
    if "?" in user_input:  # It's a question
        starters = ["Well, ", "Hmm, ", "Let's see... ", "Actually, ", ""]
    elif any(w in user_input.lower() for w in ["thanks", "thank", "appreciate"]):
        starters = ["No problem. ", "Happy to help. ", "Anytime. ", "Of course. ", ""]
    else:
        starters = ["", "You know, ", "So, ", "Well, ", "Right, "]
    
    starter = np.random.choice(starters, p=[0.5, 0.125, 0.125, 0.125, 0.125])
    
    # Add small speech disfluencies occasionally (15% chance)
    if np.random.random() < 0.15 and not starter:
        disfluencies = ["Um, ", "Ah, ", "Hmm, ", "So, "]
        response = np.random.choice(disfluencies) + response
        
    # Add the starter if we have one and didn't add a disfluency
    if starter and not response.startswith(("Um", "Ah", "Hmm", "So")):
        response = starter + response
        
    return response

# Add the main loop function to the end of the file

def main_loop():
    """Main program loop for TARS"""
    global running, current_user, voice_enabled
    
    # Initialize voice recognition and conversation systems
    recorder = AudioRecorder()
    conversation = ConversationManager()
    
    # Initialize wake word detector if available
    wake_word_detector = WakeWordDetector()
    wake_word_active = False
    
    # Print initialization messages
    if voice_enabled:
        print("\nTARS: Initialized and ready. Humor setting at 75%. Internet access enabled.")
        
        # Preload common responses in background
        def preload_voice_responses():
            for text in ["I'm not sure I understand that.", "Could you clarify?", "Processing your request."]:
                try:
                    speak(text, cache_only=True)
                except Exception as e:
                    print(f"Error preloading voice responses: {e}")
        
        # Start preloading voice responses
        executor.submit(preload_voice_responses)
    else:
        print("\nTARS: Initialized and ready. Humor setting at 75%. Internet access enabled. (Text-only mode)")
    
    # Check for voice profiles
    if voice_recognition.has_users():
        print(f"Voice recognition enabled with {len(voice_recognition.profiles)} user profiles.")
    else:
        print("No voice profiles found. Would you like to test the microphone first? (test/enroll/skip)")
        choice = input().lower()
        if choice.startswith('t'):
            # Run a quick microphone test
            test_audio_capture(recorder)
        elif choice.startswith('e'):
            print("Please enter your name:")
            name = input().strip()
            if name:
                recorder.start_enrollment(name, num_samples=3)
    
    print("\nPress Enter to start recording, or type your message directly.")
    print("Type 'quit' to exit, 'enroll' to add a voice profile, or 'test' to test your microphone.")
    
    if wake_word_detector.is_available:
        print("Wake word detection available - say 'Hey TARS', 'TARS', 'Computer', 'Jarvis', or 'Porcupine' to activate hands-free.")
    
    # Print available commands
    print("Available commands:")
    print("- 'enroll': Add a new voice profile")
    print("- 'who': Check which user is currently recognized")
    print("- 'reset': Reset user recognition for this session")
    print("- 'test': Test your microphone")
    print("- 'wake': Toggle wake word detection")
    print("- 'voice list': Show available voice styles")
    print("- 'voice [style]': Change TARS voice style")
    print("- 'voice on/off': Enable/disable voice output")
    
    # Check for previous conversations
    memory_context = conversation.memory.get_memory_context()
    if "Recent conversations" in memory_context:
        print("TARS: I've loaded our previous conversations.")
    
    # Define wake word callback
    def on_wake_word_detected():
        nonlocal wake_word_active
        
        if not wake_word_active:
            return
            
        # Play a short acknowledgment sound with more natural variation
        print("\nWake word detected! Listening...")
        acknowledgments = ["Yes?", "I'm here.", "How can I help?", "What's up?"]
        speak(np.random.choice(acknowledgments))  # More natural acknowledgments
        
        # Start recording automatically
        recorder.start_recording()
        audio_data = recorder.get_audio_data()
        
        if audio_data:
            # Try to recognize the speaker
            conversation.recognize_user(audio_data)
            current_user = voice_recognition.get_current_user()
            
            # Start transcription
            future = executor.submit(process_audio_to_text, audio_data)
            print("Processing your audio...")
            
            user_text = future.result()
            
            if user_text:
                if current_user:
                    print(f"{current_user} said: {user_text}")
                else:
                    print(f"You said: {user_text}")
                
                # Process the input
                response = chat_with_ai(user_text)
                
                # Speak the response
                speak(response)
            else:
                # More varied error responses
                errors = [
                    "I couldn't understand that. Could you try again?",
                    "Sorry, I didn't catch that. Mind repeating?",
                    "That didn't come through clearly. One more time?"
                ]
                error_msg = np.random.choice(errors)
                print(f"TARS: {error_msg}")
                speak(error_msg)
    
    # Start wake word detection if available
    if wake_word_detector.is_available:
        wake_word_active = True
        wake_word_detector.start(detected_callback=on_wake_word_detected)
    
    # Main loop
    running = True
    try:
        while running:
            user_input = input("\nPress Enter to speak or type your message (or 'quit'): ")
            
            # Check for special commands
            if user_input.lower() == 'quit':
                # Save session before exiting
                conversation.save_session()
                # Stop wake word detection
                if wake_word_detector.is_available:
                    wake_word_detector.stop()
                print("\nTARS: Powering down. Session saved to memory.")
                break
            elif user_input.lower() == 'enroll':
                print("Please enter the name for the new voice profile:")
                name = input().strip()
                if name:
                    recorder.start_enrollment(name, num_samples=3)
                continue
            elif user_input.lower() == 'who':
                current_user = voice_recognition.get_current_user()
                if current_user:
                    print(f"TARS: I currently recognize you as {current_user}.")
                else:
                    print("TARS: I haven't recognized your voice yet in this session.")
                continue
            elif user_input.lower() == 'reset':
                voice_recognition.current_user = None
                conversation.recognized_user = None
                print("TARS: Voice recognition reset for this session.")
                continue
            elif user_input.lower() == 'test':
                test_audio_capture(recorder)
                continue
            elif user_input.lower() == 'wake':
                # Toggle wake word detection
                if wake_word_detector.is_available:
                    wake_word_active = not wake_word_active
                    if wake_word_active:
                        wake_word_detector.start(detected_callback=on_wake_word_detected)
                        print("TARS: Wake word detection enabled.")
                    else:
                        wake_word_detector.stop()
                        print("TARS: Wake word detection disabled.")
                else:
                    print("TARS: Wake word detection is not available.")
                continue
                
            # If user typed something
            if user_input.strip():
                if user_input.lower() in ['quit', 'enroll', 'who', 'reset', 'test', 'wake']:
                    continue
                    
                print(f"You typed: {user_input}")
                
                # Process the input
                response = chat_with_ai(user_input)
                
                # Speak the response
                speak(response)
                continue
                
            # Voice recording for empty input (pressing Enter)
            if not voice_enabled:
                print("Voice input is disabled. Please type your message instead.")
                continue
                
            # Record audio
            recorder.start_recording()
            audio_data = recorder.get_audio_data()
            
            if audio_data:
                # Try to recognize the speaker
                conversation.recognize_user(audio_data)
                current_user = voice_recognition.get_current_user()
                
                # Start transcription immediately
                future = executor.submit(process_audio_to_text, audio_data)
                
                # Show processing indicator
                print("Processing your audio...")
                
                user_text = future.result()
                
                if user_text:
                    if current_user:
                        print(f"{current_user} said: {user_text}")
                    else:
                        print(f"You said: {user_text}")
                    
                    # Process the input
                    response = chat_with_ai(user_text)
                    
                    # Speak the response
                    speak(response)
                else:
                    print("TARS: I couldn't understand that. Please try again.")
            else:
                print("TARS: I'm detecting a distinct lack of audio. Let's try that again.")
    except KeyboardInterrupt:
        # Save session on CTRL+C exit
        conversation.save_session()
        # Stop wake word detection
        if wake_word_detector.is_available:
            wake_word_detector.stop()
        print("\nTARS: Session saved to memory. Shutting down.")

# Update the main section to call the main loop
if __name__ == "__main__":
    # Initialize the required services
    init_ai_services()
    
    # Run the main program loop
    main_loop() 