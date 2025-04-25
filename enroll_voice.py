import os
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from pathlib import Path
import io
import json
import datetime
import joblib

# Create data directories if they don't exist
MEMORY_DIR = Path("memory")
VOICE_PROFILES_DIR = MEMORY_DIR / "voice_profiles"
MEMORY_DIR.mkdir(exist_ok=True)
VOICE_PROFILES_DIR.mkdir(exist_ok=True)

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16
CHUNK_SIZE = 2048

class VoiceRecognition:
    def __init__(self):
        self.model_path = VOICE_PROFILES_DIR / "voice_model.pkl"
        self.features_path = VOICE_PROFILES_DIR / "voice_features.pkl"
        self.profiles_path = VOICE_PROFILES_DIR / "profiles.json"
        self.model = None
        self.scaler = None
        self.profiles = self._load_profiles()
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
    
    def list_users(self):
        """List all enrolled users"""
        if not self.profiles:
            print("No users enrolled yet")
            return
            
        print("\nEnrolled users:")
        for user_id, profile in self.profiles.items():
            print(f"  - {profile['name']} (enrolled on {profile['created_at']})")
        print("")

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
        
    try:
        audio_data.seek(0)
        # Use librosa to load the audio
        audio_array, sr = sf.read(audio_data)
        
        # Calculate signal-to-noise ratio and volume level
        abs_audio = np.abs(audio_array)
        mean_volume = np.mean(abs_audio)
        max_volume = np.max(abs_audio)
        
        print(f"Audio quality analysis:")
        if mean_volume < 0.01:
            print("❌ Audio volume is very low. Please speak louder or adjust microphone.")
            return False
        elif mean_volume > 0.5:
            print("⚠️ Audio volume is very high. Your microphone might be picking up too much.")
        else:
            print("✅ Audio volume level appears good.")
            
        # Simple silence detection
        silent_threshold = 0.01
        silent_portions = np.mean(abs_audio < silent_threshold)
        if silent_portions > 0.5:
            print("❌ Too much silence detected. Please speak more continuously.")
            return False
        else:
            print("✅ Speech continuity looks good.")
            
        # Test feature extraction
        features = voice_recognition.extract_features(audio_array)
        if features is not None:
            print(f"✅ Feature extraction successful: {len(features)} features extracted")
            return True
        else:
            print("❌ Feature extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Audio analysis error: {str(e)}")
        return False

def main():
    print("=== TARS Voice Enrollment Tool ===")
    voice_recognition = VoiceRecognition()
    recorder = AudioRecorder()
    
    # Show current users
    voice_recognition.list_users()
    
    while True:
        print("\nOptions:")
        print("1. Enroll a new user")
        print("2. Test microphone")
        print("3. List enrolled users")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            print("Please enter the name for the new voice profile:")
            name = input().strip()
            if name:
                # Test audio first
                if test_audio_capture(recorder):
                    recorder.start_enrollment(name, num_samples=3)
                else:
                    print("Audio test failed. Please fix your microphone setup before enrolling.")
        
        elif choice == '2':
            test_audio_capture(recorder)
        
        elif choice == '3':
            voice_recognition.list_users()
        
        elif choice == '4':
            print("Exiting voice enrollment tool.")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main() 