#!/usr/bin/env python3
import os
import logging
import time
import tempfile
import numpy as np
from datetime import datetime

# Try to import necessary audio libraries
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: sounddevice or soundfile not installed. Voice features limited.")
    AUDIO_AVAILABLE = False

# Try to import OpenAI for speech recognition
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    # Initialize OpenAI client if API key exists
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        print("Warning: OPENAI_API_KEY not set. Speech recognition unavailable.")
        OPENAI_AVAILABLE = False
except ImportError:
    print("Warning: OpenAI library not installed. Speech recognition unavailable.")
    OPENAI_AVAILABLE = False
    openai_client = None

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_chat")

class AudioRecorder:
    """Simple audio recorder for voice input"""
    
    def __init__(self):
        """Initialize the audio recorder"""
        self.sample_rate = 44100
        self.channels = 1
        self.dtype = 'float32'
        
    def record_audio(self, seconds=5):
        """Record audio for a specified duration"""
        if not AUDIO_AVAILABLE:
            print("Audio recording not available. Install sounddevice and soundfile.")
            return None
            
        print(f"Recording for {seconds} seconds... (Speak now)")
        try:
            # Record audio using sounddevice
            recording = sd.rec(
                int(seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype
            )
            
            # Wait for recording to complete
            sd.wait()
            print("Recording finished.")
            return recording
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None

def transcribe_audio(audio_data, sample_rate=44100):
    """Transcribe audio data to text using OpenAI Whisper API"""
    if not OPENAI_AVAILABLE or openai_client is None:
        print("OpenAI API not available for transcription.")
        return ""
        
    try:
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)
        
        # Transcribe using OpenAI's Whisper API
        with open(temp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return transcript.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def speak_text(text):
    """Convert text to speech and play it"""
    if not AUDIO_AVAILABLE:
        print(f"[Speaking] {text}")
        return False
        
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except ImportError:
        print(f"[Speaking] {text}")
        print("Text-to-speech not available. Install pyttsx3 for voice output.")
        return False
    except Exception as e:
        print(f"Error with text-to-speech: {e}")
        print(f"[Speaking] {text}")
        return False

class SimpleChat:
    def __init__(self):
        self.conversation = []
        self.logger = logging.getLogger("simple_chat.core")
        self.voice_enabled = AUDIO_AVAILABLE
        self.recorder = AudioRecorder() if AUDIO_AVAILABLE else None
        self.logger.info("Simple Chat initialized")
        self.logger.info(f"Voice enabled: {self.voice_enabled}")
        
    def add_message(self, role, content):
        """Add a message to the conversation history"""
        self.conversation.append({"role": role, "content": content})
        
    def generate_response(self, query):
        """Generate a simple response"""
        start_time = time.time()
        
        # Simple echo response for now
        response = f"You said: {query}"
        
        # Add a timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        response = f"[{timestamp}] {response}"
        
        # Log processing time
        elapsed = time.time() - start_time
        self.logger.info(f"Generated response in {elapsed:.2f} seconds")
        
        return response
        
    def chat(self):
        """Main chat loop"""
        print("Welcome to Simple Chat!")
        print("Type 'exit' to quit")
        print("Type 'voice' to toggle voice mode")
        print("Current voice mode:", "Enabled" if self.voice_enabled else "Disabled")
        
        voice_mode = False  # Start with text input by default
        
        while True:
            if voice_mode and self.voice_enabled:
                print("Listening... (Say something or type 'text' to switch back)")
                
                # Allow user to type instead if desired
                user_input = input("Or type here: ")
                
                if user_input:
                    # User decided to type instead
                    if user_input.lower() == "text":
                        voice_mode = False
                        print("Switched to text input mode")
                        continue
                else:
                    # Record audio and transcribe
                    audio_data = self.recorder.record_audio(5)
                    if audio_data is not None:
                        user_input = transcribe_audio(audio_data)
                        print(f"You (transcribed): {user_input}")
            else:
                # Get user input via text
                user_input = input("You: ")
            
            # Check for voice mode toggle
            if user_input.lower() == "voice":
                if self.voice_enabled:
                    voice_mode = not voice_mode
                    print(f"Voice input mode: {'Enabled' if voice_mode else 'Disabled'}")
                    continue
                else:
                    print("Voice features are not available. Please install required packages.")
                    continue
                    
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
                
            # Add to conversation history
            self.add_message("user", user_input)
            
            # Generate response
            response = self.generate_response(user_input)
            
            # Add to conversation history
            self.add_message("assistant", response)
            
            # Display response
            print(f"Assistant: {response}")
            
            # Speak response if voice is enabled
            if self.voice_enabled:
                speak_text(response)
            
def main():
    """Main entry point"""
    chat = SimpleChat()
    chat.chat()
    
if __name__ == "__main__":
    main() 