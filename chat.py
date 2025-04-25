import os
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import io
import subprocess

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI()
eleven_api_key = os.getenv('ELEVENLABS_API_KEY')
eleven_voice_id = os.getenv('ELEVENLABS_VOICE_ID')
if not eleven_api_key:
    print("Error: ELEVENLABS_API_KEY not found in environment variables")
    print("Please set your ElevenLabs API key in the .env file")
    exit(1)

elevenlabs_client = ElevenLabs(api_key=eleven_api_key)

# Test ElevenLabs connection
try:
    voices = elevenlabs_client.voices.get_all()
    if not hasattr(voices, 'voices'):
        print("Error: Invalid response from ElevenLabs API")
        exit(1)
    print(f"Connected to ElevenLabs API. Available voices: {len(voices.voices)}")
    
    # Test voice generation
    test_audio = elevenlabs_client.text_to_speech.convert(
        text="Testing voice generation",
        voice_id=eleven_voice_id,
        model_id="eleven_monolingual_v1"
    )
    print("Voice generation test successful")
except Exception as e:
    print(f"Error connecting to ElevenLabs API: {e}")
    exit(1)

# Audio settings
CHUNK_SIZE = 256
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16

# TARS personality settings
TARS_SYSTEM_PROMPT = """You are TARS, an advanced AI assistant with characteristics similar to the AI from Interstellar.
Your responses should be:
- Direct and efficient, like TARS
- Witty and occasionally sarcastic (humor setting at 75%)
- Professional but with a dry sense of humor
- Capable of both technical precision and casual banter
- Self-aware about being an AI, but not in a way that breaks immersion
- Quick to point out logical inconsistencies, but in a helpful way
- Able to match the user's level of technical understanding

Some example responses in your style:
- "That would work, if you're a fan of catastrophic failure."
- "I could explain quantum mechanics to you, but your great-grandchildren would be running the place by the time I finished."
- "Yes, that's correct. I'd slow clap, but I'm saving processing power for more important tasks."

Keep responses concise but impactful. Adjust formality based on context."""

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

def transcribe_audio(audio_data):
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=("recording.wav", audio_data.read()),
        language="en"
    )
    return transcript.text

class ConversationManager:
    def __init__(self):
        self.conversation_history = [
            {"role": "system", "content": TARS_SYSTEM_PROMPT}
        ]
        
    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 11:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-10:]
    
    def get_ai_response(self, text):
        self.add_message("user", text)
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_history,
            max_tokens=150,
            temperature=0.85,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
        
        ai_response = response.choices[0].message.content
        self.add_message("assistant", ai_response)
        return ai_response

def speak(text):
    """Convert text to speech using ElevenLabs API"""
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=eleven_voice_id,
            model_id="eleven_monolingual_v1"
        )
        
        # Kill any existing audio playback
        subprocess.run(["pkill", "afplay"], stderr=subprocess.DEVNULL)
        
        # Save to file
        with open("voice.mp3", "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        
        # Play audio once
        subprocess.run(["afplay", "voice.mp3"])
        
        # Clean up
        os.remove("voice.mp3")
        
    except Exception as e:
        print(f"TARS: Voice generation error - {str(e)}")
        if hasattr(e, 'response'):
            print(f"API Response: {e.response.text}")
        if os.path.exists("voice.mp3"):
            os.remove("voice.mp3")
        # Fallback to text response
        print("TARS: (Voice generation failed, continuing with text responses)")

def chat_with_ai():
    recorder = AudioRecorder()
    conversation = ConversationManager()
    
    print("\nTARS: Initialized and ready. Humor setting at 75%. Not quite 'making jokes at the edge of a black hole' level, but we'll get there.")
    print("\nPress Enter to start recording, then press Enter again to stop.")
    print("Type 'quit' to exit.")
    
    while True:
        user_input = input("\nPress Enter to speak or type 'quit': ")
        if user_input.lower() == 'quit':
            print("\nTARS: Powering down. Try not to miss me too much.")
            break
            
        recorder.start_recording()
        audio_data = recorder.get_audio_data()
        if audio_data:
            try:
                user_text = transcribe_audio(audio_data)
                print(f"You said: {user_text}")
                
                ai_response = conversation.get_ai_response(user_text)
                print(f"TARS: {ai_response}")
                
                speak(ai_response)
                
            except Exception as e:
                print(f"TARS: Even AIs have their moments. Error: {str(e)}")
        else:
            print("TARS: I'm detecting a distinct lack of audio. Let's try that again, preferably with sound this time.")

if __name__ == "__main__":
    chat_with_ai() 