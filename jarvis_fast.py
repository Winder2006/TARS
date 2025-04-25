#!/usr/bin/env python3
"""
JARVIS - Fast Voice Assistant for Raspberry Pi
Optimized for minimal latency and real-time interaction
"""

import os
import time
import json
import queue
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import sounddevice as sd
import soundfile as sf
import openai
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
RECORD_SECONDS = 5  # Maximum recording duration

# Initialize APIs
openai.api_key = OPENAI_API_KEY

# Response cache
response_cache: Dict[str, str] = {}

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None

    def callback(self, indata, frames, time, status):
        if status:
            print(f"Audio input status: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.recording = True
        self.audio_queue = queue.Queue()
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()

    def _record(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                          callback=self.callback):
            sd.sleep(int(RECORD_SECONDS * 1000))

    def stop_recording(self) -> np.ndarray:
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        # Combine all audio chunks
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        if audio_data:
            return np.concatenate(audio_data)
        return np.array([])

def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribe audio using OpenAI's Whisper API"""
    start_time = time.time()
    
    # Save audio to temporary file
    temp_file = "temp_recording.wav"
    sf.write(temp_file, audio_data, SAMPLE_RATE)
    
    try:
        with open(temp_file, "rb") as audio_file:
            # Transcribe using Whisper API
            response = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                response_format="text"
            )
        
        print(f"Transcription complete in {time.time() - start_time:.1f}s")
        return response
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""
    finally:
        # Clean up
        os.remove(temp_file)

def get_gpt_response(text: str) -> str:
    """Get response from GPT-3.5-turbo with caching"""
    start_time = time.time()
    
    # Check cache
    if text in response_cache:
        print("Using cached response")
        return response_cache[text]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            max_tokens=150,
            temperature=0.7
        )
        response_text = response.choices[0].message.content
        
        # Cache the response
        response_cache[text] = response_text
        
        print(f"GPT response received in {time.time() - start_time:.1f}s")
        return response_text
    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return "I'm sorry, I couldn't process that request."

def stream_tts(text: str):
    """Stream TTS audio using ElevenLabs"""
    start_time = time.time()
    
    try:
        # Generate audio using ElevenLabs API
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Save to temporary file and play
            temp_file = "temp_response.mp3"
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            # Play audio using appropriate command for the OS
            if os.uname().sysname == "Darwin":
                subprocess.run(["afplay", temp_file])
            else:
                subprocess.run(["aplay", temp_file])
            
            # Clean up
            os.remove(temp_file)
            
            print(f"TTS playback complete in {time.time() - start_time:.1f}s")
        else:
            print(f"Error in TTS API call: {response.status_code}")
            raise Exception(f"API call failed with status {response.status_code}")
            
    except Exception as e:
        print(f"Error in TTS: {e}")
        # Fallback to local TTS if available
        try:
            subprocess.run(["say" if os.uname().sysname == "Darwin" else "espeak", text])
        except:
            print("No fallback TTS available")

def main():
    print("JARVIS Fast Assistant initialized.")
    print("Commands:")
    print("  'r' - Start recording (records for 5 seconds)")
    print("  'q' - Quit")
    
    recorder = AudioRecorder()
    
    while True:
        command = input("\nEnter command (r/q): ").lower()
        
        if command == 'q':
            break
        elif command == 'r':
            print("\nRecording for 5 seconds...")
            recorder.start_recording()
            time.sleep(5)  # Wait for recording to complete
            
            print("Processing...")
            audio_data = recorder.stop_recording()
            
            if len(audio_data) > 0:
                # Transcribe
                text = transcribe_audio(audio_data)
                print(f"You said: {text}")
                
                # Get response
                response = get_gpt_response(text)
                print(f"JARVIS: {response}")
                
                # Play response
                stream_tts(response)
        else:
            print("Invalid command. Use 'r' to record or 'q' to quit.")

if __name__ == "__main__":
    main() 