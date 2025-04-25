import os
import struct
import pvporcupine
import pyaudio
import time
import numpy as np
from pathlib import Path
import threading

class WakeWordDetector:
    """Wake word detection for activating TARS with voice"""
    
    def __init__(self, wake_words=["tars"], sensitivity=0.5, callback=None):
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
            # Initialize Porcupine with available built-in keywords 
            # (Using "porcupine" as closest to "TARS")
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=["porcupine", "computer"],
                sensitivities=[sensitivity, sensitivity]
            )
            
            self.is_available = True
            print(f"Wake word detector initialized with desired phrase: {', '.join(wake_words)}")
            print(f"Actually using: 'porcupine', 'computer' (custom wake words require training)")
            
            # Set up PyAudio
            self.audio = pyaudio.PyAudio()
            
        except Exception as e:
            print(f"Error initializing wake word detector: {e}")
            self.is_available = False
    
    def start(self):
        """Start listening for wake word in a background thread"""
        if not self.is_available:
            print("Wake word detection not available")
            return False
            
        if self.is_running:
            print("Wake word detector already running")
            return True
            
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
        if self.audio:
            self.audio.terminate()

# Example usage
if __name__ == "__main__":
    def on_wake_word():
        print("Wake word detected! TARS is listening...")
        # This would trigger the main recording loop in a real integration
    
    # Create and start detector
    detector = WakeWordDetector(callback=on_wake_word)
    
    if detector.is_available:
        detector.start()
        
        # Keep running until user presses ctrl+c
        try:
            while True:
                print("Waiting for wake word (say 'Porcupine' or 'Computer')...")
                time.sleep(3)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            detector.stop()
    else:
        print("Wake word detection not available. Please check requirements.") 