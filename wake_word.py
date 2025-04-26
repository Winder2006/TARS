import os
import struct
import time
import numpy as np
from pathlib import Path
import threading
from dotenv import load_dotenv
import platform
import sys
import logging
import select  # Import select at the module level for the mock detector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("wake_word")

# Load environment variables from .env file
load_dotenv()

# Try to import pvporcupine but provide fallback for unsupported platforms
PORCUPINE_AVAILABLE = False
try:
    import pvporcupine
    import pyaudio
    PORCUPINE_AVAILABLE = True
except ImportError:
    logger.warning("pvporcupine or pyaudio not installed. Wake word detection will be limited.")
except Exception as e:
    logger.error(f"Error loading pvporcupine: {e}")

# Detect platform - especially for Raspberry Pi
IS_RASPBERRY_PI = False
try:
    with open('/proc/device-tree/model', 'r') as f:
        model = f.read()
        if 'Raspberry Pi' in model:
            IS_RASPBERRY_PI = True
            PI_MODEL = model.strip('\0')
            logger.info(f"Detected Raspberry Pi: {PI_MODEL}")
except:
    # Not a Raspberry Pi or can't determine
    pass

class MockWakeWordDetector:
    """A mock implementation of wake word detection for unsupported platforms"""
    
    def __init__(self, wake_words=None, sensitivity=0.6, callback=None):
        self.wake_words = wake_words or ["hey tars", "tars", "computer"]
        self.sensitivity = sensitivity
        self.callback = callback
        self.is_running = False
        self.wake_word_thread = None
        self.is_available = True
        logger.warning("Using MockWakeWordDetector - limited functionality")
        
    def start(self, detected_callback=None):
        """Start listening for keyboard input as a wake word alternative"""
        if self.is_running:
            logger.info("Mock wake word detector already running")
            return True
            
        # Update callback if provided
        if detected_callback:
            self.callback = detected_callback
            
        self.is_running = True
        self.wake_word_thread = threading.Thread(target=self._mock_listener)
        self.wake_word_thread.daemon = True
        self.wake_word_thread.start()
        return True
        
    def stop(self):
        """Stop listening"""
        self.is_running = False
        if self.wake_word_thread:
            self.wake_word_thread.join(timeout=2.0)
            self.wake_word_thread = None
            
    def _mock_listener(self):
        """Simulate wake word detection with keyboard input"""
        logger.info("🎤 Mock wake word detector active")
        logger.info("Press 'w' + Enter to trigger wake word detection")
        
        try:
            while self.is_running:
                # Simple approach that works cross-platform
                # In a real implementation, you might use platform-specific
                # keyboard monitoring that doesn't block
                sys.stdout.write("\rListening for wake word (press 'w' + Enter): ")
                sys.stdout.flush()
                
                # Use a non-blocking approach with timeout
                if sys.stdin in select.select([sys.stdin], [], [], 1.0)[0]:
                    key = sys.stdin.readline().strip()
                    if key.lower() == 'w':
                        logger.info("🎯 Wake word triggered via keyboard!")
                        if self.callback:
                            self.callback()
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in mock wake word detection: {e}")
            
    def __del__(self):
        """Clean up resources"""
        self.stop()

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
            logger.warning("No PICOVOICE_ACCESS_KEY found. Wake word detection requires an API key.")
            logger.info("Get a free key from https://console.picovoice.ai/")
            logger.info("Then add PICOVOICE_ACCESS_KEY to your .env file")
            self.is_available = False
            return
            
        # Check if pvporcupine is available
        if not PORCUPINE_AVAILABLE:
            logger.warning("pvporcupine not available on this system")
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
                    logger.info(f"Wake word detector initialized with {len(custom_keyword_paths)} custom model(s)!")
                except Exception as model_error:
                    logger.error(f"Error loading custom models: {model_error}")
                    logger.info("Falling back to default wake words.")
                    # Fall back to built-in keywords if custom model fails
                    try:
                        self.porcupine = pvporcupine.create(
                            access_key=self.access_key,
                            keywords=["jarvis", "computer", "porcupine"],
                            sensitivities=[self.sensitivity, self.sensitivity, self.sensitivity]
                        )
                    except NotImplementedError as e:
                        logger.error(f"This platform is not supported by pvporcupine: {e}")
                        self.is_available = False
                        return
            else:
                # Fall back to built-in keywords if no custom model found
                try:
                    self.porcupine = pvporcupine.create(
                        access_key=self.access_key,
                        keywords=["jarvis", "computer", "porcupine"],
                        sensitivities=[self.sensitivity, self.sensitivity, self.sensitivity]
                    )
                    logger.info(f"Custom wake word models not found. Using default keywords.")
                    logger.info(f"For custom wake words, place models in the project directory:")
                    logger.info(f"- hey_tars_wasm.ppn: For 'Hey TARS'")
                    logger.info(f"- tars_wasm.ppn: For 'TARS'")
                except NotImplementedError as e:
                    logger.error(f"This platform is not supported by pvporcupine: {e}")
                    self.is_available = False
                    return
            
            self.is_available = True
            
            # Set up PyAudio
            self.audio = pyaudio.PyAudio()
            
        except Exception as e:
            logger.error(f"Error initializing wake word detector: {e}")
            self.is_available = False
    
    def start(self, detected_callback=None):
        """Start listening for wake word in a background thread"""
        if not self.is_available:
            logger.warning("Wake word detection not available")
            return False
            
        if self.is_running:
            logger.info("Wake word detector already running")
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
        
        logger.info("🎤 Listening for wake word...")
        
        try:
            while self.is_running:
                # Read audio frame
                pcm = stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Process with Porcupine
                result = self.porcupine.process(pcm)
                
                # If wake word detected (result >= 0 indicates which wake word)
                if result >= 0:
                    logger.info(f"🎯 Wake word detected! ({result})")
                    # Run the callback function if provided
                    if self.callback:
                        self.callback()
                    # Small pause to avoid re-triggering immediately
                    time.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
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

def get_wake_word_detector(wake_words=None, sensitivity=0.6, callback=None):
    """Factory function to get the appropriate wake word detector for the platform"""
    try:
        # First try the real detector
        detector = WakeWordDetector(wake_words, sensitivity, callback)
        if detector.is_available:
            logger.info("Using real wake word detector")
            return detector
    except Exception as e:
        logger.error(f"Failed to initialize real wake word detector: {e}")
    
    # Use mock detector as fallback
    logger.warning("Falling back to mock wake word detector")
    return MockWakeWordDetector(wake_words, sensitivity, callback)

# Example usage
if __name__ == "__main__":
    def on_wake_word():
        print("Wake word detected! TARS is listening...")
        # This would trigger the main recording loop in a real integration
    
    # Create and start detector using the factory function
    detector = get_wake_word_detector(callback=on_wake_word)
    
    if detector.is_available:
        detector.start()
        
        # Keep running until user presses ctrl+c
        try:
            while True:
                if hasattr(detector, 'porcupine') and detector.porcupine:
                    available_keywords = detector.porcupine._keyword_paths or detector.porcupine._keywords
                    keyword_info = ", ".join([os.path.basename(k).replace(".ppn", "") if ".ppn" in k else k 
                                           for k in available_keywords]) if available_keywords else "Unknown"
                    print(f"Waiting for wake word... ({keyword_info})")
                else:
                    print("Waiting for wake word...")
                time.sleep(3)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            detector.stop()
    else:
        print("Wake word detection not available. Please check requirements.") 