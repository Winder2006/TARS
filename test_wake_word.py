#!/usr/bin/env python3
"""
Test script for wake word detection with Raspberry Pi 5 compatibility
"""

import os
import time
import logging
import sys
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_wake_word")

# Load environment variables
load_dotenv()

def main():
    """Test wake word detection with platform compatibility"""
    print("\n=== TARS Wake Word Detector Test ===\n")
    
    try:
        # Import our factory function
        from wake_word import get_wake_word_detector
        
        # Define a callback function for wake word detection
        def on_wake_word():
            logger.info("Wake word detected!")
            print("\n🎤 Wake word detected! TARS would start listening now...\n")
        
        # Create wake word detector with the factory function
        # This will use the real detector if available, or the mock if not
        print("Initializing wake word detector...")
        detector = get_wake_word_detector(callback=on_wake_word)
        
        if not detector:
            print("Failed to create wake word detector.")
            return
        
        # Start the detector
        print("Starting wake word detection...")
        success = detector.start()
        
        if not success:
            print("Failed to start wake word detector.")
            return
        
        # Run until user exits
        print("\nWake word detector is running.")
        if isinstance(detector.__class__.__name__, str) and "Mock" in detector.__class__.__name__:
            print("Using MOCK wake word detector (fallback mode).")
            print("Press 'w' + Enter to simulate wake word detection.")
        else:
            print("Using real wake word detector.")
            print("Say one of the wake words (e.g., 'Hey TARS').")
        
        print("\nPress Ctrl+C to exit.")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            # Clean up
            detector.stop()
            print("Wake word detector stopped.")
    
    except ImportError:
        print("Error: Wake word module not found.")
        print("Please ensure 'wake_word.py' is in the current directory.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 