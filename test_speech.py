#!/usr/bin/env python3
"""
Test script for speech generation

This script tests the text-to-speech capabilities using the speak function
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from chat.py
try:
    from chat import (
        speak, 
        voice_enabled, 
        VOICE_STYLES, 
        chat_with_ai,
        init_ai_services,
        get_rag_system
    )
except ImportError:
    print("Error: Could not import required functions from chat.py")
    sys.exit(1)

def test_speech():
    """Test speech generation functionality"""
    print("Testing speech functionality...")
    
    # Check if speech is enabled
    global voice_enabled
    if not voice_enabled:
        print("Speech is currently disabled.")
        print("Do you want to enable it for testing? (y/n)")
        choice = input().lower()
        if choice == 'y':
            voice_enabled = True
            print("Speech enabled for testing")
        else:
            print("Speech will remain disabled")
    
    # Initialize the system
    init_ai_services()
    
    # Initialize RAG system
    get_rag_system()
    
    # List available voices
    print("\nAvailable voice styles:")
    for style, voice_id in VOICE_STYLES.items():
        print(f"- {style}: {voice_id}")
    
    # Test basic speech
    print("\nTesting basic speech...")
    result = speak("This is a test of the speech system.")
    print(f"Speech test result: {'Success' if result else 'Failed'}")
    
    # Test interactive mode
    print("\n=== Speech Test Interactive Mode ===")
    print("Type your messages below. Type 'exit' or 'quit' to end the test.")
    print("Voice commands: 'voice on', 'voice off', 'voice list', 'voice [style]'")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Get AI response
        response = chat_with_ai(user_input)
        
        # Print and speak response
        print(f"TARS: {response}")
        speak(response)
    
    print("Speech test completed.")

if __name__ == "__main__":
    test_speech() 