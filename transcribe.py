#!/usr/bin/env python3
import os
import logging
import numpy as np
import soundfile as sf
import openai
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tars.transcribe")

def transcribe_audio(audio_data, sample_rate=44100):
    """
    Transcribe audio data to text using OpenAI's Whisper model.
    
    Args:
        audio_data (numpy.ndarray): Audio data as a numpy array
        sample_rate (int): Sample rate of the audio
        
    Returns:
        str: Transcribed text
    """
    try:
        logger.info("Transcribing audio...")
        
        # Check if audio data is valid
        if audio_data is None or len(audio_data) == 0:
            logger.error("No audio data to transcribe")
            return ""
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_path = temp_audio.name
            try:
                # Save the audio data to the temp file
                sf.write(temp_path, audio_data, sample_rate)
                
                # Get the OpenAI client
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Open the audio file
                with open(temp_path, "rb") as audio_file:
                    # Transcribe with Whisper
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                
                # Get the transcribed text
                transcription = response.text
                logger.info(f"Transcription successful: {transcription}")
                return transcription
            
            except Exception as e:
                logger.error(f"Error transcribing audio: {str(e)}")
                return ""
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        return ""

def transcribe_audio_file(file_path):
    """
    Transcribe an audio file to text.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    try:
        logger.info(f"Transcribing audio file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return ""
        
        # Get the OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Open the audio file
        with open(file_path, "rb") as audio_file:
            # Transcribe with Whisper
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Get the transcribed text
        transcription = response.text
        logger.info(f"Transcription successful: {transcription}")
        return transcription
    
    except Exception as e:
        logger.error(f"Error transcribing audio file: {str(e)}")
        return ""

if __name__ == "__main__":
    # Simple test
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        result = transcribe_audio_file(test_file)
        print(f"Transcription: {result}")
    else:
        print(f"Test file {test_file} not found") 