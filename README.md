# Voice Assistant with OpenAI and ElevenLabs

A voice assistant that uses OpenAI's GPT-4 for conversation and ElevenLabs for text-to-speech.

## Features

- Voice recording and transcription using OpenAI's Whisper
- Conversation with GPT-4
- Text-to-speech with ElevenLabs
- TARS personality (inspired by Interstellar)

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (see `.env.example`)
4. Run the application: `python chat.py`

## Git Usage

This project is version-controlled with Git. Here are some useful commands:

### View changes
```
git status  # See what files have changed
git diff    # See detailed changes
```

### Save changes
```
git add .                    # Stage all changes
git commit -m "Your message" # Save changes with a description
```

### Return to this version
If you want to discard changes and return to this version:
```
git reset --hard HEAD  # WARNING: This will delete all uncommitted changes
```

### Create branches for experiments
```
git branch experiment-name   # Create a new branch
git checkout experiment-name # Switch to the branch
```

You can switch back to the main branch with:
```
git checkout main
```

## File Structure

- `chat.py` - Main application file with TARS personality
- `jarvis_fast.py` - Alternative implementation focused on speed
- `test_openai.py` - Simple test script for OpenAI API
- `.env` - Your API keys (not tracked by Git)
- `.env.example` - Template for API keys
- `requirements.txt` - Required Python packages 