# Wake Word Detection for TARS Voice Assistant

This documentation covers the wake word detection implementation for TARS Voice Assistant, including compatibility with Raspberry Pi 5 and other platforms.

## Overview

The wake word detection system allows TARS to be activated by voice commands such as "Hey TARS" or "Computer". The implementation uses Picovoice's Porcupine wake word engine when available, with a fallback mechanism for unsupported platforms.

## Platform Compatibility

### Compatible Platforms
- Windows (x86/x64)
- macOS (x86/ARM64)
- Linux (x86/x64)
- Raspberry Pi 3B, 4B (with limitations)

### Limited Compatibility
- Raspberry Pi 5
- Other ARM platforms not officially supported by Picovoice

## How It Works

The system implements two key approaches:

1. **Real Wake Word Detection**: Uses Picovoice's Porcupine engine with the official API for supported platforms.
2. **Mock Wake Word Detection**: A fallback system for unsupported platforms that simulates wake word detection using keyboard input.

### Factory Pattern

The `get_wake_word_detector()` function serves as a factory that automatically selects the appropriate implementation:

```python
from wake_word import get_wake_word_detector

# The function will return either a real or mock detector based on platform compatibility
detector = get_wake_word_detector(callback=my_callback_function)
detector.start()
```

## Raspberry Pi 5 Support

At the time of implementation, Picovoice's Porcupine library doesn't officially support Raspberry Pi 5, resulting in a `NotImplementedError` on this platform. Our solution:

1. First attempts to use the real Porcupine implementation
2. Catches the `NotImplementedError` if thrown
3. Automatically falls back to the mock implementation
4. Provides clear logging about the fallback

## Using the Mock Implementation

When using the mock wake word detector (on unsupported platforms):

1. The system will display a command-line prompt
2. Users can press 'w' + Enter to trigger the wake word detection
3. This simulates the detection process, allowing the assistant to function normally

## Custom Wake Words

To use custom wake words (on supported platforms):

1. Get custom `.ppn` files from Picovoice Console
2. Place them in the project directory as:
   - `hey_tars_wasm.ppn` - For "Hey TARS" wake word
   - `tars_wasm.ppn` - For "TARS" wake word

## Testing

You can test the wake word implementation with:

```bash
python test_wake_word.py
```

This test script will automatically use either the real or mock implementation based on your platform's compatibility.

## Troubleshooting

Common issues:

1. **Missing API Key**: Set the `PICOVOICE_ACCESS_KEY` in your `.env` file
2. **NotImplementedError**: Your platform is not supported by Porcupine, but the system will automatically fall back to the mock implementation
3. **Import Error**: Make sure you've installed all dependencies from `requirements.txt`

## Dependencies

- `pvporcupine`: Picovoice Porcupine Wake Word Engine
- `pyaudio`: For audio capture
- `dotenv`: For loading environment variables
- Standard Python libraries 