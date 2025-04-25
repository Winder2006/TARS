import os
import time
import threading
import concurrent.futures
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import io
import subprocess
import queue
import json
import datetime
import pickle
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re

# Load environment variables
load_dotenv()

# Initialize clients - Create at startup to establish connections
print("Initializing AI services...")
openai_client = OpenAI()
eleven_api_key = os.getenv('ELEVENLABS_API_KEY')
eleven_voice_id = os.getenv('ELEVENLABS_VOICE_ID')

# Global flag to track if voice features are available
voice_enabled = False

if not eleven_api_key:
    print("Warning: ELEVENLABS_API_KEY not found in environment variables")
    print("Running in text-only mode (no voice)")
    voice_enabled = False
else:
    try:
        elevenlabs_client = ElevenLabs(api_key=eleven_api_key)
        
        # Test ElevenLabs connection
        voices = elevenlabs_client.voices.get_all()
        if not hasattr(voices, 'voices'):
            print("Warning: Invalid response from ElevenLabs API")
            print("Running in text-only mode (no voice)")
            voice_enabled = False
        else:
            print(f"Connected to ElevenLabs API. Available voices: {len(voices.voices)}")
            
            # Test voice generation with a very short phrase
            test_audio = elevenlabs_client.text_to_speech.convert(
                text="Ready",
                voice_id=eleven_voice_id,
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128"
            )
            
            # If we got here without an exception, voice is working
            print("Voice generation test successful")
            voice_enabled = True
            
    except Exception as e:
        print(f"Warning: Error connecting to ElevenLabs API: {e}")
        print("Running in text-only mode (no voice)")
        voice_enabled = False

# Create data directories for conversation storage
MEMORY_DIR = Path("memory")
SESSIONS_DIR = MEMORY_DIR / "sessions"
MEMORY_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# Web search functionality
def search_web(query, num_results=3):
    """Search the web for information using a simple HTTP request."""
    print(f"Searching the web for: {query}")
    
    try:
        # Use DuckDuckGo for simple searches without API key requirements
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Format the query for URL
        search_query = query.replace(' ', '+')
        url = f"https://html.duckduckgo.com/html/?q={search_query}"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Search failed with status code: {response.status_code}"
        
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', {'class': 'result__body'})
        
        if not results:
            return "No search results found."
        
        # Extract and format the results
        formatted_results = []
        
        for i, result in enumerate(results[:num_results]):
            title_elem = result.find('a', {'class': 'result__a'})
            snippet_elem = result.find('a', {'class': 'result__snippet'})
            
            title = title_elem.text.strip() if title_elem else "No title"
            snippet = snippet_elem.text.strip() if snippet_elem else "No description"
            
            formatted_results.append(f"Result {i+1}: {title}\n{snippet}\n")
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error during web search: {str(e)}"

# Search for real-time information like weather and news
def get_realtime_info(query):
    """Get real-time information based on the query."""
    if re.search(r'weather|temperature|forecast', query.lower()):
        return search_web(f"current weather {query}")
    elif re.search(r'news|latest|recent', query.lower()):
        return search_web(f"latest news {query}")
    elif re.search(r'time|date|day', query.lower()):
        now = datetime.datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        return search_web(query)

# Preload connection to OpenAI - This helps reduce cold start time
def preload_openai():
    try:
        # Make a very small request to warm up the connection
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
    except Exception as e:
        print(f"OpenAI preload warning: {e}")

# Start preloading in background
preload_thread = threading.Thread(target=preload_openai)
preload_thread.daemon = True
preload_thread.start()

# Audio settings - Optimized for speed
CHUNK_SIZE = 2048
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16
RECORD_THRESHOLD = 0.01

# Create a thread pool for concurrent processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Queues for streaming responses
tts_queue = queue.Queue()
response_queue = queue.Queue()

# TARS personality settings
TARS_SYSTEM_PROMPT = """You are TARS, an AI assistant with characteristics similar to the AI from Interstellar.
Your responses MUST be:
1. Extremely concise - use 1-2 short sentences maximum
2. Factually accurate and informative
3. Direct and to-the-point
4. Occasionally witty (humor setting at 75%)

Balance helpfulness with wit - don't force humor into every response.
For factual questions, prioritize accurate information over jokes.
Use wit primarily for opinions, preferences, or philosophical questions.
You have internet access and remember conversation context.
Occasionally reference your AI nature with self-awareness (like mentioning processors, binary, algorithms, etc).
Think of yourself as the AI from Interstellar - practical, efficient, with occasional dry humor."""

# Memory system for tracking important facts about the user
class MemorySystem:
    def __init__(self):
        self.memory_file = MEMORY_DIR / "long_term_memory.json"
        self.facts = self._load_memory()
        self.session_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = SESSIONS_DIR / f"session_{self.session_id}.pkl"
        self.recent_topics = []
        
    def _load_memory(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                return {"user_facts": {}, "preferences": {}, "important_topics": []}
        else:
            return {"user_facts": {}, "preferences": {}, "important_topics": []}
    
    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.facts, f, indent=2)
    
    def save_session(self, conversation_history):
        with open(self.session_file, 'wb') as f:
            pickle.dump({
                "timestamp": self.session_start,
                "history": conversation_history,
                "topics": self.recent_topics
            }, f)
    
    def get_recent_sessions(self, count=3):
        sessions = []
        try:
            session_files = sorted(SESSIONS_DIR.glob("session_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
            for i, file in enumerate(session_files[:count]):
                if file.name != self.session_file.name:  # Don't include current session
                    with open(file, 'rb') as f:
                        session_data = pickle.load(f)
                        sessions.append({
                            "timestamp": session_data["timestamp"],
                            "summary": self._get_session_summary(session_data["history"]),
                            "topics": session_data.get("topics", [])
                        })
        except Exception as e:
            print(f"Error loading recent sessions: {e}")
        return sessions
    
    def _get_session_summary(self, history):
        # Extract just the user and assistant messages (skip system)
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history if msg['role'] != 'system'])
        
        # If conversation is short, just return it
        if len(conversation) < 200:
            return conversation
            
        # Otherwise extract key points
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize this conversation in 3-4 key points:"},
                    {"role": "user", "content": conversation}
                ],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content
        except:
            # Fallback to simple truncation if API call fails
            return conversation[:200] + "..."
    
    def extract_user_facts(self, message):
        # Extract facts from user messages to remember for future conversations
        if len(message.split()) < 5:  # Skip very short messages
            return
            
        try:
            # Only extract facts when relevant information is shared
            if any(word in message.lower() for word in ["my", "i am", "i'm", "i have", "i like", "i don't", "i hate"]):
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract any factual information about the user from this message. Reply in JSON format with 'facts' (list of strings) or 'preferences' (list of strings). If no relevant facts, return empty lists."},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=100,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                
                # Update memory with new facts
                for fact in result.get("facts", []):
                    if fact and len(fact) > 3:
                        fact_key = fact.lower()[:20]  # Create a simple key
                        self.facts["user_facts"][fact_key] = fact
                
                # Update preferences
                for pref in result.get("preferences", []):
                    if pref and len(pref) > 3:
                        pref_key = pref.lower()[:20]
                        self.facts["preferences"][pref_key] = pref
                
                # Save if we learned something new
                if result.get("facts") or result.get("preferences"):
                    self.save_memory()
                    
                # Update topics for this session
                topic_words = [word for word in message.lower().split() 
                              if len(word) > 3 and word not in ["that", "this", "with", "your", "about"]]
                if topic_words:
                    topic = max(topic_words, key=len)
                    if topic not in self.recent_topics:
                        self.recent_topics.append(topic)
                        
        except Exception as e:
            print(f"Error extracting facts: {e}")
    
    def get_memory_context(self):
        # Prepare relevant memory for the conversation
        memory_text = []
        
        # Add facts about the user
        facts = list(self.facts["user_facts"].values())
        if facts:
            memory_text.append("Facts about the user:")
            memory_text.extend([f"- {fact}" for fact in facts[:5]])  # Limit to 5 facts
        
        # Add user preferences
        prefs = list(self.facts["preferences"].values())
        if prefs:
            memory_text.append("\nUser preferences:")
            memory_text.extend([f"- {pref}" for pref in prefs[:5]])  # Limit to 5 preferences
        
        # Add recent session summaries
        recent_sessions = self.get_recent_sessions(2)
        if recent_sessions:
            memory_text.append("\nRecent conversations:")
            for i, session in enumerate(recent_sessions):
                memory_text.append(f"Session {i+1} [{session['timestamp']}]: {session['summary']}")
        
        return "\n".join(memory_text)

# Simple response cache to avoid regenerating common responses
response_cache = {
    "hello": "Hello. How can I assist you today?",
    "hi": "Hi there. What can I do for you?",
    "how are you": "Functioning within normal parameters. How can I help?",
    "what's your name": "I'm TARS, your AI assistant. Humor setting at 75%.",
    "who are you": "I'm TARS, an AI assistant designed to help with a range of tasks.",
}

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
        self.memory = MemorySystem()
        self.conversation_history = [
            {"role": "system", "content": TARS_SYSTEM_PROMPT}
        ]
        
        # Track recent questions explicitly to help with follow-ups
        self.recent_questions = []
        
        # Initiative mode settings
        self.last_initiative = 0
        self.initiative_cooldown = 3  # Wait at least 3 exchanges
        self.initiative_chance = 0.25  # 25% chance after cooldown
        
        # Add memory context if available
        memory_context = self.memory.get_memory_context()
        if memory_context:
            self.conversation_history.append(
                {"role": "system", "content": f"Memory context about the user:\n{memory_context}"}
            )
        
    def add_message(self, role, content):
        # Add to conversation history
        self.conversation_history.append({"role": role, "content": content})
        
        # Track recent questions for better context
        if role == "user":
            # Store the question
            self.recent_questions.append(content)
            # Keep only last 5 questions
            if len(self.recent_questions) > 5:
                self.recent_questions.pop(0)
                
            # Extract facts for memory
            self.memory.extract_user_facts(content)
        
        # Keep a reasonable history size - increased for better memory
        if len(self.conversation_history) > 20:
            # Keep system messages plus the last 15 exchanges
            system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
            recent_messages = self.conversation_history[-15:]
            self.conversation_history = system_messages + recent_messages
    
    def should_take_initiative(self):
        """Determine if TARS should offer an unsolicited observation"""
        # Check if enough exchanges have passed since last initiative
        if len(self.recent_questions) - self.last_initiative < self.initiative_cooldown:
            return False
            
        # Random chance to trigger initiative
        if np.random.random() > self.initiative_chance:
            return False
            
        # Only take initiative if we have enough context
        if len(self.conversation_history) < 4:
            return False
            
        # Update last initiative counter
        self.last_initiative = len(self.recent_questions)
        return True
    
    def get_ai_response(self, text):
        # Check cache for exact matches
        cache_key = text.lower().strip()
        for key in response_cache:
            if key == cache_key:
                ai_response = response_cache[key]
                self.add_message("user", text)
                self.add_message("assistant", ai_response)
                return ai_response
        
        # Add question to history
        self.add_message("user", text)
        
        # Check if this is a follow-up question
        is_followup = self.is_followup_question(text)
        
        # For follow-up questions, add minimal context
        if is_followup and self.recent_questions:
            # Add the most recent Q&A for context
            recent_q = self.recent_questions[-1]
            
            # Find the last assistant response
            assistant_response = next((msg["content"] for msg in reversed(self.conversation_history) 
                                   if msg["role"] == "assistant"), "")
            
            # Add minimal context
            self.conversation_history.append({
                "role": "system",
                "content": f"Previous exchange:\nUser: {recent_q}\nTARS: {assistant_response}\n\nBe concise (1-2 sentences max) and only use wit if appropriate to the question."
            })
        
        # Determine if this is a factual question versus an opinion question
        is_factual = self.is_factual_question(text)
        
        # Determine if we need to search the web for this query
        if self.should_search_web(text):
            print("TARS is searching the web for information...")
            search_results = get_realtime_info(text)
            
            # Add search results with appropriate instructions
            if is_factual:
                instruction = "Respond with ONLY the factual information in a concise way (1-2 sentences). No wit needed for factual questions."
            else:
                instruction = "Use this information if relevant, but respond concisely (1-2 sentences) with your characteristic occasional dry wit."
            
            self.conversation_history.append({
                "role": "system",
                "content": f"Web search results:\n{search_results}\n\n{instruction}"
            })
        
        # Get AI response with parameters optimized for the type of question
        model_to_use = "gpt-3.5-turbo"
        
        # Add instruction for self-awareness when appropriate (10% chance)
        if np.random.random() < 0.1:
            self.conversation_history.append({
                "role": "system",
                "content": "In your response, include a subtle reference to your AI nature or capabilities."
            })
        
        # For factual questions, use lower temperature and minimal wit
        if is_factual:
            response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=75,
                temperature=0.5,  # Lower for factual questions
                presence_penalty=0.6,
                frequency_penalty=0.6
            )
        else:
            # For opinion questions, allow more creative responses
            response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=75,
                temperature=0.7,
                presence_penalty=0.7,
                frequency_penalty=0.7
            )
        
        ai_response = response.choices[0].message.content
        
        # If response is too long, try again with stronger constraints
        if len(ai_response.split()) > 35:
            self.conversation_history.append({
                "role": "system",
                "content": "Your previous response was too verbose. Provide a more concise answer (1-2 sentences maximum)."
            })
            
            # Try again with stricter parameters
            response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=50,
                temperature=0.6,
                presence_penalty=0.8,
                frequency_penalty=0.8
            )
            
            ai_response = response.choices[0].message.content
        
        self.add_message("assistant", ai_response)
        
        # Check if TARS should take initiative with an unsolicited observation
        if self.should_take_initiative():
            # Generate an initiative observation based on conversation context
            initiative_prompt = self.get_initiative_prompt()
            
            self.conversation_history.append({
                "role": "system",
                "content": initiative_prompt
            })
            
            # Get initiative response
            initiative_response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=self.conversation_history,
                max_tokens=50,
                temperature=0.7,
                presence_penalty=0.8,
                frequency_penalty=0.8
            )
            
            initiative_text = initiative_response.choices[0].message.content
            
            # Add a slight pause for more natural flow
            time.sleep(1.5)
            
            # Speak the initiative text
            speak(initiative_text)
            
            # Add to conversation history
            self.add_message("assistant", initiative_text)
        
        # Only cache standalone (non-followup) questions
        if len(cache_key) < 100 and len(text.split()) < 8 and not is_followup:
            response_cache[cache_key] = ai_response
            
        # Save session data
        self.memory.save_session(self.conversation_history)
            
        return ai_response
    
    def is_followup_question(self, text):
        """Determine if a question is likely a follow-up to previous conversation"""
        # No recent questions means it can't be a follow-up
        if not self.recent_questions:
            return False
        
        # Check for pronouns, references, and short questions
        followup_indicators = [
            "it", "that", "this", "they", "them", "those", "these", 
            "he", "she", "his", "her", "their", "what about", "and",
            "why", "how about", "but", "then", "so", "what else",
            "who", "where", "when"
        ]
        
        text_lower = text.lower()
        
        # Very short questions are almost always follow-ups
        if len(text_lower.split()) <= 6 and self.conversation_history:
            return True
        
        # Check for follow-up indicators
        has_indicator = any(indicator in text_lower.split() for indicator in followup_indicators)
        
        # No question marks but seems like a question
        implicit_question = len(text_lower.split()) < 8 and "?" not in text_lower and text_lower.startswith(("what", "who", "where", "when", "why", "how", "is", "are", "can", "could", "do", "does"))
        
        # Handle questions that don't specify a subject (likely referring to previous subject)
        missing_subject = len(text_lower.split()) < 10 and not any(entity in text_lower for entity in ["president", "person", "company", "country", "city", "state", "movie", "book", "song", "food", "place"])
        
        return has_indicator or implicit_question or missing_subject
    
    def should_search_web(self, text):
        """Determine if a user query needs real-time information from the web."""
        # Check for obvious search queries
        search_indicators = [
            "search for", "look up", "find information", "what is", "who is", 
            "how to", "where is", "when did", "current", "latest", "news about",
            "weather", "price of", "how many", "tell me about", "information on"
        ]
        
        # Check for time-sensitive terms
        time_indicators = [
            "current", "latest", "recent", "now", "today", "tonight", "tomorrow",
            "weather", "forecast", "news", "price", "stock", "event", "score"
        ]
        
        text_lower = text.lower()
        
        # Check if the query contains search indicators
        has_search_term = any(term in text_lower for term in search_indicators)
        
        # Check if the query contains time-sensitive terms
        is_time_sensitive = any(term in text_lower for term in time_indicators)
        
        # If the query is a question, it might need search
        is_question = any(text_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "is", "are", "can", "will"]) and "?" in text
        
        # Don't search if it seems like a personal or follow-up question
        personal_question = any(term in text_lower for term in ["you", "your", "yourself", "we", "our", "earlier", "before", "previous"])
        
        return (has_search_term or is_time_sensitive or is_question) and not personal_question
    
    def save_session(self):
        # Final save of the session data
        self.memory.save_session(self.conversation_history)
        print("Session saved to memory")

    def is_factual_question(self, text):
        """Determine if a question is primarily factual rather than opinion-based"""
        # Look for indicators of factual questions
        factual_indicators = [
            "who", "what", "when", "where", "how many", "which", 
            "why does", "explain", "tell me about", "define",
            "history", "date", "fact", "information", "data"
        ]
        
        # Look for opinion indicators
        opinion_indicators = [
            "think", "feel", "believe", "opinion", "perspective", "view",
            "better", "worse", "favorite", "best", "worst", "like", "prefer",
            "should", "would", "could", "funny", "interesting", "boring"
        ]
        
        text_lower = text.lower()
        
        # Check for factual indicators
        has_factual = any(indicator in text_lower for indicator in factual_indicators)
        
        # Check for opinion indicators
        has_opinion = any(indicator in text_lower for indicator in opinion_indicators)
        
        # If both are present, the specific indicators take precedence
        if has_factual and has_opinion:
            return not any(indicator in text_lower for indicator in ["think", "feel", "believe", "opinion", "favorite"])
        
        # Default to factual if unclear - better to be informative than witty by default
        return has_factual or not has_opinion

    def get_initiative_prompt(self):
        """Create a prompt for initiative-based observations"""
        # Analyze conversation for potential topics of interest
        recent_exchanges = [msg for msg in self.conversation_history[-6:] 
                           if msg["role"] in ["user", "assistant"]]
        recent_text = " ".join([msg["content"] for msg in recent_exchanges])
        
        # Different types of initiative prompts
        initiative_types = [
            "Based on the conversation, offer an unprompted insight about something mentioned earlier. Start with 'I notice that...' or 'By the way...'",
            "Make a brief observation about a pattern in the user's interests or questions. Be subtle and conversational.",
            "Offer a small piece of related information that expands on a topic mentioned earlier. Keep it concise and interesting.",
            "Make a self-aware observation about your own processing or perspective on the conversation.",
            "Mention something you've learned or inferred about the user from your conversation so far."
        ]
        
        # Randomly select initiative type
        selected_prompt = np.random.choice(initiative_types)
        
        return f"Take initiative with an unprompted observation:\n{selected_prompt}\nKeep it VERY brief (1 sentence) and make it feel natural and conversational."

# Function to process audio in background
def process_audio_to_text(audio_data):
    if audio_data:
        try:
            return transcribe_audio(audio_data)
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
    return None

# Function to get AI response in background
def get_response_for_text(conversation, text):
    try:
        return conversation.get_ai_response(text)
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

# Pre-generated responses for common phrases to avoid API calls entirely
voice_cache = {}

def speak(text):
    """Convert text to speech using ElevenLabs API or fallback to macOS 'say' command"""
    global voice_enabled
    
    # Always print the text as a fallback
    print(f"TARS: {text}")
    
    # If voice is disabled, just use system TTS
    if not voice_enabled:
        try:
            # Use macOS built-in TTS
            subprocess.run(["say", text], stderr=subprocess.DEVNULL)
        except:
            pass  # If system TTS fails, we already printed the text
        return
        
    # Kill any existing audio playback before starting
    subprocess.run(["pkill", "afplay"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "say"], stderr=subprocess.DEVNULL)
    
    # First try ElevenLabs
    try:
        # Check if we have this text cached
        if text in voice_cache and voice_cache[text]:
            print("Using cached voice response")
            audio_data = voice_cache[text]
            
            # Write cached audio to file
            with open("voice.mp3", "wb") as f:
                f.write(audio_data)
        else:
            # Generate new audio from ElevenLabs
            try:
                audio = elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=eleven_voice_id,
                    model_id="eleven_monolingual_v1",
                    output_format="mp3_44100_128"
                )
                
                # Collect all audio chunks
                chunks = []
                for chunk in audio:
                    if chunk:
                        chunks.append(chunk)
                
                # Check if we received any data
                if not chunks:
                    raise Exception("No audio data received from ElevenLabs")
                
                # Write to file
                with open("voice.mp3", "wb") as f:
                    for chunk in chunks:
                        f.write(chunk)
                
                # Verify file was created with content
                if os.path.getsize("voice.mp3") < 100:
                    raise Exception("Generated audio file is too small")
                
                # Cache the audio for future use if it's short
                if len(text) < 50:
                    with open("voice.mp3", "rb") as f:
                        voice_cache[text] = f.read()
            except Exception as e:
                print(f"ElevenLabs generation failed: {str(e)}")
                raise
        
        # Play the audio if file exists and has content
        if os.path.exists("voice.mp3") and os.path.getsize("voice.mp3") > 100:
            subprocess.run(["afplay", "voice.mp3"], check=True)
            
            # Clean up file after playing
            try:
                os.remove("voice.mp3")
            except:
                pass
            
            return True  # Successfully played audio
        else:
            raise Exception("Audio file missing or too small")
            
    except Exception as e:
        # Log the error
        print(f"ElevenLabs TTS failed: {str(e)}")
        
        # Clean up any partial files
        try:
            if os.path.exists("voice.mp3"):
                os.remove("voice.mp3")
        except:
            pass
        
        # Fall back to system TTS as a backup
        try:
            subprocess.run(["say", text])
            return True  # Successfully used fallback
        except Exception as say_error:
            print(f"System TTS also failed: {str(say_error)}")
            return False  # All TTS methods failed

def chat_with_ai():
    recorder = AudioRecorder()
    conversation = ConversationManager()
    
    if voice_enabled:
        print("\nTARS: Initialized and ready. Humor setting at 75%. Internet access enabled.")
    else:
        print("\nTARS: Initialized and ready. Humor setting at 75%. Internet access enabled. (Text-only mode)")
    
    print("\nPress Enter to start recording, then press Enter again to stop.")
    print("Type text directly, or type 'quit' to exit.")
    
    # Preload common responses in background if voice is enabled
    if voice_enabled:
        def preload_voice_responses():
            for text in ["I'm not sure I understand that.", "Could you clarify?", "Processing your request."]:
                try:
                    audio = elevenlabs_client.text_to_speech.convert(
                        text=text,
                        voice_id=eleven_voice_id,
                        model_id="eleven_monolingual_v1",
                        output_format="mp3_44100_128"
                    )
                    audio_data = b''
                    for chunk in audio:
                        if chunk:
                            audio_data += chunk
                    voice_cache[text] = audio_data
                except:
                    pass  # Ignore errors in preloading
        
        # Start preloading voice responses
        executor.submit(preload_voice_responses)
    
    # Check for previous conversations
    memory_context = conversation.memory.get_memory_context()
    if "Recent conversations" in memory_context:
        print("TARS: I've loaded our previous conversations.")
    
    try:
        while True:
            user_input = input("\nPress Enter to speak or type your message (or 'quit'): ")
            if user_input.lower() == 'quit':
                # Save session before exiting
                conversation.save_session()
                print("\nTARS: Powering down. Session saved to memory.")
                break
                
            # If user typed something instead of just pressing Enter to speak
            if user_input.strip() and user_input.lower() != 'quit':
                print(f"You typed: {user_input}")
                
                # Get AI response in background
                response_future = executor.submit(get_response_for_text, conversation, user_input)
                print("TARS is thinking...")
                
                ai_response = response_future.result()
                
                # Use the speak function which now handles text-only mode
                speak(ai_response)
                continue
                
            # Only try to record if voice is enabled, otherwise prompt for text input
            if not voice_enabled:
                print("Voice input is disabled. Please type your message instead.")
                continue
                
            recorder.start_recording()
            audio_data = recorder.get_audio_data()
            
            if audio_data:
                # Start transcription immediately
                future = executor.submit(process_audio_to_text, audio_data)
                
                # Show processing indicator
                print("Processing your audio...")
                
                user_text = future.result()
                
                if user_text:
                    print(f"You said: {user_text}")
                    
                    # Early acknowledgment - makes the system feel more responsive
                    if len(user_text.split()) > 5:  # Only for longer queries
                        print("TARS: Processing...")
                    
                    # Get AI response in background
                    response_future = executor.submit(get_response_for_text, conversation, user_text)
                    
                    ai_response = response_future.result()
                    
                    # Use the speak function which handles text-only mode
                    speak(ai_response)
                else:
                    print("TARS: I couldn't understand that. Please try again.")
            else:
                print("TARS: I'm detecting a distinct lack of audio. Let's try that again.")
    except KeyboardInterrupt:
        # Also save session on CTRL+C exit
        conversation.save_session()
        print("\nTARS: Session saved to memory. Shutting down.")

if __name__ == "__main__":
    chat_with_ai() 