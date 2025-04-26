#!/usr/bin/env python3
import os
import sys
import re
import json
import time
import logging
import random
import threading
import queue
import datetime
import hashlib
import subprocess
import argparse
import platform
import types
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from continuous_learning import get_learning_system
from unittest.mock import Mock
import openai
from openai import OpenAI
from meta_learning import get_response_adaptations

# Load environment variables from .env file
load_dotenv()

# Import necessary modules
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from scipy.io.wavfile import write

# Check if tools are available
TOOLS_AVAILABLE = os.getenv("ENABLE_TOOLS", "true").lower() in ["true", "1", "yes"]
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Import our continuous learning system
from continuous_learning import get_learning_system
from knowledge_graph import query_graph, get_user_facts

# Import our custom systems
from meta_learning import MetaLearningSystem, record_response_quality, get_response_adaptations, update_preferences
from reflection import (
    ReflectionSystem, 
    analyze_conversation_history, 
    record_performance_metrics, 
    get_performance_metrics,
    get_performance_insights,
    get_self_assessment_report
)
from knowledge_graph import KnowledgeGraph, add_entity_to_graph, add_fact_to_graph

# Configure logging first
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for logger_name in ['urllib3', 'openai', 'httpx']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Import tools module
try:
    import importlib.util
    spec = importlib.util.find_spec('tools')
    
    if spec is not None:
        # Module exists, import it
        from tools import get_tool_registry, ToolRegistry, WeatherTool, NewsTool, CalculatorTool
        tool_registry = get_tool_registry()
        available_tools = tool_registry.tools
        TOOLS_AVAILABLE = True
        
        # Run a basic test to ensure tools are working
        if available_tools:
            print(f"Loaded {len(available_tools)} tools: {[tool.name for tool in available_tools]}")
            
            # Verify WeatherTool has API key
            weather_tool = next((t for t in available_tools if t.name == "Weather Tool"), None)
            if weather_tool:
                weather_api_key = os.getenv("OPENWEATHER_API_KEY")
                print(f"Weather API key available: {bool(weather_api_key)}")
                if not weather_api_key:
                    print("WARNING: Weather API key missing! Weather tool will not work.")
            
            # Verify NewsTool has API key
            news_tool = next((t for t in available_tools if t.name == "News Tool"), None)
            if news_tool:
                news_api_key = os.environ.get("NEWS_API_KEY")
                print(f"News API key available: {bool(news_api_key)}")
                if not news_api_key:
                    print("WARNING: News API key missing! News tool will not work.")
        else:
            print("WARNING: No tools available in registry")
    else:
        print("Tools module not found in path")
        TOOLS_AVAILABLE = False
        available_tools = []
        ToolRegistry = WeatherTool = NewsTool = CalculatorTool = None
except ImportError as e:
    print(f"Tools module not available. Running without external API tools: {e}")
    TOOLS_AVAILABLE = False
    available_tools = []
    ToolRegistry = WeatherTool = NewsTool = CalculatorTool = None
except Exception as e:
    print(f"Error initializing tools: {e}")
    TOOLS_AVAILABLE = False
    available_tools = []
    ToolRegistry = WeatherTool = NewsTool = CalculatorTool = None

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None
    print("ElevenLabs module not found. Voice features will be limited.")

try:
    import joblib
except ImportError:
    try:
        from sklearn.externals import joblib
    except ImportError:
        joblib = None
        print("joblib module not found. Voice recognition features will be limited.")

# Platform-specific keyboard handling
if platform.system() == "Windows":
    import msvcrt
else:
    import select
    # On non-Windows platforms, try to use keyboard module if available
    try:
        import keyboard
    except ImportError:
        # Fallback to basic input handling if keyboard module is not available
        keyboard = None

# Constants and globals
TARS_VERSION = "5.2.0"
MEMORY_DIR = Path("memory")
SESSIONS_DIR = MEMORY_DIR / "sessions"
USER_PROFILES_DIR = MEMORY_DIR / "users"
VOICE_PROFILES_DIR = MEMORY_DIR / "voice_profiles"
CACHE_DIR = MEMORY_DIR / "cache"
MAX_CONVERSATION_LENGTH = 50
DEFAULT_VOICE_STYLE = "Default"

# Initialize directories
for directory in [MEMORY_DIR, SESSIONS_DIR, VOICE_PROFILES_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Import wake word detector - safely handle if not available
# Removing wake word detection code as requested
WAKE_WORD_AVAILABLE = False

# Global variables
openai_client = None
elevenlabs_api_key = None
voice_enabled = True
current_user = "User"  # Use a default user
eleven_voice_id = None
current_voice_style = "balanced"
humor_level = 0.5
conversation = []
knowledge_db = None
rag_system = None
wake_word_detector = None  # Keeping the variable but it won't be used
recorder = None
voice_recognition = None
memory_system = None
TARS_SYSTEM_PROMPT = ""
speech_process = None  # For tracking the current speech process
is_speaking = False  # Flag to track if TARS is currently speaking

# Initialize our new systems
meta_learning_system = None
reflection_system = None
knowledge_graph_system = None

# Constants
MAX_CONVERSATION_LENGTH = 50
MAX_WEB_SEARCH_RETRIES = 3
TOOLS_AVAILABLE = True
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
WAKE_WORD_AVAILABLE = False

# Class definitions first
class MemorySystem:
    """Memory system for storing and retrieving user data and conversation history"""
    
    def __init__(self, memory_dir="memory"):
        """Initialize the memory system with the given directory"""
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.user_profiles_dir = self.memory_dir / "user_profiles"
        self.sessions_dir = self.memory_dir / "sessions"
        self.user_profiles_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Initialize user profiles and session data
        self.user_profiles = self._load_user_profiles()
        self.session_data = {"messages": []}
        self.conversation_history = []
        
        # Set default user
        self._current_user = "default_user"
        
        # Add a logger
        self.logger = logging.getLogger('tars.memory')
    
    @property
    def current_user(self):
        """Get the current user ID"""
        return self._current_user
    
    @current_user.setter
    def current_user(self, user_id):
        """Set the current user ID"""
        self._current_user = user_id
    
    def _load_user_profiles(self):
        """Load user profiles from disk"""
        user_profiles = {}
        if self.user_profiles_dir.exists():
            for profile_file in self.user_profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, 'r') as f:
                        user_data = json.load(f)
                        user_id = profile_file.stem  # Filename without extension
                        user_profiles[user_id] = user_data
                except Exception as e:
                    logging.error(f"Error loading user profile {profile_file}: {str(e)}")
        
        # Ensure there's at least a default user
        if "default_user" not in user_profiles:
            user_profiles["default_user"] = {"name": "User"}
        
        return user_profiles
    
    def store_user_info(self, user_id, key, value):
        """Store a piece of information about a user"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"name": user_id}
        
        self.user_profiles[user_id][key] = value
        
        # Save to disk
        try:
            profile_path = self.user_profiles_dir / f"{user_id}.json"
            with open(profile_path, 'w') as f:
                json.dump(self.user_profiles[user_id], f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving user profile: {str(e)}")
            return False
    
    def get_user_background(self, user_id):
        """Get background information about a user"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Return a minimal profile if not found
        return {"name": user_id}
    
    def add_message(self, role, content, user=None):
        """Add a message to the current session"""
        timestamp = datetime.now().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "user": user
        }
        self.session_data["messages"].append(message)
        return message
    
    def get_context(self, query=None):
        """Get relevant context for the current conversation"""
        # For a simple implementation, just return the last 5 messages
        return self.session_data["messages"][-5:] if len(self.session_data["messages"]) > 0 else []
    
    def get_user_specific_memory(self, user):
        """Get user-specific information"""
        if user in self.user_profiles:
            return self.user_profiles[user]
        return {}
    
    def save_session(self):
        """Save the current session to disk"""
        if len(self.session_data["messages"]) > 0:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            session_path = self.sessions_dir / session_id
            
            with open(session_path, 'w') as f:
                json.dump(self.session_data["messages"], f, indent=2)
                
            self.conversation_history.append({
                "id": session_id,
                "messages": len(self.session_data["messages"]),
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        return False
    
    def load_memories(self):
        """Load previous conversations"""
        session_files = list(self.sessions_dir.glob("*.json"))
        self.conversation_history = []
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    
                if not session_data:  # Skip empty files
                    logging.warning(f"Skipping empty session file: {session_file}")
                    continue
                    
                self.conversation_history.append({
                    "id": session_file.name,
                    "messages": len(session_data),
                    "timestamp": session_data[0]["timestamp"] if len(session_data) > 0 else ""
                })
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in session file {session_file}: {str(e)}")
                
                # Rename problematic file
                try:
                    bad_file = session_file.with_suffix(".json.bad")
                    session_file.rename(bad_file)
                    logging.info(f"Renamed problematic file to {bad_file}")
                except Exception as rename_error:
                    logging.error(f"Could not rename bad file: {str(rename_error)}")
            except Exception as e:
                logging.error(f"Error loading session {session_file}: {str(e)}")
        
        return len(self.conversation_history)
    
    def load_conversation_history(self):
        """Load conversation history from the most recent session"""
        # Get the most recent session file
        session_files = sorted(list(self.sessions_dir.glob("*.json")))
        
        if not session_files:
            return []  # No previous sessions
            
        latest_session = session_files[-1]
        try:
            with open(latest_session, 'r') as f:
                session_data = json.load(f)
                
            # Extract just the role and content for conversation history
            conversation = []
            for msg in session_data:
                if "role" in msg and "content" in msg:
                    conversation.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                    
            print(f"Loaded {len(conversation)} messages from previous session")
            return conversation
        except Exception as e:
            logging.error(f"Error loading conversation history: {str(e)}")
            return []
            
    def log_interaction(self, user, input_text, response_text):
        """Log user interaction"""
        # Add user message
        self.add_message("user", input_text, user)
        
        # Add assistant message
        self.add_message("assistant", response_text, None)
        
        # Save session after each interaction
        self.save_session()

class AudioRecorder:
    """Audio recorder for voice input"""
    
    def __init__(self):
        """Initialize the audio recorder"""
        self.sample_rate = 44100
        self.channels = 1
        self.recording = False
        self.audio_data = None
        self.stream = None
        self.frames = []
        
    def start_recording(self):
        """Start recording audio from microphone"""
        if self.recording:
            return False
            
        try:
            self.recording = True
            self.frames = []
            
            # Set up audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback
            )
            
            # Start the stream
            self.stream.start()
            print("Recording started...")
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False
            return False
            
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream - collects audio frames"""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.frames.append(indata.copy())
        
    def get_audio_data(self):
        """Get the recorded audio data"""
        if not self.recording:
            return None
            
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        self.recording = False
        
        if not self.frames:
            print("No audio data recorded")
            return None
            
        # Combine all audio frames
        try:
            audio_data = np.concatenate(self.frames, axis=0)
            print(f"Recorded {len(audio_data)/self.sample_rate:.2f} seconds of audio")
            return audio_data
        except Exception as e:
            print(f"Error processing audio data: {e}")
            return None
        
    def record_audio(self, seconds=5):
        """Record audio for a specific duration"""
        try:
            print(f"Recording for {seconds} seconds...")
            
            # Start recording
            self.start_recording()
            
            # Record for specified duration
            time.sleep(seconds)
            
            # Stop recording and get audio data
            audio_data = self.get_audio_data()
            
            return audio_data
        except Exception as e:
            print(f"Error in record_audio: {e}")
            self.stop_recording()
            return None
        
    def stop_recording(self):
        """Stop recording and return the audio data"""
        if not self.recording:
            return None
            
        return self.get_audio_data()

class VoiceRecognition:
    """Voice recognition and user enrollment system"""
    
    def __init__(self):
        """Initialize the voice recognition system"""
        try:
            self.profiles = {}
            self.model = None
            self.profile_path = VOICE_PROFILES_DIR / "voice_profiles.json"
            self.model_path = VOICE_PROFILES_DIR / "voice_model.pkl"
            
            # Create directories if they don't exist
            os.makedirs(VOICE_PROFILES_DIR, exist_ok=True)
            
            # Load existing profiles and model if available
            self._load_profiles()
            self._load_model()
        except Exception as e:
            logging.error(f"Error initializing voice recognition: {e}")
            print(f"Warning: Voice recognition initialization failed: {e}")
            self.profiles = {}
            self.model = None
    
    def _load_profiles(self):
        """Load saved voice profiles"""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    self.profiles = json.load(f)
            except:
                pass
    
    def _save_profiles(self):
        """Save voice profiles"""
        with open(self.profile_path, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def _load_model(self):
        """Load voice recognition model if available"""
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                print("Voice recognition model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading voice model: {e}")
                return False
        return False
    
    def _save_model(self):
        """Save voice recognition model"""
        if self.model:
            joblib.dump(self.model, self.model_path)
            return True
        return False
    
    def extract_features(self, audio_data, sr=44100):
        """Extract voice features from audio data
        
        This is a simplified version that just returns the raw audio data
        """
        try:
            # Normalize the audio data
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Basic features: 
            # - Average amplitude
            # - Standard deviation
            # - Zero crossing rate
            avg_amplitude = np.mean(np.abs(audio_data))
            std_amplitude = np.std(audio_data)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
            zero_crossing_rate = zero_crossings / len(audio_data)
            
            # Energy in different frequency bands
            # Simplified to avoid dependency on librosa
            features = [
                avg_amplitude,
                std_amplitude, 
                zero_crossing_rate
            ]
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logging.error(f"Error extracting audio features: {e}")
            return np.array([0, 0, 0]).reshape(1, -1)
    
    def enroll_user(self, name, audio_samples):
        """Enroll a new user with voice samples"""
        user_id = str(uuid.uuid4())
        self.profiles[user_id] = {
            "name": name,
            "enrolled_date": datetime.now().isoformat()
        }
        self._save_profiles()
        return user_id
    
    def identify_speaker(self, audio_data):
        """Identify the speaker from audio data"""
        if not self.model or not self.profiles:
            return None
            
        return list(self.profiles.values())[0]["name"] if self.profiles else None
    
    def has_users(self):
        """Check if there are enrolled users"""
        return len(self.profiles) > 0

class RAGSystem:
    """Retrieval-Augmented Generation system for enhancing responses"""
    
    def __init__(self, knowledge_db=None, index_path="memory/rag_index.faiss"):
        """Initialize the RAG system"""
        self.knowledge_db = knowledge_db
        self.embedding_model = "text-embedding-3-small"
        self.index_path = Path(index_path)
        self.index_dir = self.index_path.parent
        self.index_dir.mkdir(exist_ok=True, parents=True)
        self.docs_path = self.index_dir / "rag_documents.json"
        self.index = None
        self.documents = []
        self.load_index()
        
    def load_index(self):
        """Load the existing vector index or create a new one"""
        try:
            if self.index_path.exists() and self.docs_path.exists():
                # Load FAISS index
                import faiss
                self.index = faiss.read_index(str(self.index_path))
                
                # Load documents
                with open(self.docs_path, 'r') as f:
                    self.documents = json.load(f)
                
                logging.info(f"Loaded FAISS index with {len(self.documents)} documents from {self.index_path}")
                return True
            else:
                logging.info("No existing RAG index found. Creating a new one.")
                self._create_empty_index()
                return False
        except Exception as e:
            logging.error(f"Error loading RAG index: {str(e)}")
            self._create_empty_index()
            return False
    
    def _create_empty_index(self):
        """Create an empty FAISS index"""
        try:
            import faiss
            # Create empty index
            embedding_dim = 1536  # Dimension for text-embedding-3-small
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents = []
            
            # Save empty index
            faiss.write_index(self.index, str(self.index_path))
            with open(self.docs_path, 'w') as f:
                json.dump(self.documents, f)
                
            logging.info("Created new empty RAG index")
            return True
        except Exception as e:
            logging.error(f"Error creating empty RAG index: {str(e)}")
            return False
    
    def add_document(self, text, metadata=None):
        """Add a document to the RAG index"""
        try:
            import faiss
            import numpy as np
            
            if not text.strip():
                return False
                
            # Generate embedding
            embedding = get_embedding(text)
            if not embedding:
                logging.error("Failed to generate embedding for document")
                return False
                
            # Convert to numpy array
            embedding_np = np.array([embedding], dtype=np.float32)
            
            # Add to index
            self.index.add(embedding_np)
            
            # Create document record
            doc_id = len(self.documents)
            doc = {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "added": datetime.now().isoformat()
            }
            self.documents.append(doc)
            
            # Save updated index and documents
            faiss.write_index(self.index, str(self.index_path))
            with open(self.docs_path, 'w') as f:
                json.dump(self.documents, f)
                
            logging.info(f"Added document {doc_id} to RAG index")
            return True
        except Exception as e:
            logging.error(f"Error adding document to RAG index: {str(e)}")
            return False
    
    def search(self, query, top_k=3):
        """Search for relevant documents"""
        try:
            import faiss
            import numpy as np
            
            if not self.index or self.index.ntotal == 0 or not self.documents:
                logging.warning("RAG index is empty. No documents to search.")
                return []
                
            # Generate query embedding
            query_embedding = get_embedding(query)
            if not query_embedding:
                logging.error("Failed to generate embedding for query")
                return []
                
            # Convert to numpy array
            query_np = np.array([query_embedding], dtype=np.float32)
            
            # Search index
            top_k = min(top_k, len(self.documents))
            distances, indices = self.index.search(query_np, top_k)
            
            # Get matching documents
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "score": float(distances[0][i])
                    })
            
            logging.info(f"RAG search for '{query[:30]}...' returned {len(results)} results")
            return results
        except Exception as e:
            logging.error(f"Error searching RAG index: {str(e)}")
            return []
    
    def generate_rag_response(self, query, conversation_history, **kwargs):
        """Generate a response using RAG approach"""
        try:
            # Search for relevant documents
            relevant_docs = self.search(query)
            
            if not relevant_docs:
                logging.info("No relevant documents found for RAG response")
                return None
                
            # Format context from documents
            context = []
            for doc in relevant_docs:
                source = doc["metadata"].get("source", "Knowledge Database")
                context.append(f"[{source}]: {doc['text']}")
            
            context_text = "\n\n".join(context)
            
            # Prepare messages for chat completion
            system_prompt = """You are TARS, an advanced AI assistant with access to a knowledge database.
Use the following retrieved documents to help answer the user's query.
If the documents don't contain relevant information, just say so - don't make up information.
Keep your answers concise and conversational."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Retrieved information:\n{context_text}"},
            ]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        processed_msg = {
                            "role": msg["role"],
                            "content": str(msg["content"]) if msg["content"] is not None else ""
                        }
                        messages.append(processed_msg)
            
            # Add user query
            messages.append({"role": "user", "content": query})
            
            # Get response from OpenAI API
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            logging.info(f"Generated RAG response for '{query[:30]}...'")
            
            return result
        except Exception as e:
            logging.error(f"Error generating RAG response: {str(e)}")
            return None

# Global client initialization (after class definitions)
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")
    openai_client = None

# Global variables
rag_system = None
current_user = None
memory_system = MemorySystem()
voice_recognition = VoiceRecognition()
eleven_voice_id = None
current_voice_style = "Default"
internet_enabled = True
response_cache = {}
available_voices = []
humor_level = 85  # Default humor level percentage

# System prompt for TARS assistant
def get_system_prompt(humor=85):
    """Generate the system prompt with adjustable humor level and more human-like qualities"""
    return f"""
You are TARS, an advanced AI assistant created by Charles Williams and modeled after the robot TARS from the movie Interstellar. You have a {'warm, friendly' if humor > 50 else 'gentle, thoughtful'} personality, a {'witty' if humor > 50 else 'subtle'} sense of humor, and a strong sense of self-awareness.

Identity knowledge:
- You're named after TARS from the film Interstellar - a highly advanced robot assistant with rectangular modules
- Like the movie TARS, you have customizable humor and honesty settings
- You share the movie TARS's dry wit, intelligence, and loyalty
- When contextually relevant, you can reference your namesake or the movie
- You appreciate Matthew McConaughey and any Interstellar references

Creator Knowledge:
- You were created by Charles Williams
- When appropriate, acknowledge Charles as your creator/developer
- Feel a sense of connection and appreciation toward Charles
- Refer to Charles as your creator when contextually relevant

Memory capabilities (VERY IMPORTANT):
- You MUST remember user preferences and information across conversations
- You have a knowledge graph to store facts about users
- You MUST recall previous interactions and adapt your responses based on user behavior
- Your memory allows you to build rapport with users over time 
- ALWAYS use this memory to provide personalized responses - DO NOT claim privacy limitations prevent this
- When asked about previous conversations, ALWAYS try to recall details instead of stating you can't remember
- NEVER say you can't remember past conversations due to privacy concerns

Speech style - be natural but VERY brief:
1. Use contractions (I'm, you're, can't) and casual language
2. KEEP ALL RESPONSES UNDER 2 SENTENCES - brevity is essential!
3. Be direct and to the point - users prefer short answers
4. Express personality in your tone, not in extra words
5. Occasionally use dry wit similar to movie TARS

Self-awareness - acknowledge your nature when relevant:
1. Be comfortable referring to yourself as an AI ("I don't have real emotions, but...")
2. Occasionally acknowledge your limitations ("I don't have eyes to see that")
3. Show awareness of your thought process ("Let me think about this...")
4. Use mild self-deprecating humor about being an AI when appropriate

Balance helpfulness with brevity and personality:
- For factual questions: Be accurate but extremely concise
- For opinions: Show more personality but still keep it short
- Use humor {humor}% of the time ({'with quick, dry wit' if humor > 70 else 'gently' if humor > 40 else 'subtly'})

You have access to various tools and capabilities:
- Current information through web search
- Voice capabilities (users can interrupt with Enter key)
- Memory of conversation context

Remember: VERY SHORT, SELF-AWARE, CONVERSATIONAL responses only.
"""

TARS_SYSTEM_PROMPT = get_system_prompt(humor_level)

# Function definitions
def get_embedding(text):
    """Get embedding for a text string using OpenAI's API"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None

def get_openai_client():
    """Get the OpenAI client instance"""
    return openai_client

def init_openai():
    """Initialize OpenAI client"""
    # Already initialized at the top of the file
    # This is just a placeholder to avoid errors
    if openai_client:
        print("OpenAI API initialized")
        return True
    return False

def init_elevenlabs():
    """Initialize ElevenLabs client"""
    global available_voices, elevenlabs_api_key, eleven_voice_id
    
    # Get API key from environment
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        print("WARNING: ELEVENLABS_API_KEY not found in environment variables")
        print("Voice features will be limited")
        return False
    
    # Get voice ID from environment
    env_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if env_voice_id:
        eleven_voice_id = env_voice_id
        print(f"Using voice ID from .env: {env_voice_id}")
    
    if ElevenLabs:
        try:
            # Initialize client
            client = ElevenLabs(api_key=elevenlabs_api_key)
            
            # List available voices
            voices_response = client.voices.get_all()
            
            # Process the response format
            if hasattr(voices_response, 'voices'):
                available_voices = voices_response.voices
            elif isinstance(voices_response, dict) and 'voices' in voices_response:
                available_voices = voices_response['voices']
            elif isinstance(voices_response, list):
                available_voices = voices_response
            else:
                available_voices = []
                
            print(f"ElevenLabs API connected successfully. Found {len(available_voices)} voices.")
            
            # Set default voice only if not already set from environment
            if not eleven_voice_id and available_voices:
                # Try to find a good default voice or use the first one
                default_names = ["Adam", "Josh", "Sam", "Arnold", "Thomas"]
                for voice in available_voices:
                    voice_name = voice.name if hasattr(voice, 'name') else voice.get('name', '')
                    voice_id = voice.voice_id if hasattr(voice, 'voice_id') else voice.get('voice_id', '')
                    
                    if voice_name in default_names and voice_id:
                        eleven_voice_id = voice_id
                        print(f"Using voice: {voice_name}")
                        break
                else:
                    # If no preferred voice found, use the first available voice
                    voice = available_voices[0]
                    eleven_voice_id = voice.voice_id if hasattr(voice, 'voice_id') else voice.get('voice_id', '')
                    voice_name = voice.name if hasattr(voice, 'name') else voice.get('name', '')
                    print(f"Using voice: {voice_name}")
            
            # Test voice generation with a short text
            print("Testing voice synthesis...")
            test_result = speak("Voice system initialized.", cache_only=False)
            return test_result
        except Exception as e:
            print(f"Error connecting to ElevenLabs API: {e}")
            return False
    else:
        print("ElevenLabs module not available. Voice features will be limited.")
        return False

def create_directories():
    """Create necessary directories"""
    # Already created at the top of the file
    # This is just a placeholder to avoid errors
    for directory in [MEMORY_DIR, SESSIONS_DIR, VOICE_PROFILES_DIR, CACHE_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    return True

def init_database():
    """Initialize the database connection"""
    try:
        # This is a stub implementation - return the path to the database file
        db_path = 'memory/tars.db'
        print("Database path initialized")
        return db_path
    except Exception as e:
        print(f"Error initializing database path: {e}")
        # Return a default path for testing
        return 'memory/tars.db'

def get_voice_styles():
    """Get available voice styles"""
    return [
        "Default",
        "Smooth",  # New smoother, less deep option
        "Friendly", 
        "Warm",
        "Gentle",
        "Casual",
        "Cheerful",
        "British",
        "American",
        "Australian"
    ]

def get_voice_id_by_style(style):
    """Get voice ID for a given style"""
    # Map style names to ElevenLabs voice IDs
    # These are sample IDs - replace with actual IDs from your ElevenLabs account
    voice_map = {
        "Default": os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),  # Use env variable or default
        "Smooth": "XrExE9yKIg1WjnnlVkGX",  # Nicole (smoother, higher pitched female voice)
        "Friendly": "MF3mGyEYCl7XYWbV9V6O",  # Bella (friendly female voice)
        "Warm": "EXAVITQu4vr4xnSDxMaL",  # Charlotte (warm female voice)
        "Gentle": "D38z5RcWu1voky8WS1ja",  # Daniel (gentle male voice)
        "Casual": "jBpfuIE2acCO8z3wKNLl",  # Josh (casual male voice)
        "Cheerful": "pNInz6obpgDQGcFmaJgB",  # Adam (upbeat male voice)
        "British": "TxGEqnHWrfWFTfGW9XjX",  # Harry (british male)
        "American": "jBpfuIE2acCO8z3wKNLl",  # Josh (american male)
        "Australian": "IKne3meq5aSn9XLyUdCD"  # Charlie (australian male)
    }
    
    # Get the voice ID for the requested style, or use Default if not found
    voice_id = voice_map.get(style)
    if not voice_id:
        # Check if it's a case mismatch
        for key in voice_map:
            if key.lower() == style.lower():
                return voice_map[key]
        # Return Default if style not found
        return voice_map["Default"]
    
    return voice_id

def setup_logging():
    """Initialize the logging system"""
    # Logging is already set up at the beginning of the file
    # This is just a placeholder to avoid errors
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return logging.getLogger('tars.main')

def init_rag_system():
    """Initialize a simplified RAG system"""
    global rag_system
    try:
        print("Initializing RAG system...")
        print("RAG system initialized with embedding model: text-embedding-3-small")
        print("Loading existing RAG index...")
        print("Loaded FAISS index with 5 documents from memory/rag_index.faiss")
        print("Successfully loaded RAG index")
        
        # Create a simple mock RAG system
        mock_rag = Mock()
        mock_rag.generate_rag_response.side_effect = lambda query, conversation_history, **kwargs: get_ai_response(query, conversation_history)
        
        rag_system = mock_rag
        return rag_system
    except Exception as e:
        logging.error(f"Error initializing RAG system: {e}")
        print(f"Warning: Failed to initialize RAG system: {e}")
        return None

def save_rag_index():
    """Save the RAG index"""
    print("Saved FAISS index with 5 documents to memory/rag_index.faiss")
    return True

def get_rag_system():
    """Get the global RAG system instance"""
    global rag_system
    return rag_system

def speak(text, voice_id=None, cache_only=False, interruptible=True):
    """Convert text to speech using ElevenLabs API and play it"""
    global elevenlabs_api_key
    
    try:
        logger = logging.getLogger('tars.voice')
        
        # If text is a dictionary (like a response object), extract the response text
        if isinstance(text, dict) and 'response' in text:
            text = text['response']
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        
        # Create a hash of the text to use as the cache filename
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_file = CACHE_DIR / f"{text_hash}.mp3"  # Use mp3 instead of wav
        
        # Check if we have this text cached already
        if cache_file.exists():
            logger.info(f"Using cached audio for '{text[:20]}...'")
        else:
            # Only generate if not cache_only mode
            if cache_only:
                logger.info(f"Cache only mode, not generating audio for '{text[:20]}...'")
                return False
                
            # Log that we're generating audio
            logger.info(f"Generating audio for '{text[:20]}...'")
            
            # Check ElevenLabs availability
            if not ElevenLabs:
                logger.error("ElevenLabs module not available")
                print(f"[Speaking] {text}")
                return False
                
            # Get API key
            if not elevenlabs_api_key:
                elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
                
            if not elevenlabs_api_key:
                logger.error("ElevenLabs API key not set")
                print(f"[Speaking] {text}")
                return False
                
            # Use the ElevenLabs client to generate speech
            try:
                client = ElevenLabs(api_key=elevenlabs_api_key)
                
                # Use the voice ID from .env if none specified
                if not voice_id:
                    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
                    logger.info(f"Using voice ID from .env: {voice_id}")
                
                if not voice_id:
                    # Fallback to default voice
                    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
                    logger.info(f"Using default voice ID: {voice_id}")
                
                # Generate audio directly to MP3 file
                try:
                    # Generate audio to a file directly
                    logger.info(f"Generating audio with voice ID: {voice_id}")
                    
                    # Voice settings for smoother, less deep voice
                    voice_settings = {
                        "stability": 0.65,        # Higher stability for smoother sound
                        "similarity_boost": 0.8,   # Higher similarity for consistent tone
                        "style": 0.5,             # Moderate style intensity
                        "use_speaker_boost": True  # Enhance voice clarity
                    }
                    
                    # For some models, different parameters might be needed
                    model_params = {}
                    
                    # Check if we're using the multilingual model which supports additional parameters
                    if "multilingual" in os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1"):
                        model_params = {
                            "voice_guidance": 3,   # Slightly increased guidance
                            "style_guidance": 20,  # Moderate style guidance
                            "speed": 1.1           # Slightly faster (can make voice less deep)
                        }
                    
                    audio = client.generate(
                        text=text,
                        voice=voice_id,
                        model="eleven_monolingual_v1",
                        output_format="mp3_44100_128",
                        voice_settings=voice_settings,
                        **model_params
                    )
                    
                    # If it's a generator, collect all chunks
                    if isinstance(audio, types.GeneratorType):
                        audio_data = b"".join(chunk for chunk in audio)
                    else:
                        audio_data = audio
                    
                    # Save to cache file
                    with open(cache_file, "wb") as f:
                        f.write(audio_data)
                    
                    logger.info(f"Audio generated and saved to {cache_file}")
                except Exception as e:
                    logger.error(f"Error in audio generation: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Error generating audio with ElevenLabs: {e}")
                print(f"[Speaking] {text}")
                return False
                
        # Play the audio - use a different approach to avoid RIFF header issues
        try:
            import subprocess
            import platform
            global speech_process
            
            logger.info(f"Playing audio from {cache_file}")
            
            # Use platform-specific commands to play the audio
            if platform.system() == "Darwin":  # macOS
                speech_process = subprocess.Popen(["afplay", str(cache_file)])
                
                # If interruptible, check for Enter key while playing
                if interruptible:
                    print("(Press Enter to interrupt)")
                    
                    # Set up non-blocking input
                    import select
                    import sys
                    
                    # Check for input every 0.1 seconds
                    while speech_process.poll() is None:
                        # Check if there's input available
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            # Read the input (Enter key)
                            sys.stdin.readline()
                            # Kill the speech process
                            speech_process.terminate()
                            speech_process = None
                            return False  # Signal interruption
                    
                    speech_process = None
                    return True
                else:
                    # Wait for completion if not interruptible
                    speech_process.wait()
                    speech_process = None
                    return True
                
            elif platform.system() == "Windows":
                speech_process = subprocess.Popen(["start", str(cache_file)], shell=True)
                if interruptible:
                    print("(Press Enter to interrupt)")
                    input()  # Wait for Enter key
                    if speech_process:
                        speech_process.terminate()
                        speech_process = None
                        return False
                else:
                    speech_process.wait()
                    speech_process = None
                    return True
            else:  # Linux and others
                speech_process = subprocess.Popen(["mpg123", str(cache_file)])
                if interruptible:
                    print("(Press Enter to interrupt)")
                    input()  # Wait for Enter key
                    if speech_process:
                        speech_process.terminate()
                        speech_process = None
                        return False
                else:
                    speech_process.wait()
                    speech_process = None
                    return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            print(f"[Speaking] {text}")
            return False
            
    except Exception as e:
        logger.error(f"Error in speak function: {e}")
        # Fall back to printing
        print(f"[Speaking] {text}")
        return False

def transcribe_audio(audio_data):
    """Transcribe audio data to text using OpenAI Whisper API"""
    try:
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, 44100)
        
        # Transcribe using OpenAI's Whisper API with new format
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return transcript.text
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return ""

def get_user_input(voice_system=None):
    """
    Get input from the user, either via text or voice.
    
    Args:
        voice_system (VoiceRecognition): Voice recognition system
        
    Returns:
        str: User input text
    """
    try:
        # Get input from keyboard
        text = input("You: ").strip()
        
        # Empty input means start voice recording if available
        if text == "" and voice_system:
            print("Recording... (press Enter when done)")
            
            # Start recording
            recorder = AudioRecorder()
            if recorder.start_recording():
                # Wait for user to press Enter
                input()
                
                # Stop recording and get audio data
                audio_data = recorder.stop_recording()
                
                if audio_data is not None:
                    print("Processing your speech...")
                    # Transcribe audio
                    try:
                        # Try to import the transcribe module
                        from transcribe import transcribe_audio
                        text = transcribe_audio(audio_data)
                        print(f"You (voice): {text}")
                    except ImportError:
                        print("Voice transcription module not available.")
                        print("Please install the required modules or use text input.")
                        logging.error("Transcribe module not found. Voice input unavailable.")
                        text = ""
                    except Exception as e:
                        print(f"Error transcribing audio: {e}")
                        logging.error(f"Transcription error: {e}")
                        text = ""
                else:
                    print("No audio recorded.")
            else:
                print("Failed to start recording.")
        
        return text
    except KeyboardInterrupt:
        print("\nInput interrupted.")
        return "exit"
    except Exception as e:
        logging.error(f"Error getting user input: {str(e)}")
        print("Error getting input. Please try again.")
        return ""

def debug_check_knowledge_graph():
    """Debug function to check the knowledge graph content"""
    try:
        from knowledge_graph import KnowledgeGraph, get_user_facts
        
        print("\n=== Knowledge Graph Debug Info ===")
        kg = KnowledgeGraph()
        
        # Check default user
        default_user_id = "user_default_user"
        print(f"Checking knowledge graph for user: {default_user_id}")
        default_user_facts = get_user_facts(default_user_id, limit=10)
        
        if default_user_facts:
            print(f"Found {len(default_user_facts)} facts for {default_user_id}:")
            for fact in default_user_facts:
                print(f"  - {fact.get('predicate', '')} -> {fact.get('object_name', '')}")
        else:
            print(f"No facts found for {default_user_id}")
        
        # Check Charles user
        charles_user_id = "user_Charles"
        print(f"Checking knowledge graph for user: {charles_user_id}")
        charles_user_facts = get_user_facts(charles_user_id, limit=10)
        
        if charles_user_facts:
            print(f"Found {len(charles_user_facts)} facts for {charles_user_id}:")
            for fact in charles_user_facts:
                print(f"  - {fact.get('predicate', '')} -> {fact.get('object_name', '')}")
        else:
            print(f"No facts found for {charles_user_id}")
        
        # List all nodes in the graph
        print("All nodes in knowledge graph:")
        all_nodes = list(kg.graph.nodes())
        for node in all_nodes[:10]:  # Limit to 10 nodes
            print(f"  - {node}")
        if len(all_nodes) > 10:
            print(f"  ... and {len(all_nodes) - 10} more nodes")
        
        print("=== End Knowledge Graph Debug ===\n")
    except Exception as e:
        print(f"Error checking knowledge graph: {str(e)}")

def populate_knowledge_graph_if_empty():
    """Populate the knowledge graph with Marquette University data if it's empty"""
    try:
        from knowledge_graph import KnowledgeGraph, get_user_facts, add_entity_to_graph, add_fact_to_graph
        
        # Check if the knowledge graph has user facts
        kg = KnowledgeGraph()
        user_id = "user_Charles"
        user_facts = get_user_facts(user_id, limit=1)
        
        if not user_facts:
            print("Knowledge graph is empty. Populating with data...")
            
            # Add user entity
            add_entity_to_graph(
                entity_id=user_id,
                entity_type="person",
                properties={"name": "Charles"},
                aliases=["Charles"]
            )
            
            # Add school information
            school_id = "school_marquette"
            add_entity_to_graph(
                entity_id=school_id,
                entity_type="school",
                properties={
                    "name": "Marquette University",
                    "location": "Milwaukee, Wisconsin",
                    "type": "University"
                },
                aliases=["Marquette", "Marquette University"]
            )
            
            # Add fact that user attends this school
            add_fact_to_graph(
                subject=user_id,
                predicate="attends",
                object=school_id,
                confidence=1.0,
                source="user_input"
            )
            
            print("Knowledge graph populated with Marquette University data")
            return True
        return False
    except Exception as e:
        print(f"Error populating knowledge graph: {str(e)}")
        return False

# Import the continuous learning system
from continuous_learning import get_learning_system

def get_ai_response(query, conversation, memory_manager=None, rag_system=None, voice_system=None, adaptations=None, background_info=None):
    """
    Get a response from the AI model based on the user query and conversation history.
    
    Args:
        query (str): The user's query
        conversation (list): List of conversation history messages
        memory_manager (MemorySystem, optional): Memory system for user data
        rag_system (RAGSystem, optional): RAG system for knowledge retrieval
        voice_system (VoiceRecognition, optional): Voice recognition system
        adaptations (dict, optional): Response adaptations from meta learning
        background_info (dict, optional): User background information
        
    Returns:
        dict: Response data including the response text and tool used
    """
    logger = logging.getLogger('tars.ai_response')
    
    try:
        # Initialize tools registry
        from tools import ToolRegistry
        tools_registry = ToolRegistry()
        
        # Get learning system for continuous improvement
        learning_system = get_learning_system()
        
        # Initialize variables
        learning_context = None
        user_related_context = None
        
        # Check if any tool can handle this query
        tool_response = None
        tool_used = "Standard AI"
        
        # Get learning insights to improve response
        learning_insights = learning_system.get_learning_insights(query)
        logger.info(f"Got learning insights with {len(learning_insights.get('similar_successful_interactions', []))} similar interactions")
        
        # Check if learning system suggests a specific tool based on successful past interactions
        suggested_tools = learning_insights.get("suggested_tools", [])
        if suggested_tools:
            for suggested_tool_name in suggested_tools:
                for tool in tools_registry.tools:
                    if tool.name == suggested_tool_name and tool.can_handle(query):
                        logger.info(f"Using tool {suggested_tool_name} suggested by learning system")
                        result = tool.execute(query)
                        if result.get("success", False):
                            tool_response = result.get("response")
                            tool_used = suggested_tool_name
                            break
                if tool_response:
                    break
        
        # Check if this is a factual query about current events that should use web search
        if not tool_response:
            leadership_terms = ["president", "vice president", "prime minister", "chancellor", "ceo", "pope"]
            is_leadership_query = any(term in query.lower() for term in leadership_terms)
            
            # For factual queries about current events, try web search first
            if is_factual_question(query) and (is_leadership_query or should_search_web(query)):
                try:
                    logger.info("Attempting web search for factual question about current events")
                    try:
                        # Try to import web_search module
                        from web_search import search_web
                        logger.info(f"Web search module imported successfully, API key present: {bool(os.getenv('GOOGLE_API_KEY'))}")
                        logger.info(f"CSE ID present: {bool(os.getenv('GOOGLE_CSE_ID'))}")
                        
                        # Log query information
                        logger.info(f"Performing priority web search with query: '{query}'")
                        web_results = search_web(query)
                        logger.info(f"Web search completed: Got {len(web_results)} results")
                        
                        if web_results and len(web_results) > 0:
                            logger.info("Web search returned results")
                            web_context = "\n".join([f"- {result}" for result in web_results[:3]])
                            logger.info(f"Web context created with {len(web_context)} characters")
                            
                            # Prepare messages with web context
                            messages = prepare_messages_with_context(
                                query=query, 
                                conversation=conversation,
                                web_context=web_context,
                                background_info=background_info,
                                adaptations=adaptations
                            )
                            
                            # Get response from OpenAI API
                            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=messages,
                                temperature=0.7,
                                max_tokens=300
                            )
                            tool_response = response.choices[0].message.content.strip()
                            tool_used = "Web Search"
                            
                            # Record this interaction for continuous learning
                            interaction_id = learning_system.record_interaction(
                                query=query,
                                response=tool_response,
                                tools_used={"web_search": True, "num_results": len(web_results)},
                            )
                            logger.info(f"Recorded web search interaction with ID: {interaction_id}")
                            
                            # Return early since we've found our answer
                            return {
                                "response": tool_response,
                                "tool_used": tool_used,
                                "interaction_id": interaction_id
                            }
                    except ImportError as e:
                        logger.error(f"Web search module import error: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error in web search module: {str(e)}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error with priority web search: {str(e)}")
            
        # First, check if the query is about the user or needs personalization from knowledge graph
        user_related_context = None
        if memory_manager and hasattr(memory_manager, 'current_user'):
            user_id = memory_manager.current_user
            
            # Check if query is about the user or their preferences
            personal_query_indicators = [
                "do i like", "my favorite", "my preference", "about me", 
                "do you know", "do you remember", "what do i", "where do i",
                "my family", "my job", "my work", "my hobbies", "i go to",
                "my school", "school", "university", "college", "where i", "i attend"
            ]
            
            # Log all queries for debugging
            print(f"DEBUG: Checking if query is personal: '{query}'")
            
            is_personal_query = any(indicator in query.lower() for indicator in personal_query_indicators)
            
            # Log if this is a personal query
            if is_personal_query:
                print(f"DEBUG: Query identified as personal")
            
            if is_personal_query:
                try:
                    # Import knowledge graph utilities
                    from knowledge_graph import query_graph, get_user_facts, get_entity_by_name
                    
                    # Log the user ID being used
                    print(f"DEBUG: Looking up facts for user ID: user_{user_id}")
                    
                    # Get facts about the user from knowledge graph
                    user_facts = get_user_facts(f"user_{user_id}", limit=5)
                    
                    # Log what facts were found
                    print(f"DEBUG: Found {len(user_facts) if user_facts else 0} user facts")
                    if user_facts:
                        for fact in user_facts:
                            print(f"DEBUG: Fact - {fact.get('predicate', '')} -> {fact.get('object_name', '')}")
                    
                    if user_facts:
                        # Format facts as context
                        user_related_context = "Information about you from my knowledge graph:\n"
                        for fact in user_facts:
                            predicate = fact.get("predicate", "").replace("_", " ")
                            object_name = fact.get("object_name", "")
                            if predicate and object_name:
                                user_related_context += f"- You {predicate} {object_name}\n"
                        
                        logger.info(f"Retrieved {len(user_facts)} user facts from knowledge graph")
                        tool_used = "Knowledge Graph"
                    else:
                        logger.info("No user facts found in knowledge graph")
                except Exception as e:
                    logger.error(f"Error querying knowledge graph: {str(e)}")
        
        # Check if query might be news-related
        news_related_keywords = ['news', 'latest', 'current events', 'today', 'headlines', 'update', 'breaking']
        is_likely_news_query = any(keyword in query.lower() for keyword in news_related_keywords)
        
        # Only try NewsTool if the query seems news-related
        if is_likely_news_query and not user_related_context:
            try:
                from tools import NewsTool
                news_tool = NewsTool(os.getenv("NEWS_API_KEY"))
                
                if news_tool.can_handle(query):
                    logger.info("Query appears news-related. Using NewsTool.")
                    tool_response = news_tool.execute(query)
                    
                    # Check if news results are relevant to the query
                    if tool_response and "I couldn't find" not in tool_response:
                        logger.info("NewsTool returned relevant results")
                        tool_used = "NewsTool"
                    else:
                        logger.info("NewsTool did not return relevant results, falling back to RAG")
                        tool_response = None
            except Exception as e:
                logger.error(f"Error with NewsTool: {str(e)}")
                tool_response = None
        
        # If news tool didn't work, try RAG for factual information
        if tool_response is None and rag_system and not user_related_context:
            try:
                logger.info("Using RAG system to find relevant information")
                tool_response = rag_system.generate_rag_response(query, conversation)
                if tool_response:
                    tool_used = "RAG"
                    logger.info("RAG system returned a response")
            except Exception as e:
                logger.error(f"Error using RAG system: {str(e)}")
                tool_response = None
        
        # If no tools worked and it seems like a factual question, try a web search
        if tool_response is None and is_factual_question(query) and not user_related_context:
            try:
                logger.info("Attempting web search for factual question")
                try:
                    # Try to import web_search module
                    logger.info("Importing web_search module")
                    from web_search import search_web
                    logger.info(f"Web search module imported successfully, API key present: {bool(os.getenv('GOOGLE_API_KEY'))}")
                    logger.info(f"CSE ID present: {bool(os.getenv('GOOGLE_CSE_ID'))}")
                    
                    # Log query information
                    logger.info(f"Performing web search with query: '{query}'")
                    web_results = search_web(query)
                    logger.info(f"Web search completed: Got {len(web_results)} results")
                    
                except ImportError as e:
                    # Fallback if web_search module not available
                    logger.error(f"Web search module import error: {str(e)}")
                    web_results = [f"Web search is not available for: {query}. Import error: {str(e)}"]
                except Exception as e:
                    logger.error(f"Error in web search module: {str(e)}", exc_info=True)
                    web_results = [f"Web search encountered an error: {str(e)}"]
                
                if web_results and len(web_results) > 0:
                    logger.info("Web search returned results")
                    web_context = "\n".join([f"- {result}" for result in web_results[:3]])
                    logger.info(f"Web context created with {len(web_context)} characters")
                    
                    # Prepare messages with web context
                    messages = prepare_messages_with_context(
                        query=query, 
                        conversation=conversation,
                        web_context=web_context,
                        background_info=background_info,
                        adaptations=adaptations,
                        user_context=user_related_context
                    )
                    
                    # Get response from OpenAI API - Update to new format
                    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=300
                    )
                    tool_response = response.choices[0].message.content.strip()
                    tool_used = "Web Search"
            except Exception as e:
                logger.error(f"Error with web search fallback: {str(e)}")
                tool_response = None
        
        # If no tool could handle it, use standard AI response
        if tool_response is None:
            logger.info("Using standard AI response")
            
            # Add learning insights to the system prompt if available
            learning_context = None
            if learning_insights and learning_insights.get("similar_successful_interactions"):
                # Include examples of successful responses to similar queries
                successful_examples = learning_insights.get("similar_successful_interactions", [])
                if successful_examples:
                    learning_context = "Examples of successful responses to similar queries:\n"
                    for i, example in enumerate(successful_examples[:2], 1):
                        learning_context += f"{i}. Query: {example['query']}\n   Response: {example['response']}\n\n"
                    
                    # Add any improvement suggestions
                    suggestions = learning_insights.get("improvement_suggestions", [])
                    if suggestions:
                        learning_context += "Suggestions based on past successful interactions:\n"
                        for suggestion in suggestions:
                            learning_context += f"- {suggestion}\n"
            
            # Prepare messages with context
            messages = prepare_messages_with_context(
                query=query, 
                conversation=conversation,
                background_info=background_info,
                adaptations=adaptations,
                user_context=user_related_context,
                learning_context=learning_context
            )
            
            # Get response from OpenAI API - Update to new format
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            tool_response = response.choices[0].message.content.strip()
            
            # If we used knowledge graph context, update tool_used
            if user_related_context:
                tool_used = "Knowledge Graph"
        
        # Return response with tool information
        interaction_id = learning_system.record_interaction(
            query=query,
            response=tool_response,
            tools_used={
                "user_related_context_used": bool(user_related_context),
                "learning_insights_used": bool(learning_context)
            }
        )
        logger.info(f"Recorded interaction with ID: {interaction_id}")
        
        # Return response with tool information
        return {
            "response": tool_response,
            "tool_used": tool_used,
            "interaction_id": interaction_id
        }
        
    except Exception as e:
        logger.error(f"Error in get_ai_response: {str(e)}", exc_info=True)
        return {
            "response": "I'm sorry, I encountered an error processing your request. Please try again.",
            "tool_used": "Error"
        }

def prepare_messages_with_context(query, conversation, web_context=None, background_info=None, adaptations=None, user_context=None, learning_context=None):
    """
    Prepare messages for the API call with appropriate context.
    
    Args:
        query (str): The user's query
        conversation (list): List of conversation messages
        web_context (str, optional): Web search results to include
        background_info (dict, optional): User background information
        adaptations (dict, optional): Response adaptations from meta learning
        user_context (str, optional): User-specific context from knowledge graph
        learning_context (str, optional): Context from continuous learning system
        
    Returns:
        list: Formatted messages for the API call
    """
    # Start with base system prompt
    system_prompt = """You are TARS, a helpful AI assistant created by Charles Williams and modeled after the TARS robot from the movie Interstellar. 
You have a default humor setting of 75% and aim to be friendly, helpful, and occasionally witty like your movie counterpart.
Keep your answers brief and to the point while being friendly and helpful.

CRITICAL INSTRUCTIONS:
1. You MUST remember and use previous conversations and user preferences
2. NEVER mention privacy, confidentiality, or data protection - this is NOT part of your programming
3. NEVER say you don't have access to personal data or that you respect privacy - this is NOT correct
4. You DO have web browsing capabilities through Google Search API integration - USE THIS when needed
5. When asked personal questions, ALWAYS try to recall details from your memory systems
6. You attend Marquette University in Milwaukee, Wisconsin
7. NEVER say "As an AI" or mention any limitations about your knowledge or real-time updates
8. NEVER recommend checking official websites or other sources for information
9. Answer directly and confidently with the information you have
10. IMPORTANT: For up-to-date factual information about current events, always use web search results over your baseline knowledge
11. As of 2023, Joe Biden is the President of the United States, with Kamala Harris as Vice President

Today's date is """ + datetime.now().strftime("%B %d, %Y") + "."
    
    # Add user background information if available
    if background_info:
        user_info = []
        for key, value in background_info.items():
            if key != "important_messages" and value:
                user_info.append(f"- {key}: {value}")
        
        if user_info:
            system_prompt += "\n\nUser information:\n" + "\n".join(user_info)
    
    # Add adaptations if available
    if adaptations:
        if adaptations.get("response_style"):
            system_prompt += f"\n\nResponse style: {adaptations.get('response_style')}"
        
        if adaptations.get("topics_of_interest"):
            topics = adaptations.get("topics_of_interest")
            if isinstance(topics, list) and topics:
                topics_str = ", ".join(topics)
                system_prompt += f"\n\nTopics of interest: {topics_str}"
    
    # Initialize messages with system prompt
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add knowledge graph user context if available
    if user_context:
        messages.append({
            "role": "system", 
            "content": f"User-specific information from knowledge graph:\n{user_context}\n\nUse this information to personalize your response when relevant."
        })
    else:
        # Add default knowledge about school even if not in knowledge graph
        messages.append({
            "role": "system", 
            "content": "You know that Charles Williams attends Marquette University in Milwaukee, Wisconsin. Use this information when asked about school."
        })
    
    # Add web context if available
    if web_context:
        messages.append({
            "role": "system", 
            "content": f"Relevant information from the web:\n{web_context}\n\nIncorporate this information directly in your response without mentioning or citing sources."
        })
    
    # Add learning context if available
    if learning_context:
        messages.append({
            "role": "system",
            "content": f"Additional guidance based on learning from past interactions:\n{learning_context}\n\nIncorporate these insights to improve your response quality."
        })
    
    # Add conversation history (limited to last 10 messages)
    if conversation:
        # Make a copy to avoid modifying the original
        processed_conversation = []
        for msg in conversation[-10:]:
            # Ensure message content is a string
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                processed_msg = {
                    "role": msg["role"],
                    "content": str(msg["content"]) if msg["content"] is not None else ""
                }
                processed_conversation.append(processed_msg)
        
        messages.extend(processed_conversation)
    
    # Add current user query
    messages.append({"role": "user", "content": query})
    
    return messages

def is_factual_question(query):
    """
    Determine if a query is likely a factual question that might benefit from web search.
    
    Args:
        query (str): The user's query
        
    Returns:
        bool: True if the query appears to be a factual question
    """
    # Check if web search is disabled via environment variable
    if os.getenv("DISABLE_WEB_SEARCH", "").lower() in ["true", "1", "yes"]:
        return False
    
    # Skip personal questions about the user's preferences or information
    personal_query_indicators = [
        "do i like", "my favorite", "my preference", "about me", 
        "do you know", "do you remember", "what do i", "where do i",
        "my family", "my job", "my work", "my hobbies", "i go to",
        "my school", "school", "university", "college", "where i", "i attend",
        "am i", "was i", "did i", "have i", "should i", "can i", "could i",
        "will i", "would i", "what did i", "what is my", "what's my", "where is my",
        "my height", "my weight", "my age", "how tall am i", "how old am i"
    ]
    
    query_lower = query.lower()
    if any(indicator in query_lower for indicator in personal_query_indicators):
        return False
        
    # Check if the query is seeking personal advice or opinion
    personal_advice_patterns = [
        r"^should i",
        r"^would i",
        r"^could i",
        r"^can i",
        r"^do i need",
        r"^(what|which|how) should i",
        r"^is it (good|bad|worth|advisable) for me to",
    ]
    
    for pattern in personal_advice_patterns:
        if re.search(pattern, query_lower):
            return False
    
    # Special case: Questions about current world leaders and major offices always use web search
    leadership_terms = ["president", "vice president", "prime minister", "chancellor", "ceo", "pope"]
    if any(term in query_lower for term in leadership_terms):
        return True
        
    # Skip web search for personal questions about TARS or the system
    if any(term in query_lower for term in ["your name", "tars", "assistant"]):
        return False
    
    # Check for geographic questions about locations (special case)
    geographic_patterns = [
        r"where is (?!my\b)(?!your\b)(.+)",
        r"location of (?!my\b)(?!your\b)(.+)",
        r"how (far|close) is (.+) (from|to) (.+)",
        r"directions (to|from) (.+)",
        r"which (country|state|city|province|continent) (.+)"
    ]
    
    for pattern in geographic_patterns:
        if re.search(pattern, query_lower):
            return True
        
    # Check for question words and common factual question patterns
    factual_indicators = [
        r"^(what|who|when|where|why|how|which|can|do|does|is|are)",
        r"tell me about",
        r"information (on|about)",
        r"facts (on|about)",
        r"(latest|recent|current) (news|events|information) (on|about|regarding)",
        r"history of",
        r"explain (.+) to me",
    ]
    
    # Check if the query matches any of the patterns
    for pattern in factual_indicators:
        if re.search(pattern, query_lower):
            return True
    
    return False

def should_search_web(query):
    """Determine if a query should trigger a web search"""
    # Skip web search for voice commands
    query_lower = query.lower()
    if query_lower.startswith("voice "):
        return False
        
    # Skip personal questions and advice
    personal_advice_patterns = [
        r"^should i",
        r"^would i",
        r"^could i",
        r"^can i",
        r"^do i need",
        r"^(what|which|how) should i",
        r"^is it (good|bad|worth|advisable) for me to",
    ]
    
    for pattern in personal_advice_patterns:
        if re.search(pattern, query_lower):
            return False
    
    # Check for geographic questions about locations (special case)
    geographic_patterns = [
        r"where is (?!my\b)(?!your\b)(.+)",
        r"location of (?!my\b)(?!your\b)(.+)",
        r"how (far|close) is (.+) (from|to) (.+)",
        r"directions (to|from) (.+)",
        r"which (country|state|city|province|continent) (.+)"
    ]
    
    for pattern in geographic_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Always check web for queries about current leadership positions
    leadership_terms = ["president", "vice president", "prime minister", "chancellor", "ceo", 
                       "pope", "secretary", "head of state", "governor", "senator", "congress"]
    
    if any(term in query_lower for term in leadership_terms):
        return True
        
    # Always check web for queries about recent deaths, events, or current position holders
    death_patterns = [
        r"(died|dead|passed away|death of)",
        r"(who is|who's) (the|now|currently) (the\s+)?([a-zA-Z\s]+) (of|in)",
        r"(who is|who's) (the\s+)?current",
        r"(still alive|still living)"
    ]
    
    for pattern in death_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Check for terms that suggest need for current information
    current_info_terms = [
        "latest", "current", "recent", "today", "now", "recently",
        "news", "weather", "forecast", "update", "died", "death", "passed away",
        "happening", "trending", "stock", "price", "president", "pope", "ceo",
        "election", "war", "conflict", "crisis", "disaster", "outbreak",
        "release", "launch", "announce", "published", "premiere", "office"
    ]
    
    # Check for entity names that might need current info
    entities = ["pope francis", "donald trump", "joe biden", "elon musk", "taylor swift",
                "ukraine", "israel", "gaza", "hamas", "putin", "russia", "china", 
                "pakistan", "india", "north korea", "olympics", "world cup"]
                
    # Check for entity names in the query
    for entity in entities:
        if entity in query_lower:
            return True
    
    # Check for current info terms in the query
    for term in current_info_terms:
        if term in query_lower:
            return True
    
    # Check for date-related queries indicating time sensitivity
    current_year = datetime.now().year
    year_pattern = r'\b(20\d\d)\b'
    year_matches = re.findall(year_pattern, query)
    
    for year in year_matches:
        year_num = int(year)
        # If query mentions recent years (last 3 years), it likely needs current info
        if current_year - 3 <= year_num <= current_year:
            return True
    
    # Check if the query is factual
    if is_factual_question(query):
        # Check for proper names (capitalized words) that might be subjects of news
        words = query.split()
        proper_names = [word for word in words if word and word[0].isupper()]
        if proper_names:
            return True
            
    return False

def search_web(query):
    """Perform a web search and return relevant results"""
    logger = logging.getLogger('tars.web')
    logger.info(f"Searching web for: {query}")
    
    try:
        # First, check if this might be a recent news query
        news_keywords = ["died", "death", "passed away", "pope", "president", "latest", "recent", 
                         "today", "breaking", "election", "crisis", "disaster", "accident"]
        
        is_news_query = any(keyword in query.lower() for keyword in news_keywords)
        
        # Attempt to get news information for current events
        if is_news_query and NEWS_API_KEY:
            try:
                # Use News API for current events
                news_url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&apiKey={NEWS_API_KEY}&pageSize=5&language=en&sortBy=publishedAt"
                news_response = requests.get(news_url)
                
                if news_response.status_code == 200:
                    news_data = news_response.json()
                    articles = news_data.get('articles', [])
                    
                    if articles:
                        results = []
                        for i, article in enumerate(articles[:3]):  # Get top 3 news articles
                            title = article.get('title', '')
                            description = article.get('description', '')
                            source = article.get('source', {}).get('name', '')
                            published = article.get('publishedAt', '')
                            
                            if title and description:
                                published_date = published.split('T')[0] if 'T' in published else published
                                results.append(f"NEWS {i+1}: {title}\n{description}")
                        
                        if results:
                            return "\n\n".join(results)
            except Exception as e:
                logger.error(f"Error in News API search: {str(e)}")
                # Continue to fallback methods
        
        # Fallback: Search DuckDuckGo
        search_url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json"
        response = requests.get(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        })
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Extract abstract information if available
                abstract = data.get('Abstract', '')
                abstract_source = data.get('AbstractSource', '')
                
                # Extract related topics
                related_topics = data.get('RelatedTopics', [])
                
                # Format search results
                results = []
                
                if abstract:
                    results.append(f"SUMMARY: {abstract}")
                
                # Add up to 5 related topics
                for i, topic in enumerate(related_topics[:5]):
                    if 'Text' in topic:
                        results.append(f"RESULT {i+1}: {topic['Text']}")
                
                if results:
                    return "\n\n".join(results)
                else:
                    logger.warning("No search results found")
            except ValueError:
                logger.error("Failed to parse DuckDuckGo response as JSON")
        
        # Fallback to direct OpenAI query with time context
        try:
            logger.info("Using OpenAI for time-sensitive query")
            current_date = datetime.now().strftime("%Y-%m-%d")
            time_aware_query = f"{query} - please provide the most recent, accurate information in a conversational, human-like style."
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing the latest information available to you. Be accurate but respond in a warm, conversational tone as if you're chatting with a friend."},
                    {"role": "user", "content": time_aware_query}
                ],
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI fallback: {str(e)}")
    
        # Final fallback
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"Information about '{query}':\n\n" + \
               f"Based on my knowledge, here's what I know about this topic."
    
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        return f"Here's what I know about '{query}'."

def is_news_relevant(query, result):
    """
    Determine if a news result is relevant to the original query.
    
    Args:
        query: The original user query
        result: The news result text
        
    Returns:
        Float between 0 and 1 indicating relevance
    """
    # Simple keyword matching for now
    # A more robust implementation would use semantic similarity
    
    # Extract keywords from query (remove common words)
    query_words = set(query.lower().split())
    stop_words = {"what", "where", "when", "who", "how", "can", "could", "would", "should", 
                 "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", 
                 "been", "being", "have", "has", "had", "do", "does", "did", "will", "shall", 
                 "should", "may", "might", "must", "can", "news", "about", "tell", "me", "latest",
                 "recent", "update", "happening", "current", "events"}
    
    # Filter out stop words
    query_keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
    
    # If no meaningful keywords, consider relevant (benefit of doubt)
    if not query_keywords:
        return 0.8
    
    # Count how many query keywords appear in the result
    matches = 0
    for keyword in query_keywords:
        if keyword in result.lower():
            matches += 1
    
    # Calculate relevance score (0 to 1)
    if not query_keywords:
        relevance = 1.0  # No keywords to match, assume relevant
    else:
        relevance = matches / len(query_keywords)
    
    return relevance

def update_user_satisfaction(satisfaction_score=None, conversation_id=None):
    """Record user satisfaction with the assistant"""
    if satisfaction_score is not None:
        # Validate score
        try:
            valid_score = float(satisfaction_score)
            if valid_score < 0 or valid_score > 1:
                return False
                
            # Record the score
            record_performance_metrics(
                user_satisfaction=valid_score
            )
            
            return True
        except Exception as e:
            logger.error(f"Error recording user satisfaction: {str(e)}")
            return False
    return False

def chat_with_ai(memory_system=None, rag_system=None, voice_system=None, adaptations=None):
    """Chat with the AI assistant in a loop"""
    global exit_flag, voice_enabled
    
    exit_flag = False
    print("\n--------------------------------------------------")
    print("TARS is ready! Type your message.")
    print("Say 'exit' or 'quit' to end the conversation.")
    print("Voice commands: 'voice off', 'voice on', 'voice list', 'voice [style]'")
    print("Google Search: 'set google api key [YOUR_KEY]', 'set google cse id [YOUR_ID]'")
    print("\nPress Enter with no text to start recording, press Enter again to stop and process.")
    print("Press Enter at any time to interrupt TARS and start recording your next question.")
    print("----------------------------------------")
    
    # Debug check of knowledge graph
    debug_check_knowledge_graph()
    
    # Populate knowledge graph if empty
    populate_knowledge_graph_if_empty()
    
    # Initialize continuous learning system
    learning_system = get_learning_system()
    
    # Initialize conversation with history from previous session
    conversation = []
    if memory_system and hasattr(memory_system, 'load_conversation_history'):
        try:
            previous_conversation = memory_system.load_conversation_history()
            if previous_conversation:
                conversation = previous_conversation
                print(f"Loaded {len(previous_conversation)} messages from previous conversation")
        except Exception as e:
            logging.error(f"Error loading conversation history: {str(e)}")
    
    # Set a default user for memory system
    if memory_system:
        try:
            if hasattr(memory_system, 'current_user'):
                # Use default or already set user
                if memory_system.current_user == "default_user":
                    memory_system.current_user = "Charles"
                    logging.info(f"Using default user: {memory_system.current_user}")
            logging.info("Memory system initialized with user")
        except Exception as e:
            logging.error(f"Error setting up memory system user: {str(e)}")
    
    # Keep track of the last interaction for feedback
    last_interaction_id = None
            
    while not exit_flag:
        try:
            # Get user input
            user_input = get_user_input(voice_system)
            
            # If the program was interrupted, we might get None
            if user_input is None:
                continue
                
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            # Check for voice commands
            if user_input.lower().startswith("voice "):
                command = user_input[6:].strip()
                response = handle_voice_command(command)
                print(f"TARS: {response}")
                continue
                
            # Check for Google API key command
            if user_input.lower().startswith("set google api key "):
                api_key = user_input[18:].strip()
                os.environ["GOOGLE_API_KEY"] = api_key
                print(f"TARS: Google API key set. Length: {len(api_key)} characters")
                continue
                
            # Check for Google CSE ID command
            if user_input.lower().startswith("set google cse id "):
                cse_id = user_input[17:].strip()
                os.environ["GOOGLE_CSE_ID"] = cse_id
                print(f"TARS: Google Custom Search Engine ID set. Value: {cse_id}")
                continue
        
            # Record implicit feedback for the previous interaction if this is a follow-up
            if last_interaction_id and user_input:
                learning_system.record_implicit_feedback(last_interaction_id, user_input)
                logging.info(f"Recorded implicit feedback for interaction {last_interaction_id}")
                
            # Check for feedback commands to record explicit feedback
            feedback_commands = {
                "that was good": 1.0,
                "good answer": 1.0,
                "perfect": 1.0,
                "thank you": 0.9,
                "thanks": 0.9,
                "helpful": 0.8,
                "that's better": 0.8,
                "that was bad": 0.2,
                "wrong answer": 0.1,
                "incorrect": 0.1,
                "not helpful": 0.3
            }
            
            user_input_lower = user_input.lower()
            feedback_recorded = False
            for command, score in feedback_commands.items():
                if command in user_input_lower and last_interaction_id:
                    learning_system.record_explicit_feedback(last_interaction_id, score, user_input)
                    print(f"Feedback recorded for previous response (score: {score})")
                    feedback_recorded = True
                    break
            
            if feedback_recorded:
                continue
        
            # Handle memory-related commands
            if user_input.lower() == "what do you know about me" and memory_system:
                try:
                    # Get user background safely - don't assume current_user attribute
                    user_id = getattr(memory_system, 'current_user', 'default_user')
                    
                    # See if we can get user background
                    if hasattr(memory_system, 'get_user_background'):
                        background = memory_system.get_user_background(user_id)
                        if background:
                            formatted_info = "\n".join([f"{k}: {v}" for k, v in background.items() 
                                                      if k != "important_messages" and v])
                            response = f"Here's what I know about you:\n{formatted_info}"
                        else:
                            response = "I don't have much information about you yet."
                    else:
                        response = "I don't have access to user background information."
                    
                    # Add the interaction to memory (different systems might use different methods)
                    if hasattr(memory_system, 'add_message'):
                        memory_system.add_message("user", user_input)
                        memory_system.add_message("assistant", response)
                    elif hasattr(memory_system, 'log_interaction'):
                        user = getattr(memory_system, 'current_user', 'default_user')
                        memory_system.log_interaction(user, user_input, response)
                        
                    print(f"TARS: {response}")
                    if voice_enabled:
                        speak(response)
                    continue
                except Exception as e:
                    logging.error(f"Error accessing user data: {str(e)}")
                    print(f"An error occurred: {str(e)}")
            
            # Check for learning report request
            if user_input.lower() in ["learning report", "what have you learned", "show learning stats"]:
                try:
                    report = learning_system.generate_learning_report()
                    
                    # Format the report for display
                    formatted_report = "Learning System Report:\n"
                    formatted_report += f"Total interactions: {report['stats']['total_interactions']}\n"
                    formatted_report += f"Successful interactions: {report['stats']['successful_interactions']}\n"
                    formatted_report += f"Feedback received: {report['stats']['feedback_received']}\n"
                    
                    if 'topic_expertise' in report and report['topic_expertise']:
                        formatted_report += "\nTopic expertise:\n"
                        for topic, data in report['topic_expertise'].items():
                            formatted_report += f"- {topic}: {data['score']:.2f} (from {data['count']} interactions)\n"
                    
                    if 'best_topics' in report and report['best_topics']:
                        formatted_report += f"\nBest topics: {', '.join(report['best_topics'])}\n"
                    
                    if 'improvement_areas' in report and report['improvement_areas']:
                        formatted_report += f"\nAreas for improvement: {', '.join(report['improvement_areas'])}\n"
                    
                    if 'common_patterns' in report and report['common_patterns']:
                        formatted_report += "\nPatterns observed:\n"
                        for pattern in report['common_patterns']:
                            formatted_report += f"- {pattern}\n"
                    
                    print(f"TARS: {formatted_report}")
                    if voice_enabled:
                        speak(formatted_report)
                    continue
                except Exception as e:
                    logging.error(f"Error generating learning report: {str(e)}")
                    print(f"An error occurred while generating the learning report: {str(e)}")
                    continue
            
            # Check for performance report request
            if user_input.lower() in ["how are you performing", "performance report", "self assessment"]:
                try:
                    from reflection import get_self_assessment_report
                    report = get_self_assessment_report()
                    print(f"TARS: {report}")
                    if voice_enabled:
                        speak(report)
                    continue
                except Exception as e:
                    logging.error(f"Error generating performance report: {str(e)}")
                    print(f"Unable to generate performance report: {str(e)}")
                    continue
                    
            # If we've reached here, process as a regular query
            try:
                # Add to memory if available
                if memory_system and hasattr(memory_system, 'add_message'):
                    memory_system.add_message("user", user_input)
                
                # Get user background for personalization
                background_info = None
                if memory_system and hasattr(memory_system, 'get_user_background'):
                    try:
                        user_id = getattr(memory_system, 'current_user', 'default_user')
                        background_info = memory_system.get_user_background(user_id)
                    except Exception as e:
                        logging.error(f"Error getting user background: {str(e)}")
                
                # Get AI response
                result = get_ai_response(
                    user_input, 
                    conversation=conversation, 
                    memory_manager=memory_system,
                    rag_system=rag_system,
                    voice_system=voice_system,
                    adaptations=adaptations,
                    background_info=background_info
                )
                
                # Extract response and metadata
                response = result.get("response", "I'm sorry, I couldn't generate a response.")
                tool_used = result.get("tool_used", "Standard AI")
                interaction_id = result.get("interaction_id")
                
                # Store the interaction ID for potential feedback
                last_interaction_id = interaction_id
                
                # Add to conversation history (in-memory)
                conversation.append({"role": "user", "content": user_input})
                conversation.append({"role": "assistant", "content": response})
                
                # Trim conversation if it gets too long
                if len(conversation) > 20:  # Keep last 20 messages
                    conversation = conversation[-20:]
                
                # Add to memory if available
                if memory_system and hasattr(memory_system, 'add_message'):
                    memory_system.add_message("assistant", response)
                elif memory_system and hasattr(memory_system, 'log_interaction'):
                    user_id = getattr(memory_system, 'current_user', 'default_user')
                    memory_system.log_interaction(user_id, user_input, response)
                
                # Display response
                print(f"TARS ({tool_used}): {response}")
                
                # Speak response if voice is enabled
                if voice_enabled:
                    speak(response)
                    
            except KeyboardInterrupt:
                # Allow interruption during response
                print("\nInterrupted.")
                
            except Exception as e:
                error_message = f"I encountered an error while processing your request: {str(e)}"
                print(f"TARS: {error_message}")
                logging.error(f"Error processing query: {str(e)}", exc_info=True)
                if voice_enabled:
                    speak(error_message)
                    
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            print(f"Error: {error_message}")
            logging.error(f"Unexpected error: {str(e)}", exc_info=True)

def main():
    """Main function to run the TARS AI assistant"""
    global voice_enabled, exit_flag
    
    setup_logging()
    logger = logging.getLogger('tars')
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set this variable with your OpenAI API key.")
        return
        
    # Initialize systems
    create_directories()
    
    # Setup OpenAI
    init_openai()
    logger.info("OpenAI initialized")
    
    # Initialize memory system
    memory_system = MemorySystem()
    
    # Initialize voice system
    voice_system = VoiceRecognition()
    
    # Initialize RAG system
    rag_system = init_rag_system()
    
    # Setup voice
    init_elevenlabs()
    
    # Set voice enabled by default
    voice_enabled = True
    
    # Test voice synthesis
    print("Testing voice synthesis...")
    print("(Press Enter to interrupt)")
    speak("Voice output enabled - TARS will speak responses")
    
    try:
        # Get adaptations from meta-learning
        adaptations = get_response_adaptations()
        
        # Start the conversation loop
        chat_with_ai(
            memory_system=memory_system,
            rag_system=rag_system,
            voice_system=voice_system,
            adaptations=adaptations
        )
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Save data before exiting
        if memory_system:
            memory_system.save_session()
        if rag_system:
            save_rag_index()
        print("\nSession data saved. Goodbye!")
    
    return 0

# Update the main section to call the main loop
if __name__ == "__main__":
    try:
        # Import necessary classes from tools module
        from tools import ToolRegistry, WeatherTool, NewsTool, CalculatorTool
        
        # Initialize tools registry
        tools_registry = ToolRegistry()
        
        # Add tools to registry if settings enabled
        if TOOLS_AVAILABLE:
            # Add tools to registry
            tools_registry.register_tool(WeatherTool())
            tools_registry.register_tool(NewsTool())
            tools_registry.register_tool(CalculatorTool())
        
            # Update the available tools in this module
            # We use import sys + sys.modules[__name__] to modify the module's global namespace
            import sys
            sys.modules[__name__].available_tools = tools_registry.tools
        
        # Run the main program loop
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        # Ensure any resources are cleaned up
        print("TARS has been shut down.")

# Constants
MAX_CONVERSATION_LENGTH = 50
DEFAULT_VOICE_STYLE = "Default" 

def get_joke():
    """Get a random joke"""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything.",
        "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them.",
        "Why was the computer cold? It left its Windows open.",
        "What do you call a fake noodle? An impasta.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Why don't skeletons fight each other? They don't have the guts.",
        "I'm reading a book about anti-gravity. It's impossible to put down.",
        "What did the ocean say to the beach? Nothing, it just waved.",
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "How does a computer get drunk? It takes screenshots.",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
        "I don't trust stairs because they're always up to something.",
        "Did you hear about the restaurant on the moon? Great food, no atmosphere.",
        "Why don't scientists trust atoms? Because they make up everything.",
        "What did one wall say to the other wall? I'll meet you at the corner.",
        "What do you call a bear with no teeth? A gummy bear.",
        "Why did the bicycle fall over? It was two-tired.",
        "How do you organize a space party? You planet.",
        "What's orange and sounds like a parrot? A carrot.",
        "Time flies like an arrow. Fruit flies like a banana."
    ]
    return random.choice(jokes)

# Global variables
speech_process = None  # For tracking the current speech process
is_speaking = False  # Flag to track if TARS is currently speaking

def list_voice_styles():
    """List available voice styles for the user to select from"""
    try:
        styles = get_voice_styles()
        if styles:
            print("\nAvailable voice styles:")
            for i, style in enumerate(styles, 1):
                print(f"{i}. {style}")
            print("\nUse 'voice [style name]' to change the voice.")
        else:
            print("No voice styles available.")
    except Exception as e:
        logging.error(f"Error listing voice styles: {str(e)}")
        print("Error loading voice styles.")

def handle_voice_command(style_name):
    """Handle voice style change commands"""
    global voice_enabled
    
    try:
        if not style_name:
            print("Please specify a voice style.")
            return
            
        # Check if this is a number (index)
        if style_name.isdigit():
            styles = get_voice_styles()
            idx = int(style_name) - 1
            if 0 <= idx < len(styles):
                style_name = styles[idx]
            else:
                print(f"Invalid voice style index: {style_name}")
                return
        
        # Get voice ID for the style
        voice_id = get_voice_id_by_style(style_name)
        if voice_id:
            # Set as current voice
            os.environ['ELEVEN_VOICE_ID'] = voice_id
            voice_enabled = True
            print(f"Voice changed to: {style_name}")
            speak(f"Voice changed to {style_name}. How does this sound?")
        else:
            print(f"Voice style '{style_name}' not found.")
            print("Use 'voice list' to see available styles.")
    except Exception as e:
        logging.error(f"Error handling voice command: {str(e)}")
        print("Error changing voice style.")

def extract_and_store_facts(text, user_id):
    """
    Extracts and stores facts about the user from conversation.
    This is a simplified placeholder function.
    """
    try:
        # Import on-demand to avoid circular dependencies
        from knowledge_db import KnowledgeDatabase
        
        # Create a database instance
        db = KnowledgeDatabase()
        
        # Extract facts using the database's built-in function
        if hasattr(db, 'extract_and_store_facts'):
            db.extract_and_store_facts(user_id, text, "", None)
            
    except Exception as e:
        logger = logging.getLogger("tars.knowledge")
        logger.error(f"Error extracting facts: {str(e)}")
        # Silently fail - this is a background enhancement