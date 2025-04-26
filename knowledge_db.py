"""
Knowledge Database and Enhanced Memory System for TARS

This module provides:
1. KnowledgeDatabase - For storing and retrieving knowledge entries
2. EnhancedMemory - For managing user-specific memories and contextual retrieval

The knowledge database stores information from conversations and searches for future reference.
The enhanced memory system improves context awareness and personalization.
"""

import os
import json
import sqlite3
import datetime
import logging
import numpy as np
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Change from INFO to WARNING to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_db.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('knowledge_db')

# Try to import OpenAI for embeddings, with fallback to sklearn
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI available for vector embeddings")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available, falling back to TF-IDF")
    
    # Fallback to sklearn for vector embeddings
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        SKLEARN_AVAILABLE = True
        logger.info("Using sklearn TF-IDF for vector embeddings")
    except ImportError:
        SKLEARN_AVAILABLE = False
        logger.warning("Neither OpenAI nor sklearn available - vector search disabled")

class KnowledgeDatabase:
    """
    Persistent knowledge storage with vector search capabilities.
    
    This database stores knowledge extracted from conversations and web searches.
    It supports both keyword and semantic search for efficient retrieval.
    """
    
    def __init__(self, db_path: str = "memory/knowledge.db", vector_search: bool = True):
        """
        Initialize the knowledge database.
        
        Args:
            db_path: Path to the SQLite database file
            vector_search: Whether to enable vector search (requires OpenAI API or sklearn)
        """
        self.db_path = db_path
        self.conn = None
        self.vector_search_enabled = False
        self.embeddings = {}
        self.embeddings_path = os.path.join(os.path.dirname(db_path), "embeddings.json")
        self.vectorizer = None
        self.openai_client = None
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Initialize vector search if requested
        if vector_search:
            self.init_vector_search()
            
    def init_database(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create knowledge entries table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                timestamp TIMESTAMP NOT NULL,
                embedding_id TEXT,
                confidence REAL DEFAULT 1.0,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
            ''')
            
            # Create topics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                knowledge_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_entries(id)
            )
            ''')
            
            # Create user facts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_facts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                source TEXT,
                timestamp TIMESTAMP NOT NULL,
                confidence REAL DEFAULT 1.0
            )
            ''')
            
            # Create metadata table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            ''')
            
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None
    
    def init_vector_search(self):
        """Initialize vector search capabilities using OpenAI or TF-IDF."""
        if not self.conn:
            logger.error("Cannot initialize vector search: Database not connected")
            return False
        
        try:
            # Try to use OpenAI embeddings if available
            if OPENAI_AVAILABLE:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    try:
                        self.openai_client = OpenAI(api_key=api_key)
                        self.vector_search_enabled = True
                        logger.info("Vector search initialized with OpenAI embeddings")
                        self.load_embeddings()
                        return True
                    except Exception as e:
                        logger.error(f"Error initializing OpenAI: {e}")
            
            # Fallback to TF-IDF vectorization with sklearn
            if SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(max_features=100)
                self.vector_search_enabled = True
                logger.info("Vector search initialized with TF-IDF (sklearn)")
                self.load_embeddings()
                return True
            
            logger.warning("Vector search could not be initialized")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing vector search: {e}")
            return False
    
    def load_embeddings(self):
        """Load existing embeddings from file."""
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, 'r') as f:
                    # Load embeddings as dictionary mapping entry id to embedding
                    raw_data = json.load(f)
                    self.embeddings = {}
                    for entry_id, embedding_data in raw_data.items():
                        self.embeddings[entry_id] = np.array(embedding_data)
                logger.info(f"Loaded {len(self.embeddings)} embeddings from file")
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
                self.embeddings = {}
    
    def save_embeddings(self):
        """Save embeddings to file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            embeddings_json = {}
            for entry_id, embedding in self.embeddings.items():
                # Handle numpy arrays
                if isinstance(embedding, np.ndarray):
                    embeddings_json[entry_id] = embedding.tolist()
                else:
                    embeddings_json[entry_id] = embedding
                    
            with open(self.embeddings_path, 'w') as f:
                json.dump(embeddings_json, f)
            logger.info(f"Saved {len(self.embeddings)} embeddings to file")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI API."""
        if not text or len(text) < 3:
            return None
            
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error getting OpenAI embedding: {e}")
                return None
        elif self.vectorizer:
            # Fallback to TF-IDF
            try:
                vectors = self.vectorizer.fit_transform([text])
                return vectors.toarray()[0]
            except Exception as e:
                logger.error(f"Error getting TF-IDF vector: {e}")
                return None
        else:
            return None
    
    def add_entry(self, content: str, source: Optional[str] = None, 
                 topic: Optional[str] = None, confidence: float = 1.0) -> Optional[str]:
        """
        Add a new knowledge entry to the database.
        
        Args:
            content: The knowledge content text
            source: Source of the knowledge (e.g., "conversation", "web_search")
            topic: Topic category for the knowledge
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            Entry ID if successful, None otherwise
        """
        if not self.conn:
            logger.error("Cannot add entry: Database not connected")
            return None
            
        try:
            # Generate a unique ID
            entry_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            
            # Get embedding if vector search is enabled
            embedding_id = None
            if self.vector_search_enabled and len(content) > 5:
                embedding = self.get_embedding(content)
                if embedding is not None:
                    embedding_id = entry_id
                    self.embeddings[embedding_id] = embedding
            
            # Insert entry
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO knowledge_entries 
            (id, content, source, timestamp, embedding_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (entry_id, content, source, timestamp, embedding_id, confidence))
            
            # Add topic if provided
            if topic:
                topic_id = str(uuid.uuid4())
                cursor.execute('''
                INSERT INTO topics (id, knowledge_id, topic)
                VALUES (?, ?, ?)
                ''', (topic_id, entry_id, topic))
            
            self.conn.commit()
            
            # Save embeddings periodically (every 10 entries)
            if self.vector_search_enabled and len(self.embeddings) % 10 == 0:
                self.save_embeddings()
                
            logger.info(f"Added knowledge entry: {entry_id} (topic: {topic})")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge entry: {e}")
            return None
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for knowledge entries matching the query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            
        Returns:
            List of matching entries with their metadata
        """
        if not self.conn:
            logger.error("Cannot search: Database not connected")
            return []
            
        try:
            results = []
            
            # Use vector search if enabled
            if self.vector_search_enabled and len(query) > 5:
                query_embedding = self.get_embedding(query)
                if query_embedding is not None:
                    # Calculate similarity scores
                    similarities = []
                    for entry_id, embedding in self.embeddings.items():
                        if isinstance(embedding, list):
                            embedding = np.array(embedding)
                        
                        # Skip if shapes don't match (shouldn't happen but just in case)
                        if len(embedding) != len(query_embedding):
                            continue
                            
                        # Cosine similarity
                        similarity = np.dot(embedding, query_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(query_embedding)
                        )
                        similarities.append((entry_id, similarity))
                    
                    # Sort by similarity (descending)
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top entries
                    top_ids = [entry_id for entry_id, _ in similarities[:top_k]]
                    if top_ids:
                        placeholders = ', '.join(['?'] * len(top_ids))
                        cursor = self.conn.cursor()
                        cursor.execute(f'''
                        SELECT e.id, e.content, e.source, e.timestamp, e.confidence, 
                               t.topic
                        FROM knowledge_entries e
                        LEFT JOIN topics t ON e.id = t.knowledge_id
                        WHERE e.id IN ({placeholders})
                        ''', top_ids)
                        
                        rows = cursor.fetchall()
                        for row in rows:
                            results.append({
                                'id': row[0],
                                'content': row[1],
                                'source': row[2],
                                'timestamp': row[3],
                                'confidence': row[4],
                                'topic': row[5]
                            })
                        
                        # Update access stats
                        self.update_access_stats(top_ids)
            
            # If no results from vector search or vector search not enabled,
            # fall back to keyword search
            if not results:
                cursor = self.conn.cursor()
                search_term = f"%{query}%"
                cursor.execute('''
                SELECT e.id, e.content, e.source, e.timestamp, e.confidence, 
                       t.topic
                FROM knowledge_entries e
                LEFT JOIN topics t ON e.id = t.knowledge_id
                WHERE e.content LIKE ?
                ORDER BY e.confidence DESC, e.timestamp DESC
                LIMIT ?
                ''', (search_term, top_k))
                
                rows = cursor.fetchall()
                for row in rows:
                    results.append({
                        'id': row[0],
                        'content': row[1],
                        'source': row[2],
                        'timestamp': row[3],
                        'confidence': row[4],
                        'topic': row[5]
                    })
                
                # Update access stats
                entry_ids = [row[0] for row in rows]
                if entry_ids:
                    self.update_access_stats(entry_ids)
            
            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    def update_access_stats(self, entry_ids: List[str]):
        """Update last accessed time and access count for entries."""
        if not self.conn or not entry_ids:
            return
            
        try:
            timestamp = datetime.datetime.now().isoformat()
            cursor = self.conn.cursor()
            
            for entry_id in entry_ids:
                cursor.execute('''
                UPDATE knowledge_entries
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
                ''', (timestamp, entry_id))
                
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating access stats: {e}")
    
    def add_user_fact(self, user_id: str, fact: str, topic: Optional[str] = None,
                      source: Optional[str] = None, confidence: float = 1.0) -> Optional[str]:
        """
        Add a fact about a specific user.
        
        Args:
            user_id: User identifier
            fact: The fact about the user
            topic: Topic category for the fact (optional)
            source: Source of the fact (e.g., "conversation")
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            Fact ID if successful, None otherwise
        """
        if not self.conn:
            logger.error("Cannot add user fact: Database not connected")
            return None
            
        try:
            # Generate a unique ID
            fact_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            
            # Insert fact - add topic column if it doesn't exist
            cursor = self.conn.cursor()
            
            # Check if topic column exists
            cursor.execute("PRAGMA table_info(user_facts)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if "topic" not in columns:
                # Add topic column
                cursor.execute("ALTER TABLE user_facts ADD COLUMN topic TEXT")
            
            # Insert fact
            cursor.execute('''
            INSERT INTO user_facts 
            (id, user_id, fact, topic, source, timestamp, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (fact_id, user_id, fact, topic, source, timestamp, confidence))
            
            self.conn.commit()
            logger.info(f"Added user fact for {user_id}: {fact}")
            return fact_id
            
        except Exception as e:
            logger.error(f"Error adding user fact: {e}")
            return None
    
    def get_user_facts(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get facts about a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of facts to return
            
        Returns:
            List of facts about the user
        """
        if not self.conn:
            logger.error("Cannot get user facts: Database not connected")
            return []
            
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT id, fact, source, timestamp, confidence
            FROM user_facts
            WHERE user_id = ?
            ORDER BY confidence DESC, timestamp DESC
            LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    'id': row[0],
                    'fact': row[1],
                    'source': row[2],
                    'timestamp': row[3],
                    'confidence': row[4]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting user facts: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge database."""
        if not self.conn:
            logger.error("Cannot get stats: Database not connected")
            return {
                "total_entries": 0,
                "total_topics": 0,
                "total_user_facts": 0
            }
            
        try:
            cursor = self.conn.cursor()
            
            # Get entry count
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            total_entries = cursor.fetchone()[0]
            
            # Get topic count
            cursor.execute("SELECT COUNT(DISTINCT topic) FROM topics")
            total_topics = cursor.fetchone()[0]
            
            # Get user facts count
            cursor.execute("SELECT COUNT(*) FROM user_facts")
            total_user_facts = cursor.fetchone()[0]
            
            # Get user count
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_facts")
            total_users = cursor.fetchone()[0]
            
            return {
                "total_entries": total_entries,
                "total_topics": total_topics,
                "total_user_facts": total_user_facts,
                "total_users": total_users,
                "vector_search_enabled": self.vector_search_enabled,
                "embeddings_count": len(self.embeddings) if self.vector_search_enabled else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_entries": 0,
                "total_topics": 0,
                "total_user_facts": 0,
                "error": str(e)
            }
    
    def close(self):
        """Close database connection and save state."""
        if self.vector_search_enabled:
            self.save_embeddings()
            
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def get_user_facts_by_topic(self, user_id: str, topic: str, limit: int = 10):
        """
        Retrieve user facts related to a specific topic.
        
        Args:
            user_id: The user ID to get facts for
            topic: The topic category to filter by
            limit: Maximum number of facts to return
            
        Returns:
            List of fact dictionaries filtered by topic
        """
        try:
            if not self.conn:
                logger.error("Cannot get user facts by topic: Database not connected")
                return []
                
            cursor = self.conn.cursor()
            
            # First try exact topic match
            cursor.execute(
                "SELECT fact, source, timestamp FROM user_facts WHERE user_id = ? AND topic = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, topic, limit)
            )
            facts = cursor.fetchall()
            
            # If not enough exact matches, try partial topic match
            if len(facts) < limit:
                remaining = limit - len(facts)
                cursor.execute(
                    "SELECT fact, source, timestamp FROM user_facts WHERE user_id = ? AND topic LIKE ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, f"%{topic}%", remaining)
                )
                facts.extend(cursor.fetchall())
                
            # Convert to dictionaries
            result = []
            for fact_data in facts:
                result.append({
                    "fact": fact_data[0],
                    "source": fact_data[1],
                    "timestamp": fact_data[2],
                    "topic": topic
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting user facts by topic: {e}")
            return []
    
    def get_recent_user_facts(self, user_id: str, limit: int = 5):
        """
        Get the most recent facts for a user.
        
        Args:
            user_id: The user ID to get facts for
            limit: Maximum number of facts to return
            
        Returns:
            List of the most recent fact dictionaries
        """
        try:
            if not self.conn:
                logger.error("Cannot get recent user facts: Database not connected")
                return []
                
            cursor = self.conn.cursor()
            
            cursor.execute(
                "SELECT fact, topic, source, timestamp FROM user_facts WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)
            )
            facts = cursor.fetchall()
            
            # Convert to dictionaries
            result = []
            for fact_data in facts:
                result.append({
                    "fact": fact_data[0],
                    "topic": fact_data[1],
                    "source": fact_data[2],
                    "timestamp": fact_data[3]
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting recent user facts: {e}")
            return []

    def extract_and_store_facts(self, user_id: str, user_message: str, ai_response: str, conversation_id: str = None):
        """
        Extract facts from the conversation and store them in the database.
        
        Args:
            user_id: The user's identifier
            user_message: The message from the user
            ai_response: The AI's response
            conversation_id: Optional conversation identifier
        """
        try:
            # Skip fact extraction for short messages, commands, or questions
            if (len(user_message.split()) < 5 or 
                user_message.lower().startswith(("voice", "exit", "quit", "memory", "test", "who", "reset", "wake")) or
                "?" in user_message or 
                not any(keyword in user_message.lower() for keyword in ["i", "my", "me", "we", "our"])):
                return
                
            # Prepare the prompt for fact extraction
            extraction_prompt = f"""
            Extract ONLY 1-2 IMPORTANT factual statements about the user from the following conversation exchange.
            Focus ONLY on significant personal details that would be useful to remember long-term: preferences, background, family, important life events, etc.
            
            DO NOT extract trivial, temporary, or common information.
            DO NOT extract opinions about general topics unless they're strong personal preferences.
            DO NOT extract anything that isn't explicitly stated by the user about themselves.
            
            Format as a list of clear, concise facts. If no significant facts are present, respond with "No important facts found".
            
            User: {user_message}
            AI: {ai_response}
            
            Facts (in format [topic: fact]):
            """
            
            # Extract facts using the language model
            import os
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts only significant personal facts from conversations. Be very selective and only extract truly important information."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            fact_text = response.choices[0].message.content.strip()
            
            # Process the extracted facts
            if not fact_text or "No important facts found" in fact_text or "empty list" in fact_text.lower():
                return
                
            facts = []
            for line in fact_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('-') or line.startswith('*'):
                    line = line[1:].strip()
                
                if ':' in line:
                    topic, fact = line.split(':', 1)
                    facts.append({
                        "topic": topic.strip().lower(),
                        "fact": fact.strip(),
                        "source": f"conversation:{conversation_id}" if conversation_id else "conversation",
                        "timestamp": int(time.time())
                    })
            
            # Store the facts
            if facts:
                for fact_data in facts:
                    self.add_user_fact(
                        user_id=user_id,
                        fact=fact_data["fact"],
                        topic=fact_data["topic"],
                        source=fact_data["source"],
                        timestamp=fact_data["timestamp"]
                    )
                    
        except Exception as e:
            logger.error(f"Error extracting and storing facts: {e}")

class EnhancedMemory:
    """
    Enhanced memory system that combines knowledge database with session memory.
    
    This class manages:
    1. User-specific memories and preferences
    2. Conversation context and history
    3. Knowledge extraction and retrieval
    """
    
    def __init__(self, memory_dir: str = "memory", openai_client=None):
        """
        Initialize the enhanced memory system.
        
        Args:
            memory_dir: Directory for storing memory files
            openai_client: OpenAI client for advanced features
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        
        self.sessions_dir = self.memory_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        self.current_session_id = str(uuid.uuid4())
        self.session_memory = []
        self.user_memory = {}
        self.openai_client = openai_client
        
        # Initialize knowledge database
        self.knowledge_db = KnowledgeDatabase(
            db_path=str(self.memory_dir / "knowledge.db"),
            vector_search=(openai_client is not None)
        )
        
        # Load user memories
        self.load_user_memories()
    
    def load_user_memories(self):
        """Load user memories from file."""
        user_memory_path = self.memory_dir / "user_memory.json"
        if user_memory_path.exists():
            try:
                with open(user_memory_path, 'r') as f:
                    self.user_memory = json.load(f)
                logger.info(f"Loaded memories for {len(self.user_memory)} users")
            except Exception as e:
                logger.error(f"Error loading user memories: {e}")
                self.user_memory = {}
    
    def save_user_memories(self):
        """Save user memories to file."""
        try:
            user_memory_path = self.memory_dir / "user_memory.json"
            with open(user_memory_path, 'w') as f:
                json.dump(self.user_memory, f, indent=2)
            logger.info(f"Saved memories for {len(self.user_memory)} users")
        except Exception as e:
            logger.error(f"Error saving user memories: {e}")
    
    def extract_user_facts(self, text: str, user_id: str):
        """
        Extract facts about the user from their message using pattern matching.
        
        This method identifies statements that contain personal information or preferences
        and stores them in the knowledge database.
        
        Args:
            text: The user message to extract facts from
            user_id: The user ID to associate facts with
        """
        if not text or len(text) < 5 or not user_id:
            return
            
        # Extract facts using various patterns
        facts = []
        
        # Pattern matching for common fact patterns
        patterns = [
            # Preferences
            (r"(?:I|i) (?:like|love|enjoy|prefer) (.+?)[\.!\?]", "likes {}"),
            (r"(?:I|i) (?:hate|dislike|don't like|do not like) (.+?)[\.!\?]", "dislikes {}"),
            (r"(?:my|My) favorite (.+?) (?:is|are) (.+?)[\.!\?]", "favorite {} is {}"),
            
            # Personal info
            (r"(?:I|i) (?:am|'m) (\d+) years old", "age is {}"),
            (r"(?:I|i) (?:am|'m) from (.+?)[\.!\?]", "is from {}"),
            (r"(?:I|i) (?:work|working) (?:at|for|with) (.+?)[\.!\?]", "works at {}"),
            (r"(?:my|My) name is (.+?)[\.!\?]", "name is {}"),
            (r"(?:I|i) (?:am|'m) (?:a|an) (.+?)[\.!\?]", "is a {}"),
            
            # Ownership
            (r"(?:I|i) have (?:a|an) (.+?)[\.!\?]", "has a {}"),
            (r"(?:my|My) (.+?) is (.+?)[\.!\?]", "{} is {}"),
            
            # Activities
            (r"(?:I|i) (?:recently|just) (.+?)[\.!\?]", "recently {}"),
            (r"(?:I|i) want to (.+?)[\.!\?]", "wants to {}"),
            (r"(?:I|i) need to (.+?)[\.!\?]", "needs to {}"),
            (r"(?:I|i) plan to (.+?)[\.!\?]", "plans to {}"),
            
            # Opinions
            (r"(?:I|i) think (?:that )?(.+?)[\.!\?]", "thinks {}"),
            (r"(?:I|i) believe (?:that )?(.+?)[\.!\?]", "believes {}"),
            
            # Social relationships
            (r"(?:my|My) (?:friend|partner|wife|husband|girlfriend|boyfriend|spouse) (.+?)[\.!\?]", "has relationship with {}"),
        ]
        
        for pattern, template in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) == 1:
                    fact = template.format(match.group(1).strip())
                    facts.append(fact)
                elif len(match.groups()) == 2:
                    fact = template.format(match.group(1).strip(), match.group(2).strip())
                    facts.append(fact)
        
        # Extract "I am/was" statements
        am_statements = re.finditer(r"(?:I|i) (?:am|'m|was|have been) (.+?)[\.!\?]", text)
        for match in am_statements:
            trait = match.group(1).strip()
            # Filter out common filler phrases
            skip_phrases = ["not sure", "just", "going to", "trying to", "sorry", "here", "back"]
            if not any(phrase in trait.lower() for phrase in skip_phrases) and len(trait) > 3:
                facts.append(f"is {trait}")
        
        # Extract "I can/can't" abilities
        ability_statements = re.finditer(r"(?:I|i) can(?:'t| not)? (.+?)[\.!\?]", text)
        for match in ability_statements:
            ability = match.group(1).strip()
            if "not" in text or "n't" in text:
                facts.append(f"cannot {ability}")
            else:
                facts.append(f"can {ability}")
        
        # Extract known entities using simple NER
        def extract_entities(text):
            """Extract potential named entities using simple rules"""
            entities = []
            
            # Look for capitalized phrases that might be names or places
            cap_pattern = r'(?<!\. )(?<!\? )(?<!\! )([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
            matches = re.finditer(cap_pattern, text)
            for match in matches:
                entity = match.group(1).strip()
                # Filter out common words that might be capitalized
                if len(entity) > 2 and entity not in ["I", "You", "He", "She", "They", "We", "The"]:
                    entities.append(entity)
            
            return entities
        
        entities = extract_entities(text)
        
        # Add the extracted facts to the database
        for fact in facts:
            try:
                if len(fact) > 3:
                    self.knowledge_db.add_user_fact(user_id, fact, source="conversation")
                    logger.info(f"Added user fact for {user_id}: {fact}")
            except Exception as e:
                logger.error(f"Error adding user fact: {e}")
                
        # Track mentioned entities
        for entity in entities:
            try:
                if len(entity) > 1:
                    self.knowledge_db.add_user_fact(
                        user_id, 
                        f"mentioned {entity}", 
                        source="entity_extraction"
                    )
            except Exception as e:
                logger.error(f"Error adding entity mention: {e}")
                
        return facts
    
    def get_user_memory(self, user_id: str) -> str:
        """
        Get a summary of what's known about a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Text summary of user information
        """
        if not user_id:
            return ""
            
        # Combine database facts with memory
        db_facts = self.knowledge_db.get_user_facts(user_id)
        
        # Get memory facts
        memory_facts = []
        if user_id in self.user_memory and "facts" in self.user_memory[user_id]:
            memory_facts = self.user_memory[user_id]["facts"]
        
        # Combine all facts
        all_facts = []
        fact_texts = set()  # To avoid duplicates
        
        for fact in db_facts:
            if fact["fact"] not in fact_texts:
                all_facts.append(fact["fact"])
                fact_texts.add(fact["fact"])
        
        for fact in memory_facts:
            if fact["fact"] not in fact_texts:
                all_facts.append(fact["fact"])
                fact_texts.add(fact["fact"])
        
        if all_facts:
            return "About this user: " + "; ".join(all_facts)
        else:
            return ""
    
    def get_memory_context(self, query: str = None) -> str:
        """
        Get relevant context from memory based on query.
        
        Args:
            query: Query to retrieve context for
            
        Returns:
            Text containing relevant context
        """
        if not query or len(query) < 5:
            return ""
            
        # Search knowledge database
        results = self.knowledge_db.search(query, top_k=3)
        
        if results:
            context_items = []
            for result in results:
                context_items.append(f"{result['content']}")
            return "\n".join(context_items)
        else:
            return ""
    
    def save_session(self, session_data: List[Dict[str, Any]] = None):
        """
        Save the current session to disk.
        
        Args:
            session_data: Session data to save (uses self.session_memory if None)
        """
        if session_data:
            self.session_memory = session_data
            
        if not self.session_memory:
            return
            
        try:
            session_path = self.sessions_dir / f"session_{self.current_session_id}.json"
            with open(session_path, 'w') as f:
                json.dump({
                    "session_id": self.current_session_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "messages": self.session_memory
                }, f, indent=2)
                
            # Update latest session pointer
            latest_path = self.sessions_dir / "latest_session.txt"
            with open(latest_path, 'w') as f:
                f.write(self.current_session_id)
                
            logger.info(f"Saved session with {len(self.session_memory)} messages")
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    def load_session(self):
        """Load the most recent session from disk."""
        try:
            # Find latest session
            latest_path = self.sessions_dir / "latest_session.txt"
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    session_id = f.read().strip()
                    
                session_path = self.sessions_dir / f"session_{session_id}.json"
                if session_path.exists():
                    with open(session_path, 'r') as f:
                        session_data = json.load(f)
                        self.session_memory = session_data["messages"]
                        self.current_session_id = session_id
                        logger.info(f"Loaded session with {len(self.session_memory)} messages")
                        return True
            
            # No previous session found
            self.session_memory = []
            return False
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            self.session_memory = []
            return False
    
    def process_message(self, message: Dict[str, Any]):
        """
        Process a message to extract knowledge.
        
        Args:
            message: Message dictionary with role and content
        """
        if not message or "content" not in message or "role" not in message:
            return
            
        content = message["content"]
        role = message["role"]
        
        # Only process substantial content
        if not content or len(content) < 20:
            return
            
        # Add substantial responses to knowledge database
        if role == "assistant":
            self.knowledge_db.add_entry(
                content=content,
                source="tars_response",
                topic="conversation",
                confidence=0.9
            )
        # Add substantial user messages too
        elif role == "user" and len(content) > 40:
            self.knowledge_db.add_entry(
                content=content,
                source="user_message",
                topic="conversation",
                confidence=0.7
            )
    
    def close(self):
        """Close databases and save state."""
        if hasattr(self, 'knowledge_db'):
            self.knowledge_db.close()
        self.save_user_memories()
        self.save_session()
    
    def get_relevant_memories(self, query: str, user_id: str, limit: int = 5):
        """
        Get relevant memories for a given query and user.
        
        This method retrieves memories that are contextually relevant to the current
        conversation based on semantic similarity and recency.
        
        Args:
            query: The user's current query
            user_id: The user ID to get memories for
            limit: Maximum number of memories to return
            
        Returns:
            String containing relevant memories formatted for context
        """
        if not query or not user_id:
            return ""
            
        relevant_facts = []
        try:
            # Categorize the query to improve memory retrieval
            query_category = self._categorize_query(query)
            
            # Get topic-specific facts from the knowledge database
            if query_category:
                topic_facts = self.knowledge_db.get_user_facts_by_topic(
                    user_id=user_id,
                    topic=query_category,
                    limit=limit * 2  # Get more than needed to filter best matches
                )
                
                # Add topic-specific facts with high relevance
                if topic_facts:
                    for fact in topic_facts:
                        relevant_facts.append({
                            "text": fact["fact"],
                            "score": 0.9,  # High score for topic-relevant facts
                            "source": fact.get("source", "conversation"),
                            "timestamp": fact.get("timestamp", "")
                        })
            
            # Get semantically similar facts using vector search
            similar_facts = []
            try:
                if hasattr(self.knowledge_db, 'vector_search'):
                    similar_facts = self.knowledge_db.vector_search(
                        query=query,
                        collection="user_facts",
                        filter_params={"user_id": user_id},
                        limit=limit * 2
                    )
            except Exception as e:
                logger.warning(f"Error in vector search: {e}")
                
                # Fallback to simple keyword search if vector search fails
                similar_facts = self._keyword_search(query, user_id, limit * 2)
                
            # Add semantically similar facts
            for fact in similar_facts:
                # Avoid duplicates
                if not any(rf["text"] == fact["fact"] for rf in relevant_facts):
                    relevant_facts.append({
                        "text": fact["fact"],
                        "score": fact.get("score", 0.75),
                        "source": fact.get("source", "conversation"),
                        "timestamp": fact.get("timestamp", "")
                    })
            
            # Get recent facts (temporal relevance)
            recent_facts = self.knowledge_db.get_recent_user_facts(
                user_id=user_id,
                limit=limit
            )
            
            # Add recent facts with modest relevance
            for fact in recent_facts:
                # Avoid duplicates
                if not any(rf["text"] == fact["fact"] for rf in relevant_facts):
                    relevant_facts.append({
                        "text": fact["fact"],
                        "score": 0.6,  # Modest score for recency
                        "source": fact.get("source", "conversation"),
                        "timestamp": fact.get("timestamp", "")
                    })
            
            # Sort by relevance score and take top results
            relevant_facts = sorted(relevant_facts, key=lambda x: x["score"], reverse=True)[:limit]
            
            if not relevant_facts:
                return ""
                
            # Format the relevant memories as context
            memory_context = "User information:\n"
            for fact in relevant_facts:
                memory_context += f"- {fact['text']}\n"
                
            return memory_context
                
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return ""
            
    def _categorize_query(self, query: str):
        """
        Categorize a query to help retrieve relevant memories.
        
        Args:
            query: The query to categorize
            
        Returns:
            Category string or None
        """
        query = query.lower()
        
        # Define category patterns
        categories = {
            "personal": ["you", "your name", "about you", "yourself", "who are you"],
            "preferences": ["like", "love", "enjoy", "prefer", "favorite", "hate", "dislike"],
            "background": ["from", "live", "work", "job", "education", "study", "grew up"],
            "family": ["family", "parents", "children", "kids", "mother", "father", "spouse", "married"],
            "health": ["health", "medical", "doctor", "sick", "illness", "condition", "disease"],
            "activities": ["do for fun", "hobby", "hobbies", "activity", "activities", "sport", "sports"],
            "opinions": ["think about", "opinion", "feel about", "thoughts on", "believe"],
            "plans": ["plan", "future", "goal", "goals", "want to", "going to"]
        }
        
        # Check for category matches
        for category, keywords in categories.items():
            if any(keyword in query for keyword in keywords):
                return category
                
        return None
        
    def _keyword_search(self, query: str, user_id: str, limit: int = 10):
        """
        Perform a simple keyword search as fallback when vector search is unavailable.
        
        Args:
            query: The search query
            user_id: User ID to search facts for
            limit: Maximum results to return
            
        Returns:
            List of matching fact dictionaries
        """
        # Get all user facts
        all_facts = self.knowledge_db.get_user_facts(user_id, limit=100)
        
        if not all_facts:
            return []
            
        # Extract keywords (simple approach)
        keywords = set(query.lower().split())
        keywords = {k for k in keywords if len(k) > 3}  # Filter short words
        
        # Skip if no substantial keywords
        if not keywords:
            return []
            
        # Score facts by keyword matches
        scored_facts = []
        for fact in all_facts:
            fact_text = fact["fact"].lower()
            match_count = sum(1 for k in keywords if k in fact_text)
            if match_count > 0:
                scored_facts.append({
                    "fact": fact["fact"],
                    "score": match_count / len(keywords),
                    "source": fact.get("source", "conversation"),
                    "timestamp": fact.get("timestamp", "")
                })
                
        # Sort by score and return top results
        return sorted(scored_facts, key=lambda x: x["score"], reverse=True)[:limit]

# Example usage
if __name__ == "__main__":
    # Initialize with OpenAI client if available
    import os
    openai_client = None
    if os.environ.get("OPENAI_API_KEY"):
        import openai
        openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create enhanced memory
    memory = EnhancedMemory(memory_dir="memory", openai_client=openai_client)
    
    # Add knowledge
    memory.knowledge_db.add_entry(
        content="The Earth is the third planet from the Sun",
        source="facts",
        topic="astronomy"
    )
    
    # Search knowledge
    results = memory.knowledge_db.search("planets")
    for result in results:
        print(f"Found: {result['content']}")
    
    # Get stats
    stats = memory.knowledge_db.get_stats()
    print(f"Knowledge database stats: {stats}")
    
    # Clean up
    memory.close() 