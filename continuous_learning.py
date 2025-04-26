#!/usr/bin/env python3
"""
Continuous Learning System for TARS

This module implements mechanisms for TARS to learn and improve from interactions,
enabling the assistant to get better over time without manual updates.
"""

import os
import json
import logging
import datetime
import openai
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import time
import sqlite3
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tars.learning")

class ContinuousLearningSystem:
    """
    A system for TARS to improve its responses over time based on user interactions.
    
    The system tracks:
    - Similar user queries and their successful responses
    - Tools used for specific query types
    - Response strategies that worked well
    - User preferences and patterns
    """
    
    def __init__(self, storage_path=None):
        """
        Initialize the continuous learning system.
        
        Args:
            storage_path: Path to store learning data. Defaults to user directory/.tars/learning
        """
        self.logger = logging.getLogger("ContinuousLearning")
        
        if storage_path is None:
            user_home = os.path.expanduser("~")
            self.storage_path = os.path.join(user_home, ".tars", "learning")
        else:
            self.storage_path = storage_path
            
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize the database
        self.db_path = os.path.join(self.storage_path, "learning.db")
        self._initialize_database()
        
        self.logger.info(f"Continuous learning system initialized with storage at: {self.storage_path}")
    
    def _initialize_database(self):
        """Set up the SQLite database for storing learning data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create interactions table to store query-response pairs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            tools_used TEXT,
            success INTEGER DEFAULT 0,
            timestamp REAL NOT NULL
        )
        ''')
        
        # Create table for storing semantic embeddings of queries for similarity search
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_embeddings (
            interaction_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES interactions (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_interaction(self, query, response, tools_used=None, success=None):
        """
        Record an interaction between the user and TARS.
        
        Args:
            query: The user's query
            response: TARS's response
            tools_used: List of tools used or dictionary with tool information
            success: Boolean indicating if the response was successful
                    (None if not known yet)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Make sure we convert all values to types SQLite can handle
        # Ensure query and response are strings
        query = str(query) if query is not None else ""
        response = str(response) if response is not None else ""
        
        # Convert tools_used to JSON string, ensuring all values are serializable
        if tools_used is not None:
            # For dictionaries with boolean values, convert booleans to integers
            if isinstance(tools_used, dict):
                # Convert boolean values to integers for SQLite compatibility
                serializable_tools = {}
                for key, value in tools_used.items():
                    if isinstance(value, bool):
                        serializable_tools[key] = 1 if value else 0
                    else:
                        serializable_tools[key] = value
                tools_used = serializable_tools
            
            tools_json = json.dumps(tools_used)
        else:
            tools_json = None
        
        # Convert success boolean to integer (1 or 0)
        success_int = 1 if success else 0
        
        # Store the interaction
        cursor.execute(
            "INSERT INTO interactions (query, response, tools_used, success, timestamp) VALUES (?, ?, ?, ?, ?)",
            (query, response, tools_json, success_int, time.time())
        )
        
        interaction_id = cursor.lastrowid
        
        # TODO: Generate and store embedding in query_embeddings table
        # This would require an embedding model
        
        conn.commit()
        conn.close()
        
        self.logger.debug(f"Recorded interaction with ID: {interaction_id}")
        return interaction_id
    
    def update_success(self, interaction_id, success):
        """
        Update the success status of a previously recorded interaction.
        
        Args:
            interaction_id: ID of the interaction to update
            success: Boolean indicating if the response was successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE interactions SET success = ? WHERE id = ?",
            (1 if success else 0, interaction_id)
        )
        
        conn.commit()
        conn.close()
        
        self.logger.debug(f"Updated success for interaction {interaction_id} to {success}")
    
    def record_implicit_feedback(self, interaction_id, user_followup):
        """
        Record implicit feedback from a user's follow-up message.
        
        Args:
            interaction_id: ID of the previous interaction
            user_followup: User's follow-up message that might indicate satisfaction/dissatisfaction
        """
        # Analyze followup for sentiment
        # For a simple implementation, check for positive/negative keywords
        positive_indicators = ["thanks", "thank you", "good", "great", "helpful", "perfect", "awesome"]
        negative_indicators = ["wrong", "incorrect", "bad", "not helpful", "useless", "confused", "unclear"]
        
        user_followup_lower = user_followup.lower()
        
        # Calculate a basic sentiment score
        sentiment = 0.5  # Neutral by default
        if any(indicator in user_followup_lower for indicator in positive_indicators):
            sentiment = 0.8  # Positive
        elif any(indicator in user_followup_lower for indicator in negative_indicators):
            sentiment = 0.2  # Negative
        
        # Record the feedback
        self.update_success(interaction_id, sentiment > 0.5)
        
        self.logger.debug(f"Recorded implicit feedback for interaction {interaction_id} with sentiment {sentiment}")
        return sentiment
    
    def record_explicit_feedback(self, interaction_id, score, feedback_text=None):
        """
        Record explicit feedback from the user about a previous interaction.
        
        Args:
            interaction_id: ID of the interaction to update
            score: Numeric score representing satisfaction (0.0 to 1.0)
            feedback_text: Optional text feedback
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update success based on score threshold
        success = score >= 0.5
        
        cursor.execute(
            "UPDATE interactions SET success = ? WHERE id = ?",
            (1 if success else 0, interaction_id)
        )
        
        # TODO: In a more advanced implementation, store the actual score and feedback text
        # in a separate feedback table
        
        conn.commit()
        conn.close()
        
        self.logger.debug(f"Recorded explicit feedback for interaction {interaction_id} with score {score}")
        return True
    
    def get_similar_interactions(self, query, limit=5):
        """
        Find similar previous interactions to help inform the response.
        
        Args:
            query: The current user query
            limit: Maximum number of similar interactions to return
            
        Returns:
            List of dictionaries containing similar interactions
        """
        # For now, implement a simple keyword-based similarity
        # In a more advanced version, this would use embeddings
        
        keywords = set(query.lower().split())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all successful interactions
        cursor.execute(
            "SELECT id, query, response, tools_used FROM interactions WHERE success = 1"
        )
        
        interactions = cursor.fetchall()
        conn.close()
        
        # Score interactions by keyword overlap
        scored_interactions = []
        for id, prev_query, response, tools_used in interactions:
            prev_keywords = set(prev_query.lower().split())
            overlap = len(keywords.intersection(prev_keywords))
            if overlap > 0:
                scored_interactions.append({
                    'id': id,
                    'query': prev_query,
                    'response': response,
                    'tools_used': json.loads(tools_used) if tools_used else None,
                    'score': overlap / len(keywords)  # Normalize by query length
                })
        
        # Sort by score and take top results
        scored_interactions.sort(key=lambda x: x['score'], reverse=True)
        return scored_interactions[:limit]
    
    def suggest_tools(self, query):
        """
        Suggest tools to use based on similar past queries.
        
        Args:
            query: The current user query
            
        Returns:
            List of suggested tools or None
        """
        similar = self.get_similar_interactions(query)
        
        if not similar:
            return None
        
        # Count tool usage across similar interactions
        tool_counts = defaultdict(int)
        for interaction in similar:
            if interaction['tools_used']:
                for tool in interaction['tools_used']:
                    tool_counts[tool] += 1
        
        # Return the most commonly used tools
        if tool_counts:
            most_common = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
            return [tool for tool, _ in most_common]
        
        return None
    
    def get_learning_insights(self, query):
        """
        Get insights from past learning to improve the current response.
        
        Args:
            query: The current user query
            
        Returns:
            Dictionary with learning insights
        """
        similar = self.get_similar_interactions(query)
        
        if not similar:
            return {
                'similar_interactions': [],
                'count': 0,
                'suggested_tools': None,
                'learning_context': "No similar interactions found in learning history."
            }
        
        suggested_tools = self.suggest_tools(query)
        
        # Extract patterns from similar successful interactions
        contexts = [
            f"Query: {s['query']}\nResponse: {s['response']}\nTools: {s['tools_used']}"
            for s in similar[:3]  # Only use top 3 for context
        ]
        
        learning_context = (
            f"Based on {len(similar)} similar past interactions, "
            f"the following approaches were successful:\n"
            + "\n---\n".join(contexts)
        )
        
        return {
            'similar_interactions': similar,
            'count': len(similar),
            'suggested_tools': suggested_tools,
            'learning_context': learning_context
        }
        
    def generate_learning_report(self):
        """
        Generate a comprehensive report of what the system has learned over time.
        
        Returns:
            dict: A report containing statistics and insights
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic statistics
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total_interactions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM interactions WHERE success = 1")
        successful_interactions = cursor.fetchone()[0]
        
        # Get all queries and responses for analysis
        cursor.execute("SELECT query, response, tools_used, success FROM interactions")
        interactions = cursor.fetchall()
        
        conn.close()
        
        # Initialize report data
        report = {
            'stats': {
                'total_interactions': total_interactions,
                'successful_interactions': successful_interactions,
                'feedback_received': total_interactions,  # Simplified for now
            },
            'topic_expertise': {},
            'best_topics': [],
            'improvement_areas': [],
            'common_patterns': []
        }
        
        # Simple topic extraction and scoring
        topic_counts = {}
        topic_success = {}
        
        # Basic topic extraction from queries
        common_topics = [
            'weather', 'news', 'time', 'date', 'location', 'math', 'calculation',
            'music', 'movie', 'sport', 'food', 'travel', 'health', 'technology',
            'programming', 'science', 'history', 'politics', 'entertainment'
        ]
        
        for query, response, tools_used, success in interactions:
            # Extract topics from query
            query_lower = query.lower()
            
            # Identify topics in the query
            for topic in common_topics:
                if topic in query_lower:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    topic_success[topic] = topic_success.get(topic, 0) + (1 if success else 0)
        
        # Calculate topic expertise scores
        for topic, count in topic_counts.items():
            if count >= 3:  # Only include topics with minimum interactions
                success_rate = topic_success[topic] / count
                report['topic_expertise'][topic] = {
                    'count': count,
                    'score': success_rate,
                }
        
        # Identify best topics and improvement areas
        sorted_topics = sorted(
            [(topic, data['score'], data['count']) 
             for topic, data in report['topic_expertise'].items()],
            key=lambda x: x[1],  # Sort by score
            reverse=True
        )
        
        # Best topics are top performers
        report['best_topics'] = [topic for topic, score, count in sorted_topics[:3] if score >= 0.7]
        
        # Improvement areas are low performers with decent sample size
        report['improvement_areas'] = [topic for topic, score, count in sorted_topics if score < 0.5 and count >= 3]
        
        # Identify common patterns - simplified implementation
        report['common_patterns'] = [
            "Users often follow up factual questions with clarification questions",
            "Simple, direct questions tend to get better responses",
            "Queries about current events benefit from web search integration"
        ]
        
        return report


# Global instance for easy access
_learning_system = None

def get_learning_system():
    """Get the global learning system instance."""
    global _learning_system
    if _learning_system is None:
        _learning_system = ContinuousLearningSystem()
    return _learning_system

if __name__ == "__main__":
    # Simple test when running this module directly
    learning_system = ContinuousLearningSystem()
    
    # Record a test interaction
    interaction_id = learning_system.record_interaction(
        query="What's the weather like in New York?",
        response="The current weather in New York is 72°F with partly cloudy skies. There's a 20% chance of rain later today.",
        tool_used="Weather Tool"
    )
    
    # Print recorded interaction
    print(f"Recorded interaction with ID: {interaction_id}")
    
    # Generate a learning report
    report = learning_system.get_learning_insights("What's the weather like in New York?")
    print("Learning Insights:")
    print(json.dumps(report, indent=2)) 