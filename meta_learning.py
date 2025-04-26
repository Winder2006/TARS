#!/usr/bin/env python3
import logging
import json
import os
import datetime
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tars.meta_learning")

class MetaLearningSystem:
    """System for tracking response quality and adapting to user preferences over time."""
    
    def __init__(self, data_path="meta_learning_data.json"):
        """Initialize the meta-learning system with data file path."""
        self.data_path = data_path
        self.logger = logging.getLogger("tars.meta_learning")
        self.meta_data = self._load_data()
        
    def _load_data(self):
        """Load meta-learning data from file, or create default structure if it doesn't exist."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as file:
                    return json.load(file)
            except Exception as e:
                self.logger.error(f"Error loading meta-learning data: {str(e)}")
                return self._create_default_data()
        else:
            return self._create_default_data()
    
    def _create_default_data(self):
        """Create default meta-learning data structure."""
        return {
            "response_quality": {
                "ratings": [],
                "average": 0.0
            },
            "user_preferences": {
                "response_style": "",
                "topics_of_interest": [],
                "avoid_topics": []
            },
            "adaptation_history": [],
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def _save_data(self):
        """Save meta-learning data to file."""
        try:
            with open(self.data_path, 'w') as file:
                self.meta_data["last_updated"] = datetime.datetime.now().isoformat()
                json.dump(self.meta_data, file, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving meta-learning data: {str(e)}")
    
    def record_response_quality(self, query, response, quality_score, tool_used=None):
        """
        Record the quality of a response for learning.
        
        Args:
            query (str): The user's query
            response (str): The assistant's response
            quality_score (float): A score from 0.0 to 1.0 indicating quality
            tool_used (str, optional): The tool used to generate the response
        """
        if not 0.0 <= quality_score <= 1.0:
            self.logger.warning(f"Invalid quality score: {quality_score}, must be between 0.0 and 1.0")
            quality_score = max(0.0, min(quality_score, 1.0))
        
        # Record this rating
        rating_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response_length": len(response),
            "quality_score": quality_score,
            "tool_used": tool_used
        }
        
        self.meta_data["response_quality"]["ratings"].append(rating_entry)
        
        # Keep only the last 100 ratings
        if len(self.meta_data["response_quality"]["ratings"]) > 100:
            self.meta_data["response_quality"]["ratings"] = self.meta_data["response_quality"]["ratings"][-100:]
        
        # Update average
        scores = [r["quality_score"] for r in self.meta_data["response_quality"]["ratings"]]
        self.meta_data["response_quality"]["average"] = sum(scores) / len(scores) if scores else 0.0
        
        self._save_data()
        return quality_score
    
    def update_user_preferences(self, conversation_history):
        """
        Update user preferences based on conversation history.
        
        Args:
            conversation_history (list): List of conversation messages
        
        Returns:
            dict: Updated user preferences
        """
        # Extract all user messages
        user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        
        if not user_messages:
            return self.meta_data["user_preferences"]
        
        # Analyze topics of interest
        topics = self._extract_topics(user_messages)
        
        # Update topics of interest if we found any
        if topics:
            current_topics = set(self.meta_data["user_preferences"]["topics_of_interest"])
            for topic, count in topics.items():
                if count >= 2 and topic not in current_topics:  # Topic mentioned at least twice
                    current_topics.add(topic)
            
            self.meta_data["user_preferences"]["topics_of_interest"] = list(current_topics)
            
            # Keep only top 10 topics
            if len(self.meta_data["user_preferences"]["topics_of_interest"]) > 10:
                # Sort by frequency and keep top 10
                top_topics = sorted(
                    [(t, topics.get(t, 0)) for t in self.meta_data["user_preferences"]["topics_of_interest"]],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                self.meta_data["user_preferences"]["topics_of_interest"] = [t[0] for t in top_topics]
        
        # Detect preferred response style
        style = self._detect_response_style(user_messages)
        if style:
            self.meta_data["user_preferences"]["response_style"] = style
        
        self._save_data()
        return self.meta_data["user_preferences"]
    
    def _extract_topics(self, messages):
        """Extract topics of interest from messages."""
        # Combine all messages
        text = " ".join(messages).lower()
        
        # Remove common words
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "then", 
            "once", "here", "there", "when", "where", "why", "how", "all", "any", 
            "both", "each", "few", "more", "most", "other", "some", "such", "no", 
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
            "t", "can", "will", "just", "don", "don't", "should", "now", "d", "ll", 
            "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", 
            "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", 
            "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", 
            "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", 
            "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
            "tell", "know", "like", "think", "good", "get", "make", "want", "sure", "go",
            "really", "see", "time", "well", "things", "lot", "right", "would", "could",
            "way", "make", "look", "also", "still", "thing", "something", "anything",
            "everything", "day", "say", "going", "need", "actually", "means", "means"
        }
        
        # Tokenize and clean text
        words = re.findall(r'\b[a-z]{3,15}\b', text)
        words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Find potential topics (words that appear multiple times)
        potential_topics = {word: count for word, count in word_counts.items() if count >= 2}
        
        return potential_topics
    
    def _detect_response_style(self, messages):
        """Detect preferred response style from user messages."""
        combined_text = " ".join(messages).lower()
        
        # Patterns to detect different style preferences
        style_patterns = {
            "concise": r"(brief|short|quick|concise|summarize|to the point)",
            "detailed": r"(detail|comprehensive|thorough|explain|more information|elaborate)",
            "simple": r"(simple|easy|basic|layman|beginner|understand)",
            "technical": r"(technical|advanced|specific|expert|specialized|in-depth)"
        }
        
        # Check each pattern
        style_matches = {}
        for style, pattern in style_patterns.items():
            matches = re.findall(pattern, combined_text)
            style_matches[style] = len(matches)
        
        # If we have a clear preference with more than one match, return it
        max_style = max(style_matches.items(), key=lambda x: x[1])
        if max_style[1] > 1:
            return max_style[0]
        
        return ""
    
    def get_adaptations(self):
        """
        Get current adaptations for improving responses.
        
        Returns:
            dict: Adaptation parameters for response generation
        """
        return {
            "response_style": self.meta_data["user_preferences"]["response_style"],
            "topics_of_interest": self.meta_data["user_preferences"]["topics_of_interest"],
            "avoid_topics": self.meta_data["user_preferences"].get("avoid_topics", [])
        }
    
    def analyze_response_patterns(self):
        """
        Analyze patterns in response quality to find areas for improvement.
        
        Returns:
            dict: Analysis of response patterns
        """
        ratings = self.meta_data["response_quality"]["ratings"]
        if len(ratings) < 5:  # Need at least 5 ratings for meaningful analysis
            return {"status": "insufficient_data", "message": "Need more data for analysis"}
        
        analysis = {
            "average_quality": self.meta_data["response_quality"]["average"],
            "total_responses": len(ratings),
            "tool_performance": {},
            "response_length_impact": {}
        }
        
        # Analyze tool performance
        tool_scores = defaultdict(list)
        for rating in ratings:
            tool = rating.get("tool_used", "unknown")
            tool_scores[tool].append(rating["quality_score"])
        
        for tool, scores in tool_scores.items():
            avg_score = sum(scores) / len(scores)
            analysis["tool_performance"][tool] = {
                "average_score": avg_score,
                "count": len(scores)
            }
        
        # Analyze impact of response length
        length_brackets = [(0, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, float('inf'))]
        length_scores = defaultdict(list)
        
        for rating in ratings:
            length = rating.get("response_length", 0)
            for min_len, max_len in length_brackets:
                if min_len <= length <= max_len:
                    bracket = f"{min_len}-{max_len if max_len != float('inf') else '+'}"
                    length_scores[bracket].append(rating["quality_score"])
                    break
        
        for bracket, scores in length_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                analysis["response_length_impact"][bracket] = {
                    "average_score": avg_score,
                    "count": len(scores)
                }
        
        return analysis
    
    def manually_set_preference(self, preference_type, value):
        """
        Manually set a user preference.
        
        Args:
            preference_type (str): Type of preference ('response_style', 'topics_of_interest', 'avoid_topics')
            value: The value to set for this preference
            
        Returns:
            bool: Success status
        """
        if preference_type not in ["response_style", "topics_of_interest", "avoid_topics"]:
            self.logger.warning(f"Unknown preference type: {preference_type}")
            return False
        
        # Record adaptation history
        adaptation = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "manual_preference",
            "preference_type": preference_type,
            "value": value
        }
        self.meta_data["adaptation_history"].append(adaptation)
        
        # Set the preference
        if preference_type in ["topics_of_interest", "avoid_topics"]:
            # Ensure lists are properly formatted
            if isinstance(value, str):
                value = [v.strip() for v in value.split(',')]
            elif not isinstance(value, list):
                value = [str(value)]
            self.meta_data["user_preferences"][preference_type] = value
        else:
            self.meta_data["user_preferences"][preference_type] = value
        
        self._save_data()
        return True


# Utility functions for external use

def record_response_quality(query, response, quality_score, tool_used=None):
    """
    Record the quality of a response.
    
    Args:
        query (str): The user's query
        response (str): The assistant's response
        quality_score (float): Quality score from 0.0 to 1.0
        tool_used (str, optional): The tool used for this response
        
    Returns:
        float: The recorded quality score
    """
    meta = MetaLearningSystem()
    return meta.record_response_quality(query, response, quality_score, tool_used)

def update_preferences(conversation_history):
    """
    Update user preferences based on conversation history.
    
    Args:
        conversation_history (list): List of conversation messages
        
    Returns:
        dict: Updated user preferences
    """
    meta = MetaLearningSystem()
    return meta.update_user_preferences(conversation_history)

def get_response_adaptations():
    """
    Get current adaptations for response generation.
    
    Returns:
        dict: Adaptation parameters for the AI responses
    """
    meta = MetaLearningSystem()
    return meta.get_adaptations()

def analyze_response_patterns():
    """
    Analyze patterns in response quality.
    
    Returns:
        dict: Analysis of response patterns
    """
    meta = MetaLearningSystem()
    return meta.analyze_response_patterns()

def set_user_preference(preference_type, value):
    """
    Manually set a user preference.
    
    Args:
        preference_type (str): Type of preference ('response_style', 'topics_of_interest', 'avoid_topics')
        value: The value to set
        
    Returns:
        bool: Success status
    """
    meta = MetaLearningSystem()
    return meta.manually_set_preference(preference_type, value)


if __name__ == "__main__":
    # Basic test
    ml = MetaLearningSystem("test_meta_learning.json")
    
    # Record some sample data
    ml.record_response_quality("What is the weather?", "It's sunny today.", 0.8, "WeatherTool")
    ml.record_response_quality("Tell me about Python programming", "Python is a programming language...", 0.9, "RAG")
    ml.record_response_quality("What's 2+2?", "4", 1.0, "CalculatorTool")
    
    # Update preferences
    test_conversation = [
        {"role": "user", "content": "I want to learn more about artificial intelligence and machine learning"},
        {"role": "assistant", "content": "That's great! What specific aspects are you interested in?"},
        {"role": "user", "content": "I really enjoy reading about neural networks and deep learning techniques"},
        {"role": "assistant", "content": "Neural networks are fascinating! Would you like me to explain how they work?"},
        {"role": "user", "content": "Yes, please give me a detailed explanation of how they function"}
    ]
    
    ml.update_user_preferences(test_conversation)
    
    # Get adaptations
    adaptations = ml.get_adaptations()
    print(f"Adaptations: {adaptations}")
    
    # Analyze patterns
    analysis = ml.analyze_response_patterns()
    print(f"Analysis: {analysis}")
    
    # Clean up test file
    if os.path.exists("test_meta_learning.json"):
        os.remove("test_meta_learning.json") 