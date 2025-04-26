import logging
import json
import os
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tars.reflection")

class ReflectionSystem:
    """
    A system for self-awareness and introspection that allows the assistant
    to analyze its own behavior, responses, and performance.
    """
    
    def __init__(self, data_path: str = "reflection_data.json"):
        """
        Initialize the reflection system
        
        Args:
            data_path: Path to store reflection data
        """
        self.data_path = data_path
        self.reflection_data = {
            "conversation_analyses": [],
            "performance_metrics": {
                "response_times": [],
                "accuracy_scores": [],
                "user_satisfaction": []
            },
            "strengths": [],
            "weaknesses": [],
            "improvement_goals": [],
            "self_critiques": [],
            "user_insights": {},
            "last_updated": datetime.now().isoformat()
        }
        self.load_data()
        
    def load_data(self):
        """Load existing reflection data"""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r') as f:
                    self.reflection_data = json.load(f)
                
                logger.info(f"Loaded reflection data from {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading reflection data: {str(e)}")
            # Keep the default data structure
            
    def save_data(self):
        """Save reflection data to storage"""
        self.reflection_data["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.data_path, 'w') as f:
                json.dump(self.reflection_data, f, indent=2)
                
            logger.info(f"Saved reflection data to {self.data_path}")
        except Exception as e:
            logger.error(f"Error saving reflection data: {str(e)}")
            
    def analyze_conversation(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze a conversation to extract patterns, user preferences, and self-assessment
        
        Args:
            conversation_history: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Analysis results
        """
        if not conversation_history:
            return {}
            
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(conversation_history),
            "user_messages": sum(1 for msg in conversation_history if msg.get("role") == "user"),
            "assistant_messages": sum(1 for msg in conversation_history if msg.get("role") == "assistant"),
            "common_phrases": self._extract_common_phrases(conversation_history),
            "sentiment": self._analyze_sentiment(conversation_history),
            "topics": self._extract_topics(conversation_history),
            "response_quality": self._assess_response_quality(conversation_history)
        }
        
        self.reflection_data["conversation_analyses"].append(analysis)
        self.save_data()
        
        return analysis
        
    def _extract_common_phrases(self, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Extract commonly used phrases from conversation"""
        all_text = " ".join([msg.get("content", "") for msg in conversation_history])
        
        # Simple phrase extraction using regex for 2-3 word phrases
        phrases = re.findall(r'\b(\w+\s+\w+(\s+\w+)?)\b', all_text.lower())
        phrase_counter = Counter([phrase[0] for phrase in phrases])
        
        # Return top phrases that appear more than once
        return [phrase for phrase, count in phrase_counter.most_common(5) if count > 1]
        
    def _analyze_sentiment(self, conversation_history: List[Dict[str, str]]) -> Dict[str, float]:
        """Simple sentiment analysis of conversation"""
        # This is a placeholder - in production, use a proper NLP library
        positive_words = {"good", "great", "excellent", "thanks", "helpful", "appreciate", "like", "love"}
        negative_words = {"bad", "wrong", "incorrect", "not", "issue", "problem", "difficult", "confused"}
        
        user_messages = [msg.get("content", "").lower() for msg in conversation_history 
                         if msg.get("role") == "user"]
        
        pos_count = 0
        neg_count = 0
        
        for msg in user_messages:
            words = set(re.findall(r'\b\w+\b', msg))
            pos_count += len(words.intersection(positive_words))
            neg_count += len(words.intersection(negative_words))
            
        total = pos_count + neg_count
        if total == 0:
            return {"positive": 0.5, "negative": 0.5}
            
        return {
            "positive": pos_count / total if total > 0 else 0.5,
            "negative": neg_count / total if total > 0 else 0.5
        }
        
    def _extract_topics(self, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Extract main topics from conversation"""
        # Simple keyword extraction - in production use topic modeling
        all_text = " ".join([msg.get("content", "") for msg in conversation_history])
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "is", "are"}
        words = [word.lower() for word in re.findall(r'\b\w+\b', all_text) 
                if word.lower() not in stop_words and len(word) > 3]
        
        # Return top frequent words as topics
        return [word for word, count in Counter(words).most_common(5)]
        
    def _assess_response_quality(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Assess the quality of assistant responses"""
        assistant_msgs = [msg.get("content", "") for msg in conversation_history 
                         if msg.get("role") == "assistant"]
        
        if not assistant_msgs:
            return {"average_length": 0, "complexity": 0, "diversity": 0}
        
        # Average response length
        avg_length = sum(len(msg.split()) for msg in assistant_msgs) / len(assistant_msgs)
        
        # Lexical diversity (unique words / total words)
        all_words = []
        for msg in assistant_msgs:
            all_words.extend(re.findall(r'\b\w+\b', msg.lower()))
            
        diversity = len(set(all_words)) / len(all_words) if all_words else 0
        
        # Simple estimation of complexity
        complex_words = sum(1 for word in all_words if len(word) > 7)
        complexity = complex_words / len(all_words) if all_words else 0
        
        return {
            "average_length": avg_length,
            "complexity": complexity,
            "diversity": diversity
        }
        
    def update_performance_metrics(self, 
                               accuracy_score: float = None, 
                               user_satisfaction: float = None, 
                               response_time: float = None) -> None:
        """
        Update performance metrics based on recent interactions
        
        Args:
            accuracy_score: Score indicating accuracy (0-1)
            user_satisfaction: Score indicating user satisfaction (0-1)
            response_time: Response time in seconds
        """
        if accuracy_score is not None:
            self.reflection_data["performance_metrics"]["accuracy_scores"].append(accuracy_score)
            
        if user_satisfaction is not None:
            self.reflection_data["performance_metrics"]["user_satisfaction"].append(user_satisfaction)
            
        if response_time is not None:
            self.reflection_data["performance_metrics"]["response_times"].append(response_time)
            
        self.save_data()
        
    def identify_strengths_and_weaknesses(self) -> Tuple[List[str], List[str]]:
        """
        Analyze performance data to identify strengths and weaknesses
        
        Returns:
            Tuple containing lists of strengths and weaknesses
        """
        metrics = self.reflection_data["performance_metrics"]
        analyses = self.reflection_data["conversation_analyses"]
        
        strengths = []
        weaknesses = []
        
        # Check response times
        if metrics["response_times"]:
            avg_time = sum(metrics["response_times"]) / len(metrics["response_times"])
            if avg_time < 2.0:  # Assuming 2 seconds is good
                strengths.append("Quick response times")
            else:
                weaknesses.append("Response latency could be improved")
        
        # Check user satisfaction
        if metrics["user_satisfaction"]:
            avg_satisfaction = sum(metrics["user_satisfaction"]) / len(metrics["user_satisfaction"])
            if avg_satisfaction > 4.0:  # On a 5-point scale
                strengths.append("High user satisfaction")
            elif avg_satisfaction < 3.0:
                weaknesses.append("User satisfaction needs improvement")
        
        # Check conversation analyses
        if analyses:
            latest_analyses = analyses[-5:] if len(analyses) > 5 else analyses
            
            # Check sentiment trends
            positive_sentiment = sum(analysis.get("sentiment", {}).get("positive", 0) 
                                   for analysis in latest_analyses) / len(latest_analyses)
            
            if positive_sentiment > 0.7:
                strengths.append("Maintains positive user sentiment")
            elif positive_sentiment < 0.4:
                weaknesses.append("Needs to better address user concerns")
                
            # Check response quality
            avg_complexity = sum(analysis.get("response_quality", {}).get("complexity", 0) 
                               for analysis in latest_analyses) / len(latest_analyses)
            
            if avg_complexity > 0.2:
                strengths.append("Provides detailed, complex responses")
            else:
                weaknesses.append("Could provide more nuanced responses")
                
        # Update strengths and weaknesses in reflection data
        self.reflection_data["strengths"] = strengths
        self.reflection_data["weaknesses"] = weaknesses
        self.save_data()
        
        return strengths, weaknesses
        
    def set_improvement_goal(self, goal: str) -> None:
        """
        Set a specific improvement goal
        
        Args:
            goal: Description of the improvement goal
        """
        if goal and goal not in self.reflection_data["improvement_goals"]:
            self.reflection_data["improvement_goals"].append({
                "goal": goal,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            })
            self.save_data()
        
    def add_self_critique(self, observation: str) -> None:
        """
        Add a self-critique observation
        
        Args:
            observation: The critique or observation
        """
        if observation:
            self.reflection_data["self_critiques"].append({
                "observation": observation,
                "timestamp": datetime.now().isoformat()
            })
            self.save_data()
        
    def update_user_insights(self, user_id: str, insight: Dict[str, Any]) -> None:
        """
        Update insights about a specific user
        
        Args:
            user_id: User identifier
            insight: Dictionary of user insights to update
        """
        if user_id not in self.reflection_data["user_insights"]:
            self.reflection_data["user_insights"][user_id] = []
            
        self.reflection_data["user_insights"][user_id].append({
            "insight": insight,
            "timestamp": datetime.now().isoformat()
        })
        self.save_data()
        
    def get_self_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive self-assessment based on reflection data
        
        Returns:
            Self-assessment data
        """
        strengths, weaknesses = self.identify_strengths_and_weaknesses()
        
        # Calculate performance trends
        metrics = self.reflection_data["performance_metrics"]
        trends = {}
        
        for metric_name, values in metrics.items():
            if len(values) >= 2:
                recent_values = values[-10:] if len(values) > 10 else values
                if len(recent_values) >= 2:
                    first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
                    second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
                    trends[metric_name] = "improving" if second_half > first_half else "declining"
        
        # Get recent goals
        recent_goals = self.reflection_data["improvement_goals"][-3:] if self.reflection_data["improvement_goals"] else []
        
        # Get recent self-critiques
        recent_critiques = self.reflection_data["self_critiques"][-5:] if self.reflection_data["self_critiques"] else []
        
        return {
            "timestamp": datetime.now().isoformat(),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "performance_trends": trends,
            "active_goals": recent_goals,
            "recent_critiques": recent_critiques,
            "conversation_count": len(self.reflection_data["conversation_analyses"])
        }

# Utility functions for external use
def analyze_conversation_history(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a conversation for patterns and insights
    
    Args:
        conversation_history: List of message dictionaries
        
    Returns:
        Analysis results
    """
    reflection = ReflectionSystem()
    return reflection.analyze_conversation(conversation_history)
    
def record_performance_metrics(accuracy_score: float = None, 
                         user_satisfaction: float = None,
                         response_time: float = None):
    """
    Record performance metrics
    
    Args:
        accuracy_score: Score indicating accuracy (0-1)
        user_satisfaction: Score indicating user satisfaction (0-1)
        response_time: Response time in seconds
    """
    reflection = ReflectionSystem()
    reflection.update_performance_metrics(accuracy_score, user_satisfaction, response_time)
    
def add_self_critique_observation(critique: str):
    """
    Add a self-critique observation
    
    Args:
        critique: The critique or observation
    """
    reflection = ReflectionSystem()
    reflection.add_self_critique(critique)
    
def get_self_assessment_report() -> str:
    """
    Get a comprehensive self-assessment report
    
    Returns:
        Formatted self-report string
    """
    reflection = ReflectionSystem()
    return reflection.generate_self_report()
    
def set_assistant_preferences(preferences: Dict[str, str]):
    """
    Set preference settings for the assistant
    
    Args:
        preferences: Dictionary of preference settings
    """
    reflection = ReflectionSystem()
    reflection.set_meta_preferences(preferences)

def get_performance_metrics():
    """
    Get average performance metrics for recent interactions
    
    Returns:
        Dictionary with average metrics for accuracy, satisfaction, and response time
    """
    reflection = ReflectionSystem()
    metrics = reflection.reflection_data["performance_metrics"]
    
    # Calculate averages if data exists
    result = {}
    
    if metrics["accuracy_scores"]:
        # Get recent scores (last 20 or all if fewer)
        recent_accuracy = metrics["accuracy_scores"][-20:] if len(metrics["accuracy_scores"]) > 20 else metrics["accuracy_scores"]
        result["avg_accuracy"] = sum(recent_accuracy) / len(recent_accuracy)
    else:
        result["avg_accuracy"] = None
        
    if metrics["user_satisfaction"]:
        # Get recent satisfaction scores
        recent_satisfaction = metrics["user_satisfaction"][-20:] if len(metrics["user_satisfaction"]) > 20 else metrics["user_satisfaction"]
        result["avg_satisfaction"] = sum(recent_satisfaction) / len(recent_satisfaction)
    else:
        result["avg_satisfaction"] = None
        
    if metrics["response_times"]:
        # Get recent response times
        recent_times = metrics["response_times"][-20:] if len(metrics["response_times"]) > 20 else metrics["response_times"]
        result["avg_response_time"] = sum(recent_times) / len(recent_times)
    else:
        result["avg_response_time"] = None
        
    return result

def get_performance_insights():
    """
    Analyze performance metrics and provide actionable insights
    
    Returns:
        List of specific recommendations based on current metrics
    """
    reflection = ReflectionSystem()
    metrics = reflection.reflection_data["performance_metrics"]
    insights = []
    
    # Check if we have enough data to provide insights
    accuracy_scores = metrics["accuracy_scores"]
    satisfaction_scores = metrics["user_satisfaction"]
    response_times = metrics["response_times"]
    
    # Only provide insights if we have a minimum amount of data
    if len(accuracy_scores) < 5 and len(satisfaction_scores) < 3:
        insights.append("Not enough data collected yet for meaningful insights.")
        return insights
    
    # Check accuracy trends
    if accuracy_scores:
        recent = accuracy_scores[-5:] if len(accuracy_scores) >= 5 else accuracy_scores
        avg_recent = sum(recent) / len(recent)
        
        if avg_recent < 0.6:
            insights.append("Consider improving response quality by providing more detailed and contextually relevant information.")
        elif avg_recent > 0.8:
            insights.append("Response quality is high - maintain this level of detail and relevance.")
    
    # Check satisfaction trends
    if satisfaction_scores and len(satisfaction_scores) >= 3:
        recent = satisfaction_scores[-3:]
        avg_recent = sum(recent) / len(recent)
        
        if avg_recent < 0.6:
            insights.append("User satisfaction is below optimal levels - focus on addressing user queries more directly.")
        elif avg_recent > 0.8:
            insights.append("User satisfaction is high - continue providing helpful responses.")
    
    # Check response time
    if response_times:
        recent = response_times[-10:] if len(response_times) >= 10 else response_times
        avg_recent = sum(recent) / len(recent)
        
        if avg_recent > 3.0:  # If average response time is over 3 seconds
            insights.append("Response times are longer than optimal - consider optimizing processing.")
        elif avg_recent < 1.0:
            insights.append("Response times are excellent - very responsive.")
    
    # If no specific insights were generated, provide a general statement
    if not insights:
        insights.append("Performance metrics are within normal ranges.")
    
    return insights

# Test the reflection system
if __name__ == "__main__":
    # Create a reflection system
    reflection = ReflectionSystem(data_path="test_reflection_data.json")
    
    # Sample conversation
    conversation = [
        {"role": "user", "content": "Can you help me with Python programming?"},
        {"role": "assistant", "content": "I'd be happy to help you with Python programming! What specific aspects or questions do you have?"},
        {"role": "user", "content": "How do I use dictionaries effectively?"},
        {"role": "assistant", "content": "Dictionaries in Python are versatile data structures for storing key-value pairs. Here are some tips for using them effectively: 1) Use meaningful keys, 2) Take advantage of dictionary comprehensions, 3) Use the get() method to provide default values, 4) Use the items(), keys(), and values() methods for iteration."}
    ]
    
    # Analyze the conversation
    analysis = reflection.analyze_conversation(conversation)
    print(f"Conversation analysis: {analysis['response_quality']}")
    
    # Update performance metrics
    reflection.update_performance_metrics(
        response_time=1.2,
        accuracy_score=0.85,
        user_satisfaction=4
    )
    
    # Add self-critique
    reflection.add_self_critique("Responses could include more examples")
    
    # Set improvement goal
    reflection.set_improvement_goal("Improve code example clarity")
    
    # Update user insights
    reflection.update_user_insights(
        "user123",
        {"technical_level": "intermediate", "interests": ["Python", "data science"]}
    )
    
    # Generate self-report
    report = reflection.get_self_report()
    print("\nSelf-Report:")
    print(report)
    
    # Clean up test file
    import os
    if os.path.exists("test_reflection_data.json"):
        os.remove("test_reflection_data.json") 