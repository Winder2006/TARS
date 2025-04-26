"""
Tools API for TARS

This module provides external API integrations and tools that TARS can use
to enhance responses with real-time data and computational capabilities.
"""

import os
import json
import re
import requests
import datetime
import math
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Weather API key
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class Tool:
    """Base class for TARS tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def can_handle(self, query: str) -> bool:
        """Determine if this tool can handle the given query"""
        raise NotImplementedError("Subclasses must implement can_handle")
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute the tool functionality"""
        raise NotImplementedError("Subclasses must implement execute")


class WeatherTool(Tool):
    """Weather information tool using OpenWeatherMap API"""
    
    def __init__(self):
        super().__init__(
            name="Weather Tool",
            description="Get current weather and forecasts for locations worldwide"
        )
        self.api_key = WEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def can_handle(self, query: str) -> bool:
        """Check if query is asking about weather"""
        query = query.lower()
        weather_keywords = [
            "weather", "temperature", "forecast", "rain", "snow", "sunny", 
            "cloudy", "humidity", "wind", "storm", "precipitation",
            "hot", "cold", "warm", "chilly"
        ]
        
        # Check for location indicator words near weather terms
        location_indicators = ["in", "at", "for", "of"]
        
        # Simple check for weather keywords
        if any(keyword in query for keyword in weather_keywords):
            return True
            
        # More sophisticated check for "what's it like in [location]"
        if "what's it like" in query or "how is it" in query:
            for indicator in location_indicators:
                if indicator in query:
                    return True
        
        return False
    
    def extract_location(self, query: str) -> str:
        """Extract location from query"""
        # Simple location extraction
        location_patterns = [
            r"weather (?:in|at|for) ([\w\s]+)(?:\?|$|\.)",
            r"(?:what's|what is) (?:the|) weather (?:like|) (?:in|at|for) ([\w\s]+)(?:\?|$|\.)",
            r"(?:how's|how is) (?:the|) weather (?:in|at|for) ([\w\s]+)(?:\?|$|\.)",
            r"temperature (?:in|at|for) ([\w\s]+)(?:\?|$|\.)",
            r"forecast (?:for|in|at) ([\w\s]+)(?:\?|$|\.)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try to find location after "in", "at" or "for"
        indicators = ["in", "at", "for"]
        for indicator in indicators:
            if f" {indicator} " in query:
                parts = query.split(f" {indicator} ")
                if len(parts) > 1:
                    # Take the word after the indicator
                    potential_location = parts[1].strip().split()[0].strip("?.,!")
                    if potential_location and len(potential_location) > 2:
                        return potential_location
        
        # Default fallback
        return "current location"
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get weather information"""
        if not self.api_key:
            return {
                "response": "Weather information is unavailable. OpenWeatherMap API key is missing.",
                "source": "Weather Tool",
                "success": False
            }
        
        location = kwargs.get("location") or self.extract_location(query)
        
        try:
            # Get coordinates for the location
            geocode_url = f"{self.base_url}/weather?q={location}&appid={self.api_key}&units=imperial"
            response = requests.get(geocode_url)
            response.raise_for_status()
            data = response.json()
            
            # Extract current weather
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            
            # Format result
            result = (
                f"Weather in {data['name']}: {weather_desc.capitalize()}. "
                f"Temperature: {temp}°F (feels like {feels_like}°F). "
                f"Humidity: {humidity}%. Wind: {wind_speed} mph."
            )
            
            return {
                "response": result,
                "source": "OpenWeatherMap",
                "success": True,
                "data": data
            }
            
        except Exception as e:
            return {
                "response": f"Sorry, I couldn't get weather information for {location}: {str(e)}",
                "source": "Weather Tool",
                "success": False
            }


class CalculatorTool(Tool):
    """Calculator tool for mathematical operations"""
    
    def __init__(self):
        super().__init__(
            name="Calculator",
            description="Perform mathematical calculations"
        )
    
    def can_handle(self, query: str) -> bool:
        """Check if query contains a mathematical calculation"""
        # Remove common phrases to isolate potential math expressions
        query = query.lower()
        query = re.sub(r"(what is|calculate|compute|solve|evaluate|what's|whats)", "", query)
        
        # Check for math operators
        math_operators = ["+", "-", "*", "/", "^", "×", "÷", "plus", "minus", "times", "divided by", "squared", "cubed", "square root", "cube root", "sin", "cos", "tan", "log"]
        has_operator = any(op in query for op in math_operators)
        
        # Check for numbers (including decimal)
        has_number = bool(re.search(r'\d+\.?\d*', query))
        
        return has_operator and has_number
    
    def parse_expression(self, query: str) -> str:
        """Parse a mathematical expression from the query"""
        # Remove question marks and other non-math characters
        query = query.replace("?", "").replace("=", "").strip()
        
        # Replace text operators with symbols
        replacements = {
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
            "÷": "/",
            "×": "*",
            "squared": "** 2",
            "cubed": "** 3",
            "square root": "sqrt",
            "cube root": "cbrt",
            "to the power of": "**",
            "to the power": "**",
            "to power of": "**",
            "to power": "**",
            "^": "**"
        }
        
        for text, symbol in replacements.items():
            query = query.replace(text, symbol)
        
        # Extract the expression
        expression_pattern = r'(-?\d+\.?\d*\s*[\+\-\*\/\(\)\s\*\*]+\s*-?\d+\.?\d*\s*[\+\-\*\/\(\)\s\*\*]*\d*\.?\d*)'
        match = re.search(expression_pattern, query)
        
        if match:
            return match.group(0).strip()
        
        # If no clear expression, extract parts
        numbers = re.findall(r'-?\d+\.?\d*', query)
        
        if "sqrt" in query and numbers:
            return f"math.sqrt({numbers[0]})"
        elif "cbrt" in query and numbers:
            return f"math.pow({numbers[0]}, 1/3)"
        elif "sin" in query and numbers:
            return f"math.sin(math.radians({numbers[0]}))"
        elif "cos" in query and numbers:
            return f"math.cos(math.radians({numbers[0]}))"
        elif "tan" in query and numbers:
            return f"math.tan(math.radians({numbers[0]}))"
        elif "log" in query and numbers:
            return f"math.log10({numbers[0]})"
        
        return ""
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform calculation from the query"""
        expression = self.parse_expression(query)
        
        if not expression:
            return {
                "response": "I couldn't parse a mathematical expression from your query.",
                "source": "Calculator",
                "success": False
            }
        
        try:
            # Make the math module available for evaluation
            result = eval(expression, {"__builtins__": None}, {"math": math})
            
            # Format result
            if isinstance(result, float):
                # For small numbers close to an integer, convert to int
                if abs(result - round(result)) < 1e-10:
                    result = int(round(result))
                else:
                    # Limit decimals for readability
                    result = round(result, 10)
                    # Remove trailing zeros
                    result = float(f"{result:.10f}".rstrip("0").rstrip("."))
            
            return {
                "response": f"The result is {result}",
                "source": "Calculator",
                "success": True,
                "calculation": {
                    "expression": expression,
                    "result": result
                }
            }
        except Exception as e:
            return {
                "response": f"I couldn't calculate that: {str(e)}",
                "source": "Calculator",
                "success": False
            }


class NewsTool(Tool):
    """Tool for retrieving news information"""
    
    def __init__(self):
        super().__init__("News Tool", "Get the latest news information")
        self.api_key = os.getenv("NEWS_API_KEY")
        self.base_url = "https://newsapi.org/v2"
    
    def can_handle(self, query: str) -> bool:
        """Check if this tool can handle the query"""
        query = query.lower().strip()
        
        # News-related indicators
        news_indicators = ["news", "headline", "headlines", "article", "articles", 
                         "latest", "recent", "current", "today", "update", "updates",
                         "happening", "events", "information about"]
        
        # Check for news indicators
        for indicator in news_indicators:
            if indicator in query:
                return True
                
        # Check combined patterns
        combined_patterns = [
            r"what'?s (?:going on|happening) (?:with|in|about)",
            r"tell me about (?:the current|the latest|recent)",
            r"what (?:is|are) the latest"
        ]
        
        for pattern in combined_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Check for requests about specific news topics
        specific_topics = ["tariff", "trump", "biden", "china", "trade war", "election", 
                          "president", "politics", "economic", "economy", "russia", "ukraine", 
                          "market", "stocks", "inflation", "interest rate", "war", "climate", "ai"]
        
        for topic in specific_topics:
            # Check if the topic is in the query
            if topic in query:
                return True
            # Check for topic with news indicators
            for indicator in news_indicators:
                if f"{topic} {indicator}" in query or f"{indicator} {topic}" in query:
                    return True
        
        return False
    
    def extract_topic(self, query: str) -> str:
        """Extract news topic from query with improved handling of complex topics"""
        # Clean the query
        query = query.lower().strip()
        
        # Dictionary of specific topics to recognize
        specific_topics = {
            "tariff": "tariffs",
            "tariffs": "tariffs",
            "trump": "Trump",
            "donald trump": "Trump",
            "biden": "Biden",
            "joe biden": "Biden",
            "china": "China",
            "chinese": "China",
            "trade war": "trade war",
            "trade": "trade",
            "election": "election",
            "president": "president",
            "politics": "politics",
            "economic": "economy",
            "economy": "economy",
            "russia": "Russia",
            "ukraine": "Ukraine",
            "market": "stock market",
            "stocks": "stock market",
            "inflation": "inflation",
            "interest rate": "interest rates",
            "war": "war",
            "climate": "climate",
            "ai": "artificial intelligence",
            "tech": "technology"
        }
        
        # Important combinations with specific search terms
        important_combinations = [
            # Trump + tariffs/trade/China combinations
            (["trump", "tariff"], "Trump tariffs"),
            (["trump", "tariffs"], "Trump tariffs"),
            (["trump", "tariff", "china"], "Trump China tariffs"),
            (["trump", "tariffs", "china"], "Trump China tariffs"),
            (["china", "tariff"], "China tariffs"),
            (["china", "tariffs"], "China tariffs"),
            (["trade", "china"], "China trade"),
            (["trump", "china"], "Trump China relations"),
            (["trump", "trade"], "Trump trade policy"),
            (["trade", "war", "china"], "China trade war"),
            # Biden combinations
            (["biden", "economy"], "Biden economy"),
            (["biden", "inflation"], "Biden inflation"),
            (["biden", "china"], "Biden China policy"),
            # Election related
            (["trump", "election"], "Trump election"),
            (["biden", "election"], "Biden election"),
            (["2024", "election"], "2024 election"),
            # Global issues
            (["russia", "ukraine"], "Russia Ukraine conflict"),
        ]
        
        # Check for multi-keyword combinations
        for keywords, topic in important_combinations:
            if all(keyword in query for keyword in keywords):
                return topic
        
        # Check for single specific topics
        for topic_word, topic in specific_topics.items():
            if topic_word in query:
                return topic
        
        # Simple topic extraction patterns
        topic_patterns = [
            r"news (?:about|on|regarding) ([\w\s]+)(?:\?|$|\.)",
            r"(?:what's|what is) (?:the|) (?:latest|recent) (?:news|information) (?:about|on|regarding) ([\w\s]+)(?:\?|$|\.)",
            r"headlines (?:about|on|for) ([\w\s]+)(?:\?|$|\.)",
            r"(?:tell|inform) me about (?:the|) (?:latest|recent|current) ([\w\s]+)(?:\?|$|\.)",
            r"what (?:is|are) (?:the|) (?:latest|recent|current) ([\w\s]+)(?:\?|$|\.)"
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                # Check if the extracted topic contains any specific topics
                for keyword in specific_topics:
                    if keyword in topic:
                        return specific_topics[keyword]
                return topic
        
        # Extract meaningful words (removing common stopwords)
        stopwords = ["what", "when", "where", "who", "why", "how", "is", "are", "the", "a", "an", "i", "me", "my", 
                     "with", "to", "from", "in", "on", "at", "any", "some", "tell", "about", "know", "latest",
                     "recent", "current", "news", "information", "headlines", "articles"]
        
        # Get important words
        query_words = query.split()
        important_words = [word for word in query_words if word not in stopwords and len(word) > 2]
        
        # First check for important combinations directly from the important words
        for keywords, topic in important_combinations:
            if all(keyword in important_words for keyword in keywords):
                return topic
        
        # Look for specific topics in important words
        topic_words = []
        for word in important_words:
            if word in specific_topics:
                topic_words.append(word)
                
        if len(topic_words) >= 2:
            # Try to construct a meaningful topic from multiple detected topics
            if "trump" in topic_words and ("tariff" in topic_words or "tariffs" in topic_words):
                return "Trump tariffs"
            if "china" in topic_words and ("tariff" in topic_words or "tariffs" in topic_words):
                return "China tariffs"
            if "trump" in topic_words and "china" in topic_words:
                return "Trump China relations"
            
            # Generic combination of two topics
            return f"{specific_topics[topic_words[0]]} {specific_topics[topic_words[1]]}"
        elif len(topic_words) == 1:
            # Return the single topic
            return specific_topics[topic_words[0]]
        
        # If we have important words but no specific topics
        if important_words:
            # Join the most meaningful 2-3 words
            candidate = " ".join(important_words[:min(3, len(important_words))])
            if len(candidate) > 3:  # Ensure we have something meaningful
                return candidate
        
        # Default fallback
        return "general"
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get news information"""
        if not self.api_key:
            return {
                "response": "News information is unavailable. NewsAPI key is missing.",
                "source": "News Tool",
                "success": False
            }
        
        topic = kwargs.get("topic") or self.extract_topic(query)
        print(f"Searching news for topic: {topic}")
        logging.info(f"News search topic extracted: {topic} for query: {query}")
        
        try:
            # For Trump tariffs or China tariffs, use a more specific search
            if topic in ["Trump tariffs", "China tariffs", "Trump China tariffs"]:
                search_term = topic.replace(" ", " AND ")
            else:
                search_term = topic
                
            # Get top headlines for the topic
            url = f"{self.base_url}/top-headlines"
            params = {
                "apiKey": self.api_key,
                "q": search_term if topic != "general" else "",
                "language": "en",
                "pageSize": 5
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get("articles", [])
            
            if not articles:
                # Try searching for everything if headlines didn't work
                url = f"{self.base_url}/everything"
                params = {
                    "apiKey": self.api_key,
                    "q": search_term,
                    "language": "en",
                    "pageSize": 5,
                    "sortBy": "relevancy"
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles", [])
            
            if not articles:
                # Try even broader search with simpler terms for specific topics
                if "Trump" in topic and "tariffs" in topic:
                    url = f"{self.base_url}/everything"
                    params = {
                        "apiKey": self.api_key,
                        "q": "Trump AND tariffs",
                        "language": "en",
                        "pageSize": 5,
                        "sortBy": "relevancy"
                    }
                    
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    articles = data.get("articles", [])
            
            if not articles:
                return {
                    "response": f"I couldn't find any recent news about {topic}.",
                    "source": "News Tool",
                    "success": False
                }
            
            # Format the result
            result = f"Here are the latest headlines about {topic}:\n\n"
            
            for i, article in enumerate(articles[:3], 1):
                title = article.get("title", "").split(" - ")[0]  # Remove source from title
                source = article.get("source", {}).get("name", "Unknown Source")
                url = article.get("url", "")
                published = article.get("publishedAt", "")
                if published:
                    try:
                        from datetime import datetime
                        pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        date_str = pub_date.strftime("%b %d, %Y")
                    except:
                        date_str = ""
                else:
                    date_str = ""
                
                result += f"{i}. {title} ({source}"
                if date_str:
                    result += f", {date_str}"
                result += f")\n"
            
            return {
                "response": result.strip(),
                "source": "NewsAPI",
                "success": True,
                "data": articles[:3],
                "query_topic": topic  # Include the extracted topic for validation
            }
            
        except Exception as e:
            logging.error(f"News tool error: {str(e)}")
            return {
                "response": f"Sorry, I couldn't get news about {topic}: {str(e)}",
                "source": "News Tool", 
                "success": False
            }


class ToolRegistry:
    """Registry of available tools for TARS"""
    
    def __init__(self):
        self.tools = []
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools.append(tool)
    
    def get_tool_for_query(self, query: str) -> Optional[Tool]:
        """Find the appropriate tool for a query"""
        for tool in self.tools:
            if tool.can_handle(query):
                return tool
        return None
    
    def execute_tool(self, query: str, **kwargs) -> Dict[str, Any]:
        """Find and execute the appropriate tool for a query"""
        tool = kwargs.get("tool") or self.get_tool_for_query(query)
        
        if not tool:
            return {
                "result": None,
                "source": None,
                "success": False,
                "message": "No suitable tool found for this query"
            }
        
        return tool.execute(query, **kwargs)


# Initialize the default tools
def create_default_registry() -> ToolRegistry:
    """Create and initialize the default tool registry"""
    registry = ToolRegistry()
    
    # Register available tools
    registry.register_tool(WeatherTool())
    registry.register_tool(CalculatorTool())
    registry.register_tool(NewsTool())
    
    return registry


# Singleton registry instance
_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    global _registry
    if _registry is None:
        _registry = create_default_registry()
    return _registry 