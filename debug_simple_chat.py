#!/usr/bin/env python3
import os
import sys
import logging
import re
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

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
    
    return False

def google_search(query, api_key, cse_id, num_results=3):
    """
    Perform a Google search using the Custom Search API.
    
    Args:
        query (str): The search query
        api_key (str): Google API key
        cse_id (str): Google Custom Search Engine ID
        num_results (int): Number of results to return
        
    Returns:
        list: List of search result snippets
    """
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': api_key,
            'cx': cse_id,
            'num': min(num_results, 10)  # Google CSE max is 10
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        search_results = response.json()
        
        # Extract snippets from results
        snippets = []
        if 'items' in search_results:
            for item in search_results['items']:
                if 'snippet' in item:
                    # Format the snippet with title
                    snippet = f"{item['title']}: {item['snippet']}"
                    snippets.append(snippet)
        
        return snippets
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in Google search: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error in Google search: {str(e)}")
        return []

def simple_chat():
    """Simple chat interface to test web search functionality"""
    print("\n" + "=" * 50)
    print("TARS Web Search Test")
    print("Type 'exit' to quit")
    print("=" * 50)
    
    # Check Google API credentials
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        print("Warning: Google API credentials not found in .env file")
        print("API Key present:", bool(api_key))
        print("CSE ID present:", bool(cse_id))
    
    # Main chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        # Analyze query to see if it should trigger web search
        is_factual = is_factual_question(user_input)
        should_search = should_search_web(user_input)
        
        print(f"Is factual question: {is_factual}")
        print(f"Should search web: {should_search}")
        
        # Perform web search if appropriate
        if is_factual or should_search:
            if api_key and cse_id:
                print("\nSearching the web...")
                results = google_search(user_input, api_key, cse_id)
                
                if results:
                    print("\nHere's what I found:")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result}")
                else:
                    print("I couldn't find anything specific about that.")
            else:
                print("Cannot perform web search: Missing API credentials")
        else:
            print("\nThis query doesn't need web search.")

if __name__ == "__main__":
    simple_chat() 