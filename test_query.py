#!/usr/bin/env python3
import os
import sys
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_queries")

# Load environment variables from .env file
load_dotenv()

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_query(query):
    """Test if a query would trigger web search"""
    print("\n" + "=" * 50)
    print(f"Testing query: '{query}'")
    print("=" * 50)
    
    # Import functions from chat.py
    from chat import is_factual_question, should_search_web
    
    # Check if query is factual
    is_factual = is_factual_question(query)
    print(f"Is factual question: {is_factual}")
    
    # Check if query should trigger web search
    should_search = should_search_web(query)
    print(f"Should search web: {should_search}")
    
    # If either is true, test the web search
    if is_factual or should_search:
        try:
            from web_search import search_web
            print(f"\nAttempting web search for: '{query}'")
            
            # Check if credentials are set
            api_key = os.getenv("GOOGLE_API_KEY")
            cse_id = os.getenv("GOOGLE_CSE_ID")
            print(f"API Key present: {bool(api_key)}")
            print(f"CSE ID present: {bool(cse_id)}")
            
            if api_key and cse_id:
                # Perform web search
                results = search_web(query)
                print(f"\nSearch results ({len(results)} found):")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result}")
            else:
                print("Cannot perform web search: Missing API credentials")
        except Exception as e:
            print(f"Error performing web search: {str(e)}")
    else:
        print("\nQuery would not trigger web search")

if __name__ == "__main__":
    # Test various types of queries
    test_queries = [
        # Original test cases
        "Who is the current president of the United States?",  # Leadership query
        "What is the weather like today?",  # Current info
        "Tell me about the history of Rome",  # Factual question
        "What's your name?",  # Personal question about assistant
        "Do I like chocolate?",  # Personal question about user
        "voice list",  # Voice command
        "What are the latest developments in the Ukraine conflict?",  # Current events
        "When did Albert Einstein die?",  # Historical fact
        "What's the latest iPhone model?",  # Product info
        "Write me a poem about spring",  # Creative request
        
        # Additional edge cases
        "Should I invest in Apple stock?",  # Personal decision but with factual component
        "What's the best chocolate brand?",  # General preference question
        "Do people like chocolate?",  # General preference, not personal
        "What did I eat yesterday?",  # Personal memory question
        "What is my height?",  # Personal attribute
        "What is the average height of adults?",  # Statistical fact
        "Where is my phone?",  # Personal question
        "Where is the Eiffel Tower?",  # Geographic fact
        "Can you remember what I told you about my job?",  # Personal memory recall
        "Can you explain quantum physics?",  # Factual explanation request
    ]
    
    for query in test_queries:
        test_query(query)
        print("\n" + "-" * 50) 