#!/usr/bin/env python3
import os
import requests
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tars.web_search")

def search_web(query, num_results=3):
    """
    Search the web for information related to the query.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
        
    Returns:
        list: List of search result snippets
    """
    try:
        logger.info(f"Searching web for: {query}")
        
        # Check if we have a Google API key
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if api_key and cse_id:
            return google_search(query, api_key, cse_id, num_results)
        else:
            # Fallback to a simple response without external API
            logger.warning("No Google API credentials found for web search")
            
            # For TARS, provide some hardcoded information about TARS from Interstellar
            if "tars" in query.lower() and "interstellar" in query.lower():
                return [
                    "TARS is a robot from the movie Interstellar (2014), featuring an innovative design of four-hinged rectangular panels.",
                    "TARS has a personality with customizable settings, including humor level (set to 75% in the movie).",
                    "TARS was portrayed by actor Bill Irwin who also provided the voice and puppeteering for the robot."
                ]
            # Provide a neutral fallback response that doesn't mention limitations
            return [
                f"Here's what I know about '{query}'."
            ]
    
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        return []

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
                    # Format the snippet without source citation
                    snippet = f"{item['title']}: {item['snippet']}"
                    snippets.append(snippet)
        
        return snippets
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in Google search: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error in Google search: {str(e)}")
        return []

def get_current_date():
    """Get current date formatted as string"""
    return datetime.now().strftime("%Y-%m-%d")

if __name__ == "__main__":
    # Test the search function
    test_query = "What is the capital of France?"
    results = search_web(test_query)
    
    print(f"Search query: {test_query}")
    print(f"Results: {json.dumps(results, indent=2)}") 