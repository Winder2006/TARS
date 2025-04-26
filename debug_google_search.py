#!/usr/bin/env python3
import os
import requests
import json
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def debug_google_search():
    """Test the Google Search API connection and show detailed error information"""
    # Get API key and CSE ID from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    logger.info(f"API Key present: {bool(api_key)}")
    logger.info(f"CSE ID present: {bool(cse_id)}")
    
    if not api_key or not cse_id:
        logger.error("Error: API key or CSE ID not found in environment variables.")
        return False
    
    # Mask the API key for security when logging
    masked_key = f"{api_key[:5]}...{api_key[-5:]}" if api_key else "None"
    logger.info(f"Using API key: {masked_key}")
    logger.info(f"Using CSE ID: {cse_id}")
    
    # Define the API endpoint and parameters
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': 'test query',
        'key': api_key,
        'cx': cse_id,
        'num': 3  # Number of results to return
    }
    
    try:
        # Make the API request
        logger.info("Sending request to Google API...")
        response = requests.get(url, params=params)
        
        # Print full response for debugging
        logger.info(f"Response status code: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            logger.info("API Connection Successful!")
            
            if 'items' in data:
                logger.info(f"Found {len(data['items'])} search results")
                
                # Print the first result
                if len(data['items']) > 0:
                    first_result = data['items'][0]
                    logger.info("First result:")
                    logger.info(f"Title: {first_result.get('title', 'N/A')}")
                    logger.info(f"Link: {first_result.get('link', 'N/A')}")
                    logger.info(f"Snippet: {first_result.get('snippet', 'N/A')}")
            else:
                logger.info("No search results found, but API connection is working.")
                logger.info(f"Full response: {json.dumps(data, indent=2)}")
            
            return True
        else:
            logger.error(f"API Connection Failed with status code: {response.status_code}")
            try:
                error_data = response.json()
                logger.error(f"Error details: {json.dumps(error_data, indent=2)}")
                
                # Check for specific error codes
                if 'error' in error_data:
                    error_info = error_data['error']
                    if 'code' in error_info:
                        error_code = error_info['code']
                        
                        if error_code == 403:
                            logger.error("Authorization error (403): This could be due to:")
                            logger.error("1. The API key is invalid or has expired")
                            logger.error("2. The API hasn't been enabled for this project")
                            logger.error("3. You've exceeded your quota")
                            logger.error("Visit https://console.cloud.google.com/apis/api/customsearch.googleapis.com/overview to check API status")
                        
                        if 'message' in error_info:
                            logger.error(f"Error message: {error_info['message']}")
            except:
                logger.error(f"Error response: {response.text}")
            
            return False
    
    except Exception as e:
        logger.error(f"Error connecting to Google API: {str(e)}")
        return False


if __name__ == "__main__":
    print("Debug Google Search API Connection")
    print("==================================")
    debug_google_search() 