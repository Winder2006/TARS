#!/usr/bin/env python3
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_google_search():
    # Get API key and CSE ID from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        print("Error: API key or CSE ID not found in environment variables.")
        return False
    
    print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    print(f"Using CSE ID: {cse_id}")
    
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
        print("Sending request to Google API...")
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print("\nAPI Connection Successful!")
            if 'items' in data:
                print(f"Found {len(data['items'])} search results")
                
                # Print the first result
                if len(data['items']) > 0:
                    first_result = data['items'][0]
                    print("\nFirst result:")
                    print(f"Title: {first_result.get('title', 'N/A')}")
                    print(f"Link: {first_result.get('link', 'N/A')}")
                    print(f"Snippet: {first_result.get('snippet', 'N/A')}")
            else:
                print("No search results found, but API connection is working.")
            
            return True
        else:
            print(f"\nAPI Connection Failed with status code: {response.status_code}")
            print(f"Error message: {response.text}")
            return False
    
    except Exception as e:
        print(f"\nError connecting to Google API: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Google Custom Search API connection...")
    test_google_search() 