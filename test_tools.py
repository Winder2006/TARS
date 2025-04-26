#!/usr/bin/env python3
"""
Test script for TARS tools

This script tests each tool individually to verify they're working properly.
"""

import os
import logging
import json
from dotenv import load_dotenv
from tools import get_tool_registry

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def test_weather_tool():
    """Test the weather tool functionality"""
    print("\n=== Testing Weather Tool ===")
    
    registry = get_tool_registry()
    weather_tool = None
    
    # Find the weather tool
    for tool in registry.tools:
        if tool.name == "Weather Tool":
            weather_tool = tool
            break
    
    if not weather_tool:
        print("Weather Tool not found in registry!")
        return False
    
    # Check if API key is available
    api_key = os.getenv("OPENWEATHER_API_KEY")
    print(f"Weather API key present: {bool(api_key)}")
    
    # Test with a simple query
    queries = [
        "What's the weather in New York?",
        "How's the weather today in London?",
        "Temperature in Paris",
        "Weather forecast for San Francisco"
    ]
    
    success = False
    for query in queries:
        print(f"\nTesting query: '{query}'")
        
        # Check if tool can handle the query
        can_handle = weather_tool.can_handle(query)
        print(f"Can handle: {can_handle}")
        
        if can_handle:
            # Extract location
            location = weather_tool.extract_location(query)
            print(f"Extracted location: '{location}'")
            
            # Execute the query
            result = weather_tool.execute(query)
            print(f"Result: {json.dumps(result, indent=2)}")
            
            if result.get("success", False):
                success = True
                break
    
    return success

def test_news_tool():
    """Test the news tool functionality"""
    print("\n=== Testing News Tool ===")
    
    registry = get_tool_registry()
    news_tool = None
    
    # Find the news tool
    for tool in registry.tools:
        if tool.name == "News Tool":
            news_tool = tool
            break
    
    if not news_tool:
        print("News Tool not found in registry!")
        return False
    
    # Check if API key is available
    api_key = os.getenv("NEWS_API_KEY")
    print(f"News API key present: {bool(api_key)}")
    
    # Test with a simple query
    queries = [
        "What's the latest news?",
        "News about Trump tariffs",
        "Latest headlines about Ukraine",
        "Recent news about technology"
    ]
    
    success = False
    for query in queries:
        print(f"\nTesting query: '{query}'")
        
        # Check if tool can handle the query
        can_handle = news_tool.can_handle(query)
        print(f"Can handle: {can_handle}")
        
        if can_handle:
            # Extract topic
            topic = news_tool.extract_topic(query)
            print(f"Extracted topic: '{topic}'")
            
            # Execute the query
            result = news_tool.execute(query)
            print(f"Result success: {result.get('success', False)}")
            
            if result.get("success", False):
                success = True
                print(f"Response excerpt: {result.get('response', '')[:100]}...")
                break
    
    return success

def test_calculator_tool():
    """Test the calculator tool functionality"""
    print("\n=== Testing Calculator Tool ===")
    
    registry = get_tool_registry()
    calc_tool = None
    
    # Find the calculator tool
    for tool in registry.tools:
        if tool.name == "Calculator":
            calc_tool = tool
            break
    
    if not calc_tool:
        print("Calculator Tool not found in registry!")
        return False
    
    # Test with a simple query
    queries = [
        "What is 2 + 2?",
        "Calculate 15 * 7",
        "7 divided by 3",
        "Square root of 16"
    ]
    
    success = False
    for query in queries:
        print(f"\nTesting query: '{query}'")
        
        # Check if tool can handle the query
        can_handle = calc_tool.can_handle(query)
        print(f"Can handle: {can_handle}")
        
        if can_handle:
            # Parse expression
            expression = calc_tool.parse_expression(query)
            print(f"Parsed expression: '{expression}'")
            
            # Execute the query
            result = calc_tool.execute(query)
            print(f"Result: {json.dumps(result, indent=2)}")
            
            if result.get("success", False):
                success = True
                break
    
    return success

def main():
    """Run tests for all tools"""
    print("Starting TARS tools test...")
    
    # Test the weather tool
    weather_success = test_weather_tool()
    print(f"\nWeather tool test {'PASSED' if weather_success else 'FAILED'}")
    
    # Test the news tool
    news_success = test_news_tool()
    print(f"\nNews tool test {'PASSED' if news_success else 'FAILED'}")
    
    # Test the calculator tool
    calc_success = test_calculator_tool()
    print(f"\nCalculator tool test {'PASSED' if calc_success else 'FAILED'}")
    
    # Overall status
    if weather_success and news_success and calc_success:
        print("\n✅ All tools are working correctly!")
    else:
        print("\n❌ Some tools failed - check the output for details")

if __name__ == "__main__":
    main() 