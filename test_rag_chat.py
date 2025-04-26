#!/usr/bin/env python3
"""
Test script for RAG-enhanced chat responses

This script demonstrates how to use the RAG system to enhance
conversation responses with relevant information.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG utilities from chat.py
try:
    from chat import init_rag_system, save_rag_index, get_rag_enhanced_response
except ImportError:
    print("Error: Could not import RAG system from chat.py")
    sys.exit(1)

def test_rag_chat():
    """Test the RAG system for enhancing chat responses"""
    print("Initializing RAG system...")
    
    # Initialize the RAG system
    rag = init_rag_system()
    
    # Add some documents to the system if it's empty
    if len(rag.document_store) == 0:
        print("Adding some sample documents to the RAG system...")
        documents = [
            {
                "content": "TARS is an advanced AI assistant based on the robot from the movie Interstellar. "
                         "TARS features voice recognition, personalized responses, and a humor setting that's "
                         "configurable from 0-100%. The default humor setting is 75%.",
                "source": "TARS Documentation"
            },
            {
                "content": "The voice recognition system in TARS uses a Gaussian Mixture Model for speaker "
                         "identification. This enables TARS to recognize different users by their voice "
                         "patterns and provide personalized responses based on user history.",
                "source": "Technical Specifications"
            },
            {
                "content": "TARS can search the web for real-time information using both DuckDuckGo and Bing "
                         "search engines. This functionality enables TARS to provide current information about "
                         "weather, news, and other time-sensitive topics.",
                "source": "Feature Documentation"
            }
        ]
        
        for doc in documents:
            rag.add_document(content=doc["content"], source=doc["source"])
        print(f"Added {len(documents)} documents to the RAG system")
    else:
        print(f"Found {len(rag.document_store)} existing documents in the RAG system")
    
    # Start interactive chat
    print("\n=== RAG-Enhanced Chat Demo ===")
    print("Type your questions below. Type 'exit' or 'quit' to end the demo.")
    print("The RAG system will enhance responses with relevant information from its knowledge base.")
    
    # Keep track of conversation history
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Get enhanced response
        response = get_rag_enhanced_response(user_input, conversation_history)
        
        # Print response
        print(f"TARS: {response}")
        
        # Update conversation history
        conversation_history.append({"user": user_input})
        conversation_history.append({"assistant": response})
        
        # Keep conversation history manageable
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
    
    # Save the RAG index before exiting
    print("\nSaving RAG index...")
    if save_rag_index():
        print("RAG index saved successfully")
    else:
        print("Failed to save RAG index")
    
    print("Demo completed. Goodbye!")

if __name__ == "__main__":
    test_rag_chat() 