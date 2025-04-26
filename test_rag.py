#!/usr/bin/env python3
"""
Test script for RAG system with FAISS vector search

This script demonstrates the following capabilities:
1. Initializing the RAG system
2. Adding documents to the vector database
3. Searching for relevant information
4. Saving and loading the index
"""

import os
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG system from chat.py
try:
    from chat import init_rag_system, save_rag_index
except ImportError:
    print("Error: Could not import RAG system from chat.py")
    sys.exit(1)

def test_rag_system():
    """Test the RAG system with sample documents"""
    print("Initializing RAG system...")
    
    # Initialize the RAG system
    rag = init_rag_system()
    
    # Sample documents to add
    documents = [
        {
            "content": "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. "
                    "It is the largest optical telescope in space and its high resolution and sensitivity allow it to view "
                    "objects too old, distant, or faint for the Hubble Space Telescope.",
            "source": "Astronomy Facts"
        },
        {
            "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes "
                    "code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
            "source": "Programming Languages"
        },
        {
            "content": "Quantum computing is a type of computation that harnesses the collective properties of quantum states, "
                    "such as superposition, interference, and entanglement, to perform calculations. Quantum computers are believed "
                    "to be able to solve certain computational problems, such as integer factorization, substantially faster than classical computers.",
            "source": "Quantum Computing Research"
        },
        {
            "content": "Machine learning (ML) is a field of study in artificial intelligence concerned with the development "
                    "and study of statistical algorithms that can learn from data and generalize to unseen data, and thus "
                    "perform tasks without explicit instructions.",
            "source": "AI Reference"
        },
        {
            "content": "The climate crisis is an ongoing anthropogenic climate change due to excessive greenhouse gas emissions. "
                    "Global warming is the observed long-term heating of Earth's climate system. Climate mitigation involves "
                    "reducing emissions of greenhouse gases and removing them from the atmosphere.",
            "source": "Climate Science"
        }
    ]
    
    # Check if we already have documents in the system
    if len(rag.document_store) > 0:
        print(f"Found {len(rag.document_store)} existing documents in the RAG system")
    else:
        print("Adding sample documents to the RAG system...")
        for i, doc in enumerate(documents):
            print(f"Adding document {i+1}/{len(documents)}...")
            rag.add_document(content=doc["content"], source=doc["source"])
        
        print(f"Added {len(documents)} documents to the RAG system")
    
    # Test retrieval
    print("\nTesting document retrieval...")
    
    # Sample queries to test
    queries = [
        "Tell me about space telescopes",
        "What is Python programming?",
        "Explain quantum computing",
        "How does machine learning work?",
        "What is the climate crisis?"
    ]
    
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'")
        results = rag.retrieve(query)
        
        if results:
            print(f"Found {len(results)} relevant documents:")
            for j, doc in enumerate(results):
                print(f"  {j+1}. {doc['content'][:100]}... (Score: {doc['score']:.2f})")
        else:
            print("No relevant documents found")
    
    # Save the index
    print("\nSaving RAG index...")
    if save_rag_index():
        print("RAG index saved successfully")
    else:
        print("Failed to save RAG index")

if __name__ == "__main__":
    test_rag_system() 