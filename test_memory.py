"""
Test script for the enhanced memory system
"""

import os
import json
from knowledge_db import KnowledgeDatabase, EnhancedMemory

def main():
    print("Testing enhanced memory system...")
    
    # Initialize database
    db = KnowledgeDatabase()
    print(f"Initialized database: {db.db_path}")
    
    # Add some test facts
    user_id = "test_user"
    print(f"Adding facts for user: {user_id}")
    
    fact1 = db.add_user_fact(
        user_id=user_id,
        fact="likes pizza",
        topic="food",
        source="test",
    )
    
    fact2 = db.add_user_fact(
        user_id=user_id,
        fact="lives in New York",
        topic="location",
        source="test",
    )
    
    print("Added facts, now testing retrieval...")
    
    # Test retrieving facts by topic
    topic_facts = db.get_user_facts_by_topic(user_id, "food", limit=5)
    print(f"Facts about food: {topic_facts}")
    
    # Test retrieving recent facts
    recent_facts = db.get_recent_user_facts(user_id, limit=5)
    print(f"Recent facts: {recent_facts}")
    
    # Test fact extraction
    print("\nTesting fact extraction...")
    user_message = "I really enjoy playing guitar and I've been doing it for 10 years"
    ai_response = "That's impressive! Playing an instrument for a decade shows real dedication."
    
    db.extract_and_store_facts(
        user_id=user_id,
        user_message=user_message,
        ai_response=ai_response,
        conversation_id="test123"
    )
    
    # Check if new facts were added
    print("Checking for new facts after extraction...")
    updated_facts = db.get_recent_user_facts(user_id, limit=10)
    print(f"Updated facts: {json.dumps(updated_facts, indent=2)}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main() 