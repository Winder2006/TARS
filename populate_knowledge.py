#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('populate_knowledge')

# Import knowledge graph functions
try:
    from knowledge_graph import (
        add_entity_to_graph, 
        add_fact_to_graph, 
        get_knowledge_graph,
        get_entity_by_name,
        get_user_facts
    )
except ImportError as e:
    logger.error(f"Error importing knowledge_graph module: {e}")
    print(f"Error: Could not import knowledge_graph module. Please ensure it exists.")
    sys.exit(1)

def create_directories():
    """Create necessary directories"""
    Path("memory").mkdir(exist_ok=True)

def populate_user_data(user_id="default_user", user_name="Charles"):
    """
    Populate the knowledge graph with user data
    
    Args:
        user_id: User ID to use
        user_name: User's name
    """
    try:
        logger.info(f"Populating knowledge graph for user {user_name}")
        
        # Create user entity if it doesn't exist
        user_entity_id = f"user_{user_id}"
        logger.info(f"Adding user entity: {user_entity_id}")
        
        # Add user entity
        add_entity_to_graph(
            entity_id=user_entity_id,
            entity_type="person",
            properties={
                "name": user_name,
                "user_id": user_id
            },
            aliases=[user_name, user_id]
        )
        
        # Add school information
        school_id = "school_marquette"
        add_entity_to_graph(
            entity_id=school_id,
            entity_type="school",
            properties={
                "name": "Marquette University",
                "location": "Milwaukee, Wisconsin",
                "type": "University"
            },
            aliases=["Marquette", "Marquette University"]
        )
        
        # Add fact that user attends this school
        add_fact_to_graph(
            subject=user_entity_id,
            predicate="attends",
            object=school_id,
            confidence=1.0,
            source="user_input"
        )
        
        # Add some preferences
        add_preferences(user_entity_id)
        
        # Print confirmation
        logger.info("Successfully populated knowledge graph")
        
        # Show what was added
        facts = get_user_facts(user_entity_id)
        if facts:
            print(f"\nAdded facts about {user_name}:")
            for fact in facts:
                predicate = fact.get("predicate", "").replace("_", " ")
                object_name = fact.get("object_name", "")
                print(f"- {user_name} {predicate} {object_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error populating knowledge graph: {e}")
        print(f"Error: {e}")
        return False

def add_preferences(user_entity_id):
    """Add some preferences for the user"""
    
    # Add some favorite things
    preferences = [
        {"type": "music_genre", "name": "Jazz", "predicate": "likes"},
        {"type": "movie", "name": "Interstellar", "predicate": "loves"},
        {"type": "food", "name": "Pizza", "predicate": "enjoys"},
        {"type": "hobby", "name": "Programming", "predicate": "enjoys"},
        {"type": "subject", "name": "Computer Science", "predicate": "studies"}
    ]
    
    for pref in preferences:
        # Create entity for the preference
        pref_id = f"{pref['type']}_{pref['name'].lower().replace(' ', '_')}"
        
        # Add entity
        add_entity_to_graph(
            entity_id=pref_id,
            entity_type=pref["type"],
            properties={"name": pref["name"]},
            aliases=[pref["name"]]
        )
        
        # Add fact
        add_fact_to_graph(
            subject=user_entity_id,
            predicate=pref["predicate"],
            object=pref_id,
            confidence=0.9,
            source="user_profile"
        )

def main():
    """Main function"""
    create_directories()
    
    # Get user information
    user_id = input("Enter user ID (default: default_user): ").strip() or "default_user"
    user_name = input("Enter user name (default: Charles): ").strip() or "Charles"
    
    # Populate knowledge graph
    populate_user_data(user_id, user_name)
    
    print("\nKnowledge graph populated successfully!")
    print("You can now use TARS with memory of your school and preferences.")

if __name__ == "__main__":
    main() 