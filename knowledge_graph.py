import logging
import json
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tars.knowledge_graph")

class KnowledgeGraph:
    """
    A graph-based knowledge representation system for organizing and 
    connecting memories, facts, and concepts.
    """
    
    def __init__(self, storage_path: str = "knowledge_graph.json"):
        """
        Initialize the knowledge graph
        
        Args:
            storage_path: Path to store the knowledge graph data
        """
        self.storage_path = storage_path
        self.graph = nx.DiGraph()
        self.node_types = set()
        self.relation_types = set()
        self.entity_aliases = defaultdict(set)
        self.load_data()
        
    def load_data(self):
        """Load existing knowledge graph data"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Rebuild the graph
                for node in data.get("nodes", []):
                    self.graph.add_node(
                        node["id"],
                        type=node["type"],
                        properties=node["properties"],
                        created=node.get("created", ""),
                        updated=node.get("updated", "")
                    )
                    
                    # Rebuild type tracking
                    self.node_types.add(node["type"])
                    
                    # Rebuild aliases
                    aliases = node.get("aliases", [])
                    if aliases:
                        for alias in aliases:
                            self.entity_aliases[alias].add(node["id"])
                
                # Rebuild edges
                for edge in data.get("edges", []):
                    self.graph.add_edge(
                        edge["source"],
                        edge["target"],
                        type=edge["type"],
                        properties=edge["properties"],
                        weight=edge.get("weight", 1.0),
                        created=edge.get("created", ""),
                        updated=edge.get("updated", "")
                    )
                    
                    # Track relation types
                    self.relation_types.add(edge["type"])
                    
                logger.info(f"Loaded knowledge graph from {self.storage_path}")
                logger.info(f"Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
                
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {str(e)}")
            # Keep the default empty graph
            
    def save_data(self):
        """Save knowledge graph to storage"""
        try:
            # Convert graph to serializable format
            nodes = []
            for node_id, node_data in self.graph.nodes(data=True):
                node_dict = {
                    "id": node_id,
                    "type": node_data["type"],
                    "properties": node_data["properties"],
                    "created": node_data.get("created", ""),
                    "updated": node_data.get("updated", "")
                }
                
                # Add aliases if any
                aliases = [alias for alias, node_set in self.entity_aliases.items() 
                          if node_id in node_set]
                if aliases:
                    node_dict["aliases"] = aliases
                    
                nodes.append(node_dict)
                
            edges = []
            for source, target, edge_data in self.graph.edges(data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    "type": edge_data["type"],
                    "properties": edge_data["properties"],
                    "weight": edge_data.get("weight", 1.0),
                    "created": edge_data.get("created", ""),
                    "updated": edge_data.get("updated", "")
                })
                
            data = {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "node_count": self.graph.number_of_nodes(),
                    "edge_count": self.graph.number_of_edges(),
                    "node_types": list(self.node_types),
                    "relation_types": list(self.relation_types),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved knowledge graph to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
            
    def add_entity(self, 
                  entity_id: str, 
                  entity_type: str, 
                  properties: Dict[str, Any],
                  aliases: List[str] = None) -> str:
        """
        Add a new entity node to the graph
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity (person, place, concept, etc.)
            properties: Dictionary of entity properties
            aliases: Alternative names for this entity
            
        Returns:
            The ID of the created entity
        """
        # Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # Check if node already exists
        if self.graph.has_node(entity_id):
            # Update existing node
            self.graph.nodes[entity_id]["type"] = entity_type
            self.graph.nodes[entity_id]["properties"].update(properties)
            self.graph.nodes[entity_id]["updated"] = timestamp
            logger.info(f"Updated existing entity: {entity_id}")
        else:
            # Add new node
            self.graph.add_node(
                entity_id,
                type=entity_type,
                properties=properties,
                created=timestamp,
                updated=timestamp
            )
            logger.info(f"Added new entity: {entity_id} of type {entity_type}")
            
        # Track node type
        self.node_types.add(entity_type)
        
        # Add aliases if provided
        if aliases:
            for alias in aliases:
                self.entity_aliases[alias].add(entity_id)
                
        self.save_data()
        return entity_id
        
    def add_relation(self, 
                    source_id: str, 
                    target_id: str, 
                    relation_type: str, 
                    properties: Dict[str, Any] = None,
                    weight: float = 1.0) -> bool:
        """
        Add a relation between two entities
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relation_type: Type of relation
            properties: Properties of the relation
            weight: Weight/strength of the relation (0.0-1.0)
            
        Returns:
            True if relation was added successfully
        """
        # Verify nodes exist
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.error(f"Cannot add relation: one or both nodes don't exist ({source_id}, {target_id})")
            return False
            
        # Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # Initialize properties if None
        if properties is None:
            properties = {}
            
        # Add or update edge
        if self.graph.has_edge(source_id, target_id):
            # Update existing edge
            self.graph[source_id][target_id]["type"] = relation_type
            self.graph[source_id][target_id]["properties"].update(properties)
            self.graph[source_id][target_id]["weight"] = weight
            self.graph[source_id][target_id]["updated"] = timestamp
            logger.info(f"Updated relation between {source_id} and {target_id}")
        else:
            # Add new edge
            self.graph.add_edge(
                source_id,
                target_id,
                type=relation_type,
                properties=properties,
                weight=weight,
                created=timestamp,
                updated=timestamp
            )
            logger.info(f"Added new relation: {source_id} --[{relation_type}]--> {target_id}")
            
        # Track relation type
        self.relation_types.add(relation_type)
        
        self.save_data()
        return True
        
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            Entity data as dictionary or None if not found
        """
        if not self.graph.has_node(entity_id):
            return None
            
        node_data = self.graph.nodes[entity_id]
        
        # Get aliases
        aliases = [alias for alias, node_set in self.entity_aliases.items() 
                  if entity_id in node_set]
                  
        # Build outgoing and incoming relationships
        outgoing_relations = []
        for _, target, edge_data in self.graph.out_edges(entity_id, data=True):
            outgoing_relations.append({
                "relation": edge_data["type"],
                "target": target,
                "target_type": self.graph.nodes[target]["type"],
                "properties": edge_data["properties"],
                "weight": edge_data.get("weight", 1.0)
            })
            
        incoming_relations = []
        for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
            incoming_relations.append({
                "relation": edge_data["type"],
                "source": source,
                "source_type": self.graph.nodes[source]["type"],
                "properties": edge_data["properties"],
                "weight": edge_data.get("weight", 1.0)
            })
            
        return {
            "id": entity_id,
            "type": node_data["type"],
            "properties": node_data["properties"],
            "aliases": aliases,
            "outgoing_relations": outgoing_relations,
            "incoming_relations": incoming_relations,
            "created": node_data.get("created", ""),
            "updated": node_data.get("updated", "")
        }
        
    def find_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find entities by type
        
        Args:
            entity_type: Type of entities to find
            limit: Maximum number of entities to return
            
        Returns:
            List of matching entities
        """
        matching_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data["type"] == entity_type:
                matching_nodes.append({
                    "id": node_id,
                    "type": node_data["type"],
                    "properties": node_data["properties"]
                })
                
                if len(matching_nodes) >= limit:
                    break
                    
        return matching_nodes
        
    def find_entities_by_property(self, 
                                 property_name: str, 
                                 property_value: Any,
                                 entity_type: str = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find entities by property value
        
        Args:
            property_name: Name of the property to match
            property_value: Value to match against
            entity_type: Optional type filter
            limit: Maximum number of entities to return
            
        Returns:
            List of matching entities
        """
        matching_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            # Check type constraint if specified
            if entity_type and node_data["type"] != entity_type:
                continue
                
            # Check property value
            if property_name in node_data["properties"] and node_data["properties"][property_name] == property_value:
                matching_nodes.append({
                    "id": node_id,
                    "type": node_data["type"],
                    "properties": node_data["properties"]
                })
                
                if len(matching_nodes) >= limit:
                    break
                    
        return matching_nodes
        
    def find_path(self, 
                 source_id: str, 
                 target_id: str, 
                 max_length: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path between two entities
        
        Args:
            source_id: Starting entity ID
            target_id: Target entity ID
            max_length: Maximum path length to consider
            
        Returns:
            List of nodes and edges in the path, or None if no path exists
        """
        # Check if both nodes exist
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return None
            
        try:
            # Find shortest path
            path_nodes = nx.shortest_path(
                self.graph, 
                source=source_id, 
                target=target_id,
                weight='weight',  # Use weight attribute for shortest path
                method='dijkstra'
            )
            
            # Check if path is too long
            if len(path_nodes) > max_length + 1:  # +1 because path includes source and target
                return None
                
            # Construct path with nodes and edges
            path = []
            
            for i in range(len(path_nodes) - 1):
                from_node = path_nodes[i]
                to_node = path_nodes[i + 1]
                
                # Get node and edge data
                from_data = self.graph.nodes[from_node]
                edge_data = self.graph[from_node][to_node]
                
                path.append({
                    "node": {
                        "id": from_node,
                        "type": from_data["type"],
                        "properties": from_data["properties"]
                    },
                    "edge": {
                        "type": edge_data["type"],
                        "properties": edge_data["properties"],
                        "weight": edge_data.get("weight", 1.0)
                    } if i < len(path_nodes) - 1 else None
                })
            
            # Add final node
            final_data = self.graph.nodes[path_nodes[-1]]
            path.append({
                "node": {
                    "id": path_nodes[-1],
                    "type": final_data["type"],
                    "properties": final_data["properties"]
                },
                "edge": None
            })
            
            return path
            
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error(f"Error finding path: {str(e)}")
            return None
            
    def find_connections(self, 
                        entity_id: str, 
                        max_distance: int = 2,
                        relation_types: List[str] = None) -> Dict[str, Any]:
        """
        Find connections to an entity within a certain distance
        
        Args:
            entity_id: Entity ID to find connections for
            max_distance: Maximum distance (number of hops)
            relation_types: Optional filter for relation types
            
        Returns:
            Dictionary of connected entities grouped by distance
        """
        if not self.graph.has_node(entity_id):
            return {"connections": {}}
            
        connections = defaultdict(list)
        visited = {entity_id}  # Start with source node as visited
        
        # BFS to find connections
        queue = [(entity_id, 0)]  # (node_id, distance)
        
        while queue:
            node_id, distance = queue.pop(0)
            
            if distance >= max_distance:
                continue
                
            # Explore outgoing edges
            for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                # Skip if relation type doesn't match filter
                if relation_types and edge_data["type"] not in relation_types:
                    continue
                    
                # Skip already visited nodes
                if target in visited:
                    continue
                    
                # Mark as visited
                visited.add(target)
                
                # Get target node data
                target_data = self.graph.nodes[target]
                
                # Add to connections at current distance + 1
                connections[distance + 1].append({
                    "id": target,
                    "type": target_data["type"],
                    "properties": target_data["properties"],
                    "relation": {
                        "type": edge_data["type"],
                        "direction": "outgoing",
                        "properties": edge_data["properties"],
                        "weight": edge_data.get("weight", 1.0)
                    }
                })
                
                # Add to queue for next iteration
                queue.append((target, distance + 1))
                
            # Explore incoming edges
            for source, _, edge_data in self.graph.in_edges(node_id, data=True):
                # Skip if relation type doesn't match filter
                if relation_types and edge_data["type"] not in relation_types:
                    continue
                    
                # Skip already visited nodes
                if source in visited:
                    continue
                    
                # Mark as visited
                visited.add(source)
                
                # Get source node data
                source_data = self.graph.nodes[source]
                
                # Add to connections at current distance + 1
                connections[distance + 1].append({
                    "id": source,
                    "type": source_data["type"],
                    "properties": source_data["properties"],
                    "relation": {
                        "type": edge_data["type"],
                        "direction": "incoming",
                        "properties": edge_data["properties"],
                        "weight": edge_data.get("weight", 1.0)
                    }
                })
                
                # Add to queue for next iteration
                queue.append((source, distance + 1))
                
        result = {"connections": dict(connections)}
        
        # Add summary info
        total_connections = sum(len(conn_list) for conn_list in connections.values())
        result["summary"] = {
            "total_connections": total_connections,
            "max_distance": max_distance,
            "distances": {dist: len(conn_list) for dist, conn_list in connections.items()}
        }
        
        return result
        
    def resolve_entity(self, name_or_id: str) -> Optional[str]:
        """
        Resolve an entity name or alias to its ID
        
        Args:
            name_or_id: Entity name, alias or ID
            
        Returns:
            Entity ID if found, None otherwise
        """
        # Check if it's a direct ID
        if self.graph.has_node(name_or_id):
            return name_or_id
            
        # Check aliases
        if name_or_id in self.entity_aliases:
            # Get the first alias (there might be multiple)
            alias_matches = list(self.entity_aliases[name_or_id])
            if alias_matches:
                return alias_matches[0]
                
        return None
        
    def merge_entities(self, 
                      source_id: str, 
                      target_id: str, 
                      keep_source: bool = False) -> bool:
        """
        Merge two entities into one
        
        Args:
            source_id: ID of source entity
            target_id: ID of target entity
            keep_source: Whether to keep the source entity after merging
            
        Returns:
            True if merge was successful
        """
        # Check if both entities exist
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.error(f"Cannot merge: one or both entities don't exist")
            return False
            
        source_data = self.graph.nodes[source_id]
        target_data = self.graph.nodes[target_id]
        
        # Update target properties with source properties
        for key, value in source_data["properties"].items():
            if key not in target_data["properties"]:
                target_data["properties"][key] = value
                
        # Transfer all outgoing edges from source to target
        for _, out_neighbor, edge_data in list(self.graph.out_edges(source_id, data=True)):
            if out_neighbor != target_id:  # Skip self-loops to target
                self.add_relation(
                    target_id,
                    out_neighbor,
                    edge_data["type"],
                    edge_data["properties"],
                    edge_data.get("weight", 1.0)
                )
                
        # Transfer all incoming edges from source to target
        for in_neighbor, _, edge_data in list(self.graph.in_edges(source_id, data=True)):
            if in_neighbor != target_id:  # Skip self-loops from target
                self.add_relation(
                    in_neighbor,
                    target_id,
                    edge_data["type"],
                    edge_data["properties"],
                    edge_data.get("weight", 1.0)
                )
                
        # Merge aliases
        for alias, node_set in list(self.entity_aliases.items()):
            if source_id in node_set:
                node_set.remove(source_id)
                node_set.add(target_id)
                
        # Remove source entity if not keeping it
        if not keep_source:
            self.graph.remove_node(source_id)
            logger.info(f"Removed source entity {source_id} after merging")
            
        self.save_data()
        logger.info(f"Merged entity {source_id} into {target_id}")
        return True
        
    def add_fact(self,
                subject_id: str,
                predicate: str,
                object_id: str,
                confidence: float = 1.0,
                source: str = None,
                timestamp: str = None) -> bool:
        """
        Add a fact (subject-predicate-object) to the knowledge graph
        
        Args:
            subject_id: Subject entity ID
            predicate: Predicate/relation type
            object_id: Object entity ID
            confidence: Confidence score (0.0-1.0)
            source: Source of the fact
            timestamp: When the fact was observed/recorded
            
        Returns:
            True if fact was added successfully
        """
        # Set timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        # Create properties for the relation
        properties = {
            "confidence": confidence,
            "timestamp": timestamp
        }
        
        if source:
            properties["source"] = source
            
        # Add relation with inverse weight (higher confidence = lower weight for path finding)
        weight = 2.0 - confidence if confidence > 0 else 1.0
        
        return self.add_relation(
            subject_id,
            object_id,
            predicate,
            properties,
            weight
        )
        
    def visualize(self, 
                 central_entity: str = None,
                 max_nodes: int = 20,
                 output_file: str = "knowledge_graph.png"):
        """
        Visualize the knowledge graph or a portion of it
        
        Args:
            central_entity: Optional central entity to focus on
            max_nodes: Maximum number of nodes to display
            output_file: Path to save the visualization
        """
        try:
            if central_entity and not self.graph.has_node(central_entity):
                logger.error(f"Central entity {central_entity} not found")
                return
                
            # Create subgraph for visualization
            if central_entity:
                # Extract subgraph centered on the entity
                neighbors = set()
                neighbors.add(central_entity)
                
                # Add direct neighbors
                neighbors.update(self.graph.successors(central_entity))
                neighbors.update(self.graph.predecessors(central_entity))
                
                # Limit to max_nodes
                if len(neighbors) > max_nodes:
                    neighbors = list(neighbors)[:max_nodes]
                    
                subgraph = self.graph.subgraph(neighbors)
            else:
                # Take first max_nodes from the graph
                nodes = list(self.graph.nodes())[:max_nodes]
                subgraph = self.graph.subgraph(nodes)
                
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Define node colors based on type
            node_types = {data["type"] for _, data in subgraph.nodes(data=True)}
            color_map = {}
            
            for i, ntype in enumerate(node_types):
                color_map[ntype] = plt.cm.tab10(i % 10)
                
            node_colors = [color_map[subgraph.nodes[n]["type"]] for n in subgraph.nodes]
            
            # Create layout
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                subgraph, 
                pos, 
                node_color=node_colors,
                node_size=500,
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                subgraph,
                pos,
                width=1.0,
                alpha=0.5,
                arrows=True
            )
            
            # Draw node labels
            nx.draw_networkx_labels(
                subgraph,
                pos,
                font_size=8,
                font_family="sans-serif"
            )
            
            # Draw edge labels
            edge_labels = {(u, v): data["type"] for u, v, data in subgraph.edges(data=True)}
            nx.draw_networkx_edge_labels(
                subgraph,
                pos,
                edge_labels=edge_labels,
                font_size=6
            )
            
            # Add legend for node types
            legend_handles = []
            for ntype, color in color_map.items():
                patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                  markersize=10, label=ntype)
                legend_handles.append(patch)
                
            plt.legend(handles=legend_handles, loc='upper right')
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved graph visualization to {output_file}")
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")

# Utility functions for external use
def add_entity_to_graph(entity_id: str, entity_type: str, properties: Dict[str, Any], 
                       aliases: List[str] = None) -> str:
    """
    Add an entity to the knowledge graph
    
    Args:
        entity_id: Unique ID for the entity
        entity_type: Type of entity
        properties: Entity properties
        aliases: Alternative names
        
    Returns:
        Entity ID
    """
    kg = KnowledgeGraph()
    return kg.add_entity(entity_id, entity_type, properties, aliases)
    
def add_fact_to_graph(subject: str, predicate: str, object: str, 
                     confidence: float = 1.0, source: str = None) -> bool:
    """
    Add a fact to the knowledge graph
    
    Args:
        subject: Subject entity ID
        predicate: Relation/predicate type
        object: Object entity ID
        confidence: Confidence score
        source: Source of the fact
        
    Returns:
        True if successful
    """
    kg = KnowledgeGraph()
    return kg.add_fact(subject, predicate, object, confidence, source)
    
def find_entity_in_graph(entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Find an entity in the knowledge graph
    
    Args:
        entity_id: Entity ID or alias
        
    Returns:
        Entity data if found
    """
    kg = KnowledgeGraph()
    
    # Try to resolve the ID first
    resolved_id = kg.resolve_entity(entity_id)
    if resolved_id:
        return kg.get_entity(resolved_id)
    return None
    
def find_connections_in_graph(entity_id: str, max_distance: int = 2) -> Dict[str, Any]:
    """
    Find connections to an entity
    
    Args:
        entity_id: Entity ID
        max_distance: Maximum distance
        
    Returns:
        Connected entities
    """
    kg = KnowledgeGraph()
    return kg.find_connections(entity_id, max_distance)
    
def visualize_knowledge_graph(central_entity: str = None, 
                             max_nodes: int = 20,
                             output_file: str = "knowledge_graph.png"):
    """
    Visualize the knowledge graph
    
    Args:
        central_entity: Optional central entity
        max_nodes: Maximum nodes to display
        output_file: Output file path
    """
    kg = KnowledgeGraph()
    kg.visualize(central_entity, max_nodes, output_file)

# Test the knowledge graph
if __name__ == "__main__":
    # Create a new knowledge graph
    kg = KnowledgeGraph(storage_path="test_knowledge_graph.json")
    
    # Add some entities
    kg.add_entity(
        "john",
        "person",
        {"name": "John Smith", "age": 35, "occupation": "Developer"},
        aliases=["Johnny", "John Smith"]
    )
    
    kg.add_entity(
        "acme",
        "company",
        {"name": "Acme Inc.", "industry": "Technology", "founded": 2005},
        aliases=["Acme Inc", "Acme Inc."]
    )
    
    kg.add_entity(
        "project_x",
        "project",
        {"name": "Project X", "status": "active", "budget": 50000},
        aliases=["X Project"]
    )
    
    kg.add_entity(
        "python",
        "technology",
        {"name": "Python", "type": "programming language", "version": "3.9"},
        aliases=["Python language"]
    )
    
    # Add relationships
    kg.add_relation(
        "john",
        "acme",
        "works_at",
        {"start_date": "2020-01-15", "position": "Senior Developer"},
        weight=0.9
    )
    
    kg.add_relation(
        "john",
        "project_x",
        "works_on",
        {"role": "Lead Developer", "hours_per_week": 30},
        weight=0.8
    )
    
    kg.add_relation(
        "acme",
        "project_x",
        "owns",
        {"investment": 500000},
        weight=1.0
    )
    
    kg.add_relation(
        "john",
        "python",
        "knows",
        {"proficiency": "expert", "years_experience": 8},
        weight=0.95
    )
    
    kg.add_relation(
        "project_x",
        "python",
        "uses",
        {"component": "backend"},
        weight=1.0
    )
    
    # Add a fact
    kg.add_fact(
        "john",
        "created",
        "project_x",
        confidence=0.9,
        source="company records"
    )
    
    # Retrieve entity information
    john_info = kg.get_entity("john")
    print(f"Entity: {john_info['properties']['name']}")
    print(f"Type: {john_info['type']}")
    print("Outgoing relations:")
    for rel in john_info["outgoing_relations"]:
        print(f"  --[{rel['relation']}]--> {rel['target']} ({rel['target_type']})")
    
    # Find path between entities
    path = kg.find_path("john", "python")
    if path:
        print("\nPath from John to Python:")
        for i, step in enumerate(path):
            node = step["node"]
            print(f"{i+1}. {node['id']} ({node['type']})")
            if step["edge"]:
                print(f"   --[{step['edge']['type']}]-->")
    
    # Find connections
    connections = kg.find_connections("acme", max_distance=2)
    print(f"\nFound {connections['summary']['total_connections']} connections to Acme Inc.")
    
    # Visualize the graph
    kg.visualize(central_entity="john", output_file="test_graph.png")
    
    # Clean up test file
    import os
    if os.path.exists("test_knowledge_graph.json"):
        os.remove("test_knowledge_graph.json")
    if os.path.exists("test_graph.png"):
        os.remove("test_graph.png")

# Add these functions after the KnowledgeGraph class

def get_knowledge_graph():
    """Get the global knowledge graph instance"""
    # Create the knowledge graph instance if it doesn't exist
    try:
        knowledge_graph_path = os.path.join("memory", "knowledge_graph.json")
        graph = KnowledgeGraph(knowledge_graph_path)
        return graph
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {str(e)}")
        # Return a new empty graph as fallback
        return KnowledgeGraph()

def query_graph(query_type, params):
    """
    Query the knowledge graph
    
    Args:
        query_type (str): Type of query to perform
        params (dict): Parameters for the query
        
    Returns:
        List of results
    """
    try:
        graph = get_knowledge_graph()
        
        if query_type == "entity":
            # Get entity by ID
            entity_id = params.get("id")
            if entity_id and entity_id in graph.graph.nodes:
                node_data = graph.graph.nodes[entity_id]
                return {
                    "id": entity_id,
                    "type": node_data.get("type", ""),
                    "properties": node_data.get("properties", {})
                }
        
        elif query_type == "facts":
            # Get facts about an entity
            subject_id = params.get("subject")
            limit = params.get("limit", 10)
            
            if subject_id and subject_id in graph.graph.nodes:
                facts = []
                for _, obj_id, edge_data in graph.graph.out_edges(subject_id, data=True):
                    # Get the object entity
                    if obj_id in graph.graph.nodes:
                        obj_data = graph.graph.nodes[obj_id]
                        facts.append({
                            "subject": subject_id,
                            "predicate": edge_data.get("type", ""),
                            "object": obj_id,
                            "object_type": obj_data.get("type", ""),
                            "object_name": obj_data.get("properties", {}).get("name", obj_id),
                            "confidence": edge_data.get("properties", {}).get("confidence", 1.0)
                        })
                        
                        if len(facts) >= limit:
                            break
                
                return facts
                
        elif query_type == "related":
            # Get entities related to an entity
            entity_id = params.get("id")
            relation_type = params.get("relation")
            limit = params.get("limit", 10)
            
            if entity_id and entity_id in graph.graph.nodes:
                related_entities = []
                
                # Filter by relation type if specified
                for _, obj_id, edge_data in graph.graph.out_edges(entity_id, data=True):
                    if relation_type and edge_data.get("type") != relation_type:
                        continue
                        
                    if obj_id in graph.graph.nodes:
                        obj_data = graph.graph.nodes[obj_id]
                        related_entities.append({
                            "id": obj_id,
                            "type": obj_data.get("type", ""),
                            "properties": obj_data.get("properties", {}),
                            "relation": edge_data.get("type", "")
                        })
                        
                        if len(related_entities) >= limit:
                            break
                
                return related_entities
        
        return None
    
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {str(e)}")
        return None

def get_user_facts(user_id, limit=5):
    """
    Get facts about a user from the knowledge graph
    
    Args:
        user_id (str): User entity ID
        limit (int): Maximum number of facts to return
        
    Returns:
        List of facts about the user
    """
    try:
        return query_graph("facts", {"subject": user_id, "limit": limit})
    except Exception as e:
        logger.error(f"Error getting user facts: {str(e)}")
        return []

def get_entity_by_name(name, entity_type=None):
    """
    Find entities by name
    
    Args:
        name (str): Entity name to search for
        entity_type (str, optional): Filter by entity type
        
    Returns:
        List of matching entities
    """
    try:
        graph = get_knowledge_graph()
        matches = []
        
        # Search by name in properties
        for node_id, node_data in graph.graph.nodes(data=True):
            if entity_type and node_data.get("type") != entity_type:
                continue
                
            properties = node_data.get("properties", {})
            node_name = properties.get("name", "")
            
            # Check for name match
            if name.lower() in node_name.lower():
                matches.append({
                    "id": node_id,
                    "type": node_data.get("type", ""),
                    "properties": properties
                })
                
        # Also search the aliases
        for alias, entity_ids in graph.entity_aliases.items():
            if name.lower() in alias.lower():
                for entity_id in entity_ids:
                    if entity_id in graph.graph.nodes:
                        node_data = graph.graph.nodes[entity_id]
                        
                        if entity_type and node_data.get("type") != entity_type:
                            continue
                            
                        # Check if we already added this entity
                        if not any(match["id"] == entity_id for match in matches):
                            matches.append({
                                "id": entity_id,
                                "type": node_data.get("type", ""),
                                "properties": node_data.get("properties", {})
                            })
        
        return matches
    
    except Exception as e:
        logger.error(f"Error searching for entity by name: {str(e)}")
        return []

# Helper functions for adding to knowledge graph
def add_entity_to_graph(entity_id, entity_type, properties=None, aliases=None):
    """
    Add an entity to the knowledge graph
    
    Args:
        entity_id (str): Unique identifier for the entity
        entity_type (str): Type of entity
        properties (dict, optional): Entity properties
        aliases (list, optional): Alternative names for the entity
        
    Returns:
        True if successful
    """
    try:
        graph = get_knowledge_graph()
        success = graph.add_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=properties or {},
            aliases=aliases or []
        )
        return success
    except Exception as e:
        logger.error(f"Error adding entity to knowledge graph: {str(e)}")
        return False

def add_fact_to_graph(subject, predicate, object, confidence=1.0, source=None):
    """
    Add a fact to the knowledge graph
    
    Args:
        subject (str): Subject entity ID 
        predicate (str): Relation type
        object (str): Object entity ID
        confidence (float): Confidence score (0.0-1.0)
        source (str, optional): Source of the fact
        
    Returns:
        True if successful
    """
    try:
        graph = get_knowledge_graph()
        success = graph.add_fact(
            subject_id=subject,
            predicate=predicate, 
            object_id=object,
            confidence=confidence,
            source=source
        )
        return success
    except Exception as e:
        logger.error(f"Error adding fact to knowledge graph: {str(e)}")
        return False 