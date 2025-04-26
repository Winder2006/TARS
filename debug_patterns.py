#!/usr/bin/env python3
import re

def debug_pattern_matching(query, patterns, description):
    """Debug regex pattern matching for a given query and patterns."""
    print(f"\nTesting: '{query}' against {description} patterns")
    query_lower = query.lower()
    for i, pattern in enumerate(patterns, 1):
        match = re.search(pattern, query_lower)
        result = "✓ MATCH" if match else "✗ NO MATCH"
        matched_text = match.group(0) if match else ""
        print(f"  Pattern {i}: '{pattern}'  ->  {result}  {matched_text}")

# Test problematic queries
queries = [
    "Should I invest in Apple stock?",
    "Where is the Eiffel Tower?"
]

# Personal advice patterns
personal_advice_patterns = [
    r"^should i",
    r"^would i",
    r"^could i",
    r"^can i",
    r"^do i need",
    r"^(what|which|how) should i",
    r"^is it (good|bad|worth|advisable) for me to",
]

# Geographic patterns
geographic_patterns = [
    r"where is (?!my\b)(?!your\b)(.+)",
    r"location of (?!my\b)(?!your\b)(.+)",
    r"how (far|close) is (.+) (from|to) (.+)",
    r"directions (to|from) (.+)",
    r"which (country|state|city|province|continent) (.+)"
]

# Test each query against both pattern sets
for query in queries:
    debug_pattern_matching(query, personal_advice_patterns, "personal advice")
    debug_pattern_matching(query, geographic_patterns, "geographic")
    print("-" * 50) 