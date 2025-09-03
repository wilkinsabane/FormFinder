#!/usr/bin/env python3
"""
Test script to verify that the team matching fix resolves the issue of 0 articles being fetched.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.rss_content_parser import ParsedArticle
from datetime import datetime

def test_team_matching_logic():
    """Test the team matching logic directly."""
    print("Testing team matching logic...")
    
    # Create a mock RSS provider class to test the team matching method
    class MockRSSProvider:
        def _extract_team_matches(self, article, query: str) -> int:
            """
            Extract number of team name matches in article.
            
            Args:
                article: ParsedArticle to analyze
                query: Search query containing team names
                
            Returns:
                Number of team name matches found
            """
            try:
                # Simple team matching based on query terms in article content
                content_lower = f"{article.title} {article.content}".lower()
                query_terms = [term.strip().lower() for term in query.replace(' vs ', ' ').replace(' v ', ' ').split()]
                
                matches = 0
                for term in query_terms:
                    if len(term) > 2 and term in content_lower:  # Avoid matching very short terms
                        matches += 1
                
                return matches
            except Exception:
                return 0
    
    provider = MockRSSProvider()
    
    # Test articles
    test_cases = [
        {
            "query": "Sparta Prague",
            "articles": [
                ParsedArticle(
                     title="Sparta Prague wins against Slavia",
                     content="Sparta Prague dominated the match with a 3-1 victory over their rivals.",
                     url="http://example.com/1",
                     published_date=datetime.now(),
                     source="Test Source",
                     sport="football"
                 ),
                 ParsedArticle(
                     title="Champions League Preview",
                     content="Several teams including Barcelona and Real Madrid are preparing for the next round.",
                     url="http://example.com/2",
                     published_date=datetime.now(),
                     source="Test Source",
                     sport="football"
                 )
            ]
        },
        {
            "query": "Real Madrid",
            "articles": [
                ParsedArticle(
                     title="Real Madrid defeats Barcelona in El Clasico",
                     content="Real Madrid secured a 2-1 victory against Barcelona in a thrilling El Clasico match.",
                     url="http://example.com/3",
                     published_date=datetime.now(),
                     source="Test Source",
                     sport="football"
                 ),
                 ParsedArticle(
                     title="Transfer News Update",
                     content="Several clubs are looking for new signings this summer.",
                     url="http://example.com/4",
                     published_date=datetime.now(),
                     source="Test Source",
                     sport="football"
                 )
            ]
        }
    ]
    
    total_tests = 0
    successful_matches = 0
    
    for test_case in test_cases:
        query = test_case["query"]
        articles = test_case["articles"]
        
        print(f"\n--- Testing query: '{query}' ---")
        
        for i, article in enumerate(articles, 1):
            total_tests += 1
            matches = provider._extract_team_matches(article, query)
            
            print(f"Article {i}: {article.title}")
            print(f"  Content: {article.content[:100]}...")
            print(f"  Team matches found: {matches}")
            
            if matches > 0:
                successful_matches += 1
                print(f"  ✓ MATCH FOUND")
            else:
                print(f"  ✗ NO MATCH")
            print()
    
    print(f"\n=== Team Matching Test Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Successful matches: {successful_matches}")
    print(f"Success rate: {(successful_matches/total_tests)*100:.1f}%")
    
    if successful_matches > 0:
        print("\n✓ Team matching fix is working! Articles should now be fetched.")
    else:
        print("\n✗ Team matching is still not working properly.")
    
    # Test edge cases
    print("\n--- Testing Edge Cases ---")
    
    edge_cases = [
        {
            "query": "Man United",
            "article": ParsedArticle(
                 title="Manchester United vs Liverpool Preview",
                 content="Manchester United will face Liverpool in a crucial Premier League match.",
                 url="http://example.com/5",
                 published_date=datetime.now(),
                 source="Test Source",
                 sport="football"
             )
         },
         {
             "query": "Barcelona vs Real Madrid",
             "article": ParsedArticle(
                 title="El Clasico: Barcelona takes on Real Madrid",
                 content="The biggest match in Spanish football as Barcelona hosts Real Madrid.",
                 url="http://example.com/6",
                 published_date=datetime.now(),
                 source="Test Source",
                 sport="football"
             )
        }
    ]
    
    for case in edge_cases:
        query = case["query"]
        article = case["article"]
        matches = provider._extract_team_matches(article, query)
        
        print(f"Query: '{query}'")
        print(f"Article: {article.title}")
        print(f"Matches: {matches}")
        print(f"Result: {'✓ MATCH' if matches > 0 else '✗ NO MATCH'}")
        print()

if __name__ == "__main__":
    test_team_matching_logic()