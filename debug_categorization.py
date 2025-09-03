#!/usr/bin/env python3
"""
Debug script to identify the misclassified article.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.rss_content_parser import RSSContentParser, ParsingConfig, ParsedArticle
from datetime import datetime

def debug_categorization():
    """Debug the categorization issue."""
    
    config = ParsingConfig()
    parser = RSSContentParser(config)
    
    # Test the specific articles that might be problematic
    test_articles = [
        {
            "title": "EuroLeague Basketball: Real Madrid advances",
            "content": "Real Madrid basketball team secured their spot in the EuroLeague final.",
            "expected": "basketball"
        },
        {
            "title": "Sports News: Multiple championship finals this weekend",
            "content": "This weekend features finals in various sports including football, basketball, and tennis.",
            "expected": "football"  # Should pick football due to higher scoring
        }
    ]
    
    for i, test_case in enumerate(test_articles, 1):
        print(f"\n=== Article {i} ===")
        print(f"Title: {test_case['title']}")
        print(f"Content: {test_case['content']}")
        print(f"Expected: {test_case['expected']}")
        
        # Create article
        article = ParsedArticle(
            title=test_case["title"],
            content=test_case["content"],
            url=f"https://example.com/article{i}",
            published_date=datetime.now()
        )
        
        # Debug the categorization process
        content_lower = f"{article.title} {article.content}".lower()
        print(f"\nContent (lowercase): {content_lower}")
        
        sport_scores = {}
        
        for sport, keywords in config.sport_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in content_lower:
                    if sport == 'basketball' and keyword in ['basketball', 'nba', 'euroleague basketball']:
                        score += 15
                    elif sport == 'basketball' and keyword in ['euroleague', 'fiba', 'wnba']:
                        score += 10
                    elif sport == 'tennis' and keyword in ['tennis', 'wimbledon']:
                        score += 15
                    elif sport == 'cricket' and keyword in ['cricket', 'ipl']:
                        score += 15
                    elif sport == 'rugby' and keyword in ['rugby']:
                        score += 15
                    elif sport == 'golf' and keyword in ['golf', 'pga']:
                        score += 15
                    elif sport == 'football':
                        if keyword in ['football', 'soccer', 'premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1', 'uefa champions league', 'fifa world cup']:
                            score += 5
                        else:
                            score += 1
                    else:
                        score += 1
                    
                    matched_keywords.append(keyword)
            
            if score > 0:
                sport_scores[sport] = score
                print(f"\n{sport}: score={score}, keywords={matched_keywords}")
        
        # Assign sport based on highest score
        if sport_scores:
            predicted_sport = max(sport_scores, key=sport_scores.get)
        else:
            predicted_sport = ""
        
        print(f"\nFinal scores: {sport_scores}")
        print(f"Predicted: {predicted_sport}")
        print(f"Expected: {test_case['expected']}")
        print(f"Correct: {predicted_sport == test_case['expected']}")

if __name__ == "__main__":
    debug_categorization()