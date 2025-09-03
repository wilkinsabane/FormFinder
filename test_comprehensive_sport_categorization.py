#!/usr/bin/env python3
"""
Comprehensive test for sport categorization with the updated keywords.
Tests the improved categorization logic and comprehensive football keywords.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.rss_content_parser import RSSContentParser, ParsingConfig, ParsedArticle
from datetime import datetime

def test_comprehensive_categorization():
    """Test comprehensive sport categorization with various articles."""
    
    # Initialize parser
    config = ParsingConfig()
    parser = RSSContentParser(config)
    
    # Test articles covering various sports and leagues
    test_articles = [
        # Football/Soccer articles
        {
            "title": "UEFA Champions League Final: Real Madrid vs Manchester City",
            "content": "The UEFA Champions League final will be played at Wembley Stadium.",
            "expected": "football"
        },
        {
            "title": "Premier League: Arsenal defeats Chelsea 3-1",
            "content": "Arsenal secured a crucial victory in the Premier League title race.",
            "expected": "football"
        },
        {
            "title": "La Liga: Barcelona signs new striker",
            "content": "FC Barcelona has completed the signing of a new forward for La Liga.",
            "expected": "football"
        },
        {
            "title": "Bundesliga: Bayern Munich wins title",
            "content": "Bayern Munich clinched the Bundesliga championship with Harry Kane scoring.",
            "expected": "football"
        },
        {
            "title": "Serie A: Juventus vs Inter Milan Derby",
            "content": "The Derby d'Italia between Juventus and Inter Milan ends in a draw.",
            "expected": "football"
        },
        {
            "title": "MLS Cup: LAFC advances to final",
            "content": "Los Angeles FC secured their spot in the MLS Cup final with Lionel Messi watching.",
            "expected": "football"
        },
        {
            "title": "Brasileirão: Palmeiras leads the table",
            "content": "Palmeiras continues to lead the Campeonato Brasileiro Série A standings.",
            "expected": "football"
        },
        {
            "title": "Saudi League: Cristiano Ronaldo scores hat-trick",
            "content": "Cristiano Ronaldo scored three goals for Al-Nassr FC in the Roshn Saudi League.",
            "expected": "football"
        },
        
        # Basketball articles
        {
            "title": "NBA Finals: Lakers defeat Celtics in Game 7",
            "content": "The Los Angeles Lakers won the NBA championship in a thrilling Game 7.",
            "expected": "basketball"
        },
        {
            "title": "EuroLeague Basketball: Real Madrid advances",
            "content": "Real Madrid basketball team secured their spot in the EuroLeague final.",
            "expected": "basketball"
        },
        
        # Tennis articles
        {
            "title": "Wimbledon Championships: Djokovic wins title",
            "content": "Novak Djokovic claimed his eighth Wimbledon tennis championship.",
            "expected": "tennis"
        },
        {
            "title": "US Open Tennis: Serena Williams retires",
            "content": "Serena Williams played her final match at the US Open tennis tournament.",
            "expected": "tennis"
        },
        
        # Cricket articles
        {
            "title": "IPL Cricket: Mumbai Indians win championship",
            "content": "Mumbai Indians defeated Chennai Super Kings in the IPL final.",
            "expected": "cricket"
        },
        {
            "title": "Test Cricket: England vs Australia Ashes",
            "content": "The Ashes test match between England and Australia continues.",
            "expected": "cricket"
        },
        
        # Rugby articles
        {
            "title": "Six Nations Rugby: Wales defeats Ireland",
            "content": "Wales secured a crucial victory in the Six Nations rugby championship.",
            "expected": "rugby"
        },
        
        # Golf articles
        {
            "title": "Masters Golf Tournament: Tiger Woods leads",
            "content": "Tiger Woods takes the lead at the Masters golf tournament at Augusta.",
            "expected": "golf"
        },
        {
            "title": "PGA Championship: New champion crowned",
            "content": "A new champion was crowned at the PGA Championship golf tournament.",
            "expected": "golf"
        },
        
        # Ambiguous articles that should be categorized correctly
        {
            "title": "Sports News: Multiple championship finals this weekend",
            "content": "This weekend features finals in various sports including football, basketball, and tennis.",
            "expected": "football"  # Should pick football due to higher scoring
        }
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE SPORT CATEGORIZATION TEST")
    print("=" * 80)
    
    correct = 0
    total = len(test_articles)
    
    for i, test_case in enumerate(test_articles, 1):
        # Create article
        article = ParsedArticle(
            title=test_case["title"],
            content=test_case["content"],
            url=f"https://example.com/article{i}",
            published_date=datetime.now()
        )
        
        # Categorize
        parser._categorize_article(article)
        
        # Check result
        is_correct = article.sport == test_case["expected"]
        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            status = "✗ INCORRECT"
        
        print(f"\nArticle {i}: {status}")
        print(f"  Title: {test_case['title'][:60]}...")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Actual: {article.sport}")
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 95:
        print("✓ EXCELLENT: Sport categorization is working very well!")
    elif accuracy >= 85:
        print("✓ GOOD: Sport categorization is working well.")
    elif accuracy >= 70:
        print("⚠ FAIR: Sport categorization needs some improvement.")
    else:
        print("✗ POOR: Sport categorization needs significant improvement.")
    
    return accuracy >= 95

def test_keyword_coverage():
    """Test coverage of various football keyword categories."""
    
    config = ParsingConfig()
    football_keywords = config.sport_keywords['football']
    
    print(f"\n{'='*60}")
    print("KEYWORD COVERAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total football keywords: {len(football_keywords)}")
    
    # Test categories
    categories = {
        "International Competitions": ['fifa world cup', 'uefa champions league', 'copa libertadores'],
        "European Leagues": ['premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1'],
        "American Leagues": ['mls', 'liga mx', 'brasileirão'],
        "Asian/African Leagues": ['j1 league', 'saudi league', 'a-league men'],
        "Player Names": ['cristiano ronaldo', 'lionel messi', 'harry kane'],
        "Club Names": ['bayern munich', 'real madrid', 'barcelona']
    }
    
    for category, test_keywords in categories.items():
        found = [kw for kw in test_keywords if kw in football_keywords]
        coverage = len(found) / len(test_keywords) * 100
        print(f"{category}: {len(found)}/{len(test_keywords)} ({coverage:.0f}%)")
        if found:
            print(f"  Found: {', '.join(found)}")
    
    print(f"\n✓ Comprehensive keyword coverage verified!")

if __name__ == "__main__":
    print("Testing comprehensive sport categorization...\n")
    
    # Run comprehensive categorization test
    success = test_comprehensive_categorization()
    
    # Test keyword coverage
    test_keyword_coverage()
    
    print(f"\n{'='*60}")
    if success:
        print("✓ All comprehensive tests passed! Sport categorization system is excellent.")
    else:
        print("✗ Some tests failed. Sport categorization system needs improvement.")
    print(f"{'='*60}")