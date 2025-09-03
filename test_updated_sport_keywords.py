#!/usr/bin/env python3
"""
Test script to verify updated sport keywords for article categorization.

This script tests the enhanced sport keyword categorization with the comprehensive
list of football/soccer leagues and competitions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from formfinder.rss_content_parser import RSSContentParser, ParsingConfig, ParsedArticle
from datetime import datetime

def test_updated_sport_keywords():
    """Test the updated sport keywords for better categorization."""
    print("Testing Updated Sport Keywords for Article Categorization")
    print("=" * 60)
    
    # Initialize parser with default config
    config = ParsingConfig()
    parser = RSSContentParser(config)
    
    # Test articles with various league and competition keywords
    test_articles = [
        {
            'title': 'UEFA Champions League Final: Real Madrid vs Manchester City',
            'content': 'The UEFA Champions League final will feature Real Madrid against Manchester City at Wembley Stadium.',
            'expected_sport': 'football'
        },
        {
            'title': 'Premier League: Arsenal defeats Chelsea 3-1',
            'content': 'Arsenal secured a crucial Premier League victory over Chelsea with goals from Bukayo Saka and Martin Ødegaard.',
            'expected_sport': 'football'
        },
        {
            'title': 'La Liga Update: Barcelona vs Real Madrid El Clásico Preview',
            'content': 'The upcoming El Clásico between Barcelona and Real Madrid in La Liga promises to be spectacular.',
            'expected_sport': 'football'
        },
        {
            'title': 'Bundesliga News: Bayern Munich signs Harry Kane',
            'content': 'Bayern Munich has completed the signing of Harry Kane from Tottenham in a record Bundesliga transfer.',
            'expected_sport': 'football'
        },
        {
            'title': 'Serie A: Juventus vs Inter Milan Derby d\'Italia',
            'content': 'The Derby d\'Italia between Juventus and Inter Milan in Serie A ended in a thrilling 2-2 draw.',
            'expected_sport': 'football'
        },
        {
            'title': 'MLS Cup Final: LAFC vs Philadelphia Union',
            'content': 'Major League Soccer\'s MLS Cup final will see LAFC take on Philadelphia Union at Banc of California Stadium.',
            'expected_sport': 'football'
        },
        {
            'title': 'Copa Libertadores: River Plate advances to semifinals',
            'content': 'River Plate secured their place in the Copa Libertadores semifinals with a 2-0 victory.',
            'expected_sport': 'football'
        },
        {
            'title': 'Saudi League: Cristiano Ronaldo scores hat-trick for Al-Nassr',
            'content': 'Cristiano Ronaldo scored a hat-trick as Al-Nassr defeated Al-Hilal 4-2 in the Saudi League.',
            'expected_sport': 'football'
        },
        {
            'title': 'J1 League: Tokyo FC wins against Yokohama F. Marinos',
            'content': 'Tokyo FC secured a 3-1 victory over Yokohama F. Marinos in the J1 League championship race.',
            'expected_sport': 'football'
        },
        {
            'title': 'Brasileirão: Palmeiras clinches title with victory over Santos',
            'content': 'Palmeiras won the Brasileirão championship after defeating Santos 2-1 at Allianz Parque.',
            'expected_sport': 'football'
        },
        {
            'title': 'NBA Finals: Lakers vs Celtics Game 7',
            'content': 'The NBA Finals Game 7 between Lakers and Celtics was a thrilling basketball matchup.',
            'expected_sport': 'basketball'
        },
        {
            'title': 'Tennis News: Wimbledon Championships Begin',
            'content': 'The Wimbledon tennis championships have begun with exciting first-round matches.',
            'expected_sport': 'tennis'
        }
    ]
    
    print(f"Testing {len(test_articles)} articles with updated sport keywords...\n")
    
    correct_categorizations = 0
    total_articles = len(test_articles)
    
    for i, test_data in enumerate(test_articles, 1):
        # Create ParsedArticle object
        article = ParsedArticle(
            title=test_data['title'],
            content=test_data['content'],
            url=f"https://example.com/article-{i}",
            published_date=datetime.now(),
            source="Test Source"
        )
        
        # Categorize the article
        parser._categorize_article(article)
        
        # Check if categorization is correct
        is_correct = article.sport == test_data['expected_sport']
        if is_correct:
            correct_categorizations += 1
            status = "✓ CORRECT"
        else:
            status = "✗ INCORRECT"
        
        print(f"Article {i}: {status}")
        print(f"  Title: {test_data['title'][:80]}...")
        print(f"  Expected: {test_data['expected_sport']}")
        print(f"  Actual: {article.sport}")
        print()
    
    # Print summary
    accuracy = (correct_categorizations / total_articles) * 100
    print("=" * 60)
    print(f"CATEGORIZATION RESULTS:")
    print(f"Correct: {correct_categorizations}/{total_articles}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 90:
        print("✓ EXCELLENT: Sport categorization is working very well!")
    elif accuracy >= 75:
        print("✓ GOOD: Sport categorization is working well.")
    elif accuracy >= 50:
        print("⚠ FAIR: Sport categorization needs improvement.")
    else:
        print("✗ POOR: Sport categorization needs significant improvement.")
    
    return accuracy >= 75

def test_keyword_coverage():
    """Test the coverage of new keywords."""
    print("\n" + "=" * 60)
    print("Testing Keyword Coverage")
    print("=" * 60)
    
    config = ParsingConfig()
    football_keywords = config.sport_keywords['football']
    
    print(f"Total football keywords: {len(football_keywords)}")
    
    # Check for key categories
    categories = {
        'International Competitions': ['fifa world cup', 'uefa champions league', 'copa libertadores'],
        'European Leagues': ['premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1'],
        'American Leagues': ['mls', 'liga mx', 'brasileirão'],
        'Asian/African Leagues': ['j1 league', 'saudi league', 'a-league men'],
        'Player Names': ['cristiano ronaldo', 'lionel messi', 'harry kane'],
        'Club Names': ['real madrid', 'barcelona', 'bayern munich']
    }
    
    for category, sample_keywords in categories.items():
        found_keywords = [kw for kw in sample_keywords if kw in football_keywords]
        coverage = len(found_keywords) / len(sample_keywords) * 100
        print(f"{category}: {len(found_keywords)}/{len(sample_keywords)} ({coverage:.0f}%)")
        
        if found_keywords:
            print(f"  Found: {', '.join(found_keywords)}")
        else:
            print(f"  Missing: {', '.join(sample_keywords)}")
        print()

if __name__ == "__main__":
    try:
        # Test updated sport keywords
        success = test_updated_sport_keywords()
        
        # Test keyword coverage
        test_keyword_coverage()
        
        if success:
            print("\n✓ All tests passed! Updated sport keywords are working correctly.")
        else:
            print("\n⚠ Some tests failed. Review the categorization logic.")
            
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()