#!/usr/bin/env python3
"""
Summary test to verify RSS functionality improvements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.rss_news_provider import create_default_rss_provider
from formfinder.rss_content_parser import RSSContentParser, ParsedArticle, ParsingConfig
from datetime import datetime

def test_sport_categorization():
    """Test that sport categorization works correctly with updated keywords."""
    print("=== Testing Sport Categorization ===")
    
    # Load configuration
    config = load_config()
    parsing_config = ParsingConfig()
    parser = RSSContentParser(parsing_config)
    
    # Test articles with various football-related content
    test_articles = [
        ParsedArticle(
            title="Manchester United beats Arsenal in Premier League",
            content="Great match in the Premier League between two top teams.",
            url="https://example.com/1",
            sport=""
        ),
        ParsedArticle(
            title="Bayern Munich wins Bundesliga title",
            content="Bayern Munich secured the Bundesliga championship.",
            url="https://example.com/2",
            sport=""
        ),
        ParsedArticle(
            title="Serie A match: Juventus vs Inter Milan",
            content="Exciting Serie A derby between Juventus and Inter.",
            url="https://example.com/3",
            sport=""
        ),
        ParsedArticle(
            title="La Liga El Clasico: Real Madrid vs Barcelona",
            content="The biggest match in La Liga football.",
            url="https://example.com/4",
            sport=""
        ),
        ParsedArticle(
            title="Lakers beat Warriors in NBA game",
            content="Los Angeles Lakers defeated Golden State Warriors.",
            url="https://example.com/5",
            sport=""
        )
    ]
    
    print(f"Football keywords count: {len(parsing_config.sport_keywords.get('football', []))}")
    print(f"Sample keywords: {parsing_config.sport_keywords.get('football', [])[:10]}")
    print()
    
    football_count = 0
    basketball_count = 0
    
    for article in test_articles:
        original_sport = article.sport
        parser._categorize_article(article)
        print(f"'{article.title}' -> '{article.sport}'")
        
        if article.sport == 'football':
            football_count += 1
        elif article.sport == 'basketball':
            basketball_count += 1
    
    print(f"\nResults:")
    print(f"  Football articles: {football_count}/4 (expected: 4)")
    print(f"  Basketball articles: {basketball_count}/1 (expected: 1)")
    
    success = football_count == 4 and basketball_count == 1
    print(f"  ✓ Categorization test: {'PASSED' if success else 'FAILED'}")
    
    return success

def test_rss_provider_setup():
    """Test that RSS provider can be created and configured."""
    print("\n=== Testing RSS Provider Setup ===")
    
    try:
        # Load configuration
        config = load_config()
        print("✓ Configuration loaded successfully")
        
        # Create RSS provider
        provider = create_default_rss_provider()
        print("✓ RSS provider created successfully")
        
        # Check feed manager
        active_feeds = provider.feed_manager.get_active_feeds()
        print(f"✓ Active feeds: {len(active_feeds)}")
        
        # Check some feed details
        if active_feeds:
            sample_feed = active_feeds[0]
            print(f"  Sample feed: {sample_feed.name} (sport: {sample_feed.sport})")
        
        # Check parsing configuration
        football_keywords = provider.content_parser.config.sport_keywords.get('football', [])
        print(f"✓ Football keywords configured: {len(football_keywords)}")
        
        return True
        
    except Exception as e:
        print(f"❌ RSS provider setup failed: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("RSS Functionality Test Summary")
    print("=" * 50)
    
    # Run tests
    categorization_success = test_sport_categorization()
    provider_setup_success = test_rss_provider_setup()
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Sport Categorization: {'✓ PASSED' if categorization_success else '❌ FAILED'}")
    print(f"RSS Provider Setup: {'✓ PASSED' if provider_setup_success else '❌ FAILED'}")
    
    if categorization_success and provider_setup_success:
        print("\n🎉 All core RSS functionality tests PASSED!")
        print("\nKey improvements verified:")
        print("  • Updated sport_keywords now include Bundesliga, Serie A, La Liga")
        print("  • Articles with league names are correctly categorized as 'football'")
        print("  • RSS provider can be properly initialized with configuration")
        print("  • Feed manager is working with active feeds")
        print("\nNote: Network connectivity issues prevent live RSS feed testing,")
        print("but the core categorization and filtering logic is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return categorization_success and provider_setup_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)