#!/usr/bin/env python3
"""
Test RSS feeds with proper configuration loading.
"""

import os
import sys
from pathlib import Path

# Add the formfinder package to the path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Test RSS feeds with proper configuration."""
    try:
        # Load the main configuration first
        from formfinder.config import load_config, get_config
        
        print("Loading configuration...")
        config = load_config("config.yaml")
        print(f"✓ Configuration loaded successfully")
        
        # Now test RSS functionality
        from formfinder.rss_news_provider import create_default_rss_provider
        from formfinder.rss_content_parser import ParsingConfig
        
        print("\n=== Testing RSS Provider ===\n")
        
        # Create RSS provider
        provider = create_default_rss_provider()
        print(f"✓ RSS provider created with {len(provider.feed_manager.feeds)} feeds")
        
        # List first 5 feeds
        print("\nFirst 5 RSS feeds:")
        active_feeds = provider.feed_manager.get_active_feeds()
        for i, feed in enumerate(active_feeds[:5]):
            print(f"  {i+1}. {feed.name}: sport={feed.sport}, url={feed.url[:60]}...")
        
        # Test fetching articles
        print("\n=== Fetching Articles ===\n")
        
        # Fetch all articles
        all_articles = provider.fetch_articles(query="football", max_results=20)
        print(f"Total articles fetched: {len(all_articles)}")
        
        if all_articles:
            print("\nSample articles:")
            for i, article in enumerate(all_articles[:3]):
                print(f"  {i+1}. {article.title[:60]}... (sport: {article.sport})")
        
        # Test filtering by sport
        print("\n=== Testing Sport Filtering ===\n")
        
        football_articles = provider.fetch_articles(query="football", sport="football", max_results=10)
        print(f"Football articles: {len(football_articles)}")
        
        basketball_articles = provider.fetch_articles(query="basketball", sport="basketball", max_results=10)
        print(f"Basketball articles: {len(basketball_articles)}")
        
        general_articles = provider.fetch_articles(query="sports", sport="general", max_results=10)
        print(f"General articles: {len(general_articles)}")
        
        # Test sport keywords
        print("\n=== Sport Keywords Configuration ===\n")
        
        parsing_config = ParsingConfig()
        print("Football keywords:")
        football_keywords = parsing_config.sport_keywords.get('football', [])
        print(f"  Count: {len(football_keywords)}")
        print(f"  Sample: {football_keywords[:10]}")
        
        # Test categorization with sample articles
        print("\n=== Testing Article Categorization ===\n")
        
        from formfinder.rss_content_parser import RSSContentParser, ParsedArticle
        
        parser = RSSContentParser(parsing_config)
        
        # Test articles
        test_articles = [
            ParsedArticle(
                title="Manchester United beats Arsenal 3-1 in Premier League",
                content="Manchester United secured a convincing 3-1 victory over Arsenal at Old Trafford in a thrilling Premier League encounter.",
                url="https://example.com/1",
                published_date="2024-01-15T10:00:00Z",
                author="Sports Reporter",
                sport=""
            ),
            ParsedArticle(
                title="Bayern Munich wins Bundesliga title",
                content="Bayern Munich clinched the Bundesliga title with a 2-0 victory over Borussia Dortmund.",
                url="https://example.com/2",
                published_date="2024-01-15T11:00:00Z",
                author="Football Correspondent",
                sport=""
            ),
            ParsedArticle(
                title="Serie A match report: Juventus vs Inter Milan",
                content="Juventus and Inter Milan played out a thrilling 2-2 draw in the Derby d'Italia.",
                url="https://example.com/3",
                published_date="2024-01-15T12:00:00Z",
                author="Italian Football Expert",
                sport=""
            ),
            ParsedArticle(
                title="Lakers beat Warriors in NBA showdown",
                content="The Los Angeles Lakers defeated the Golden State Warriors 115-108 in a high-scoring NBA game.",
                url="https://example.com/4",
                published_date="2024-01-15T13:00:00Z",
                author="Basketball Reporter",
                sport=""
            )
        ]
        
        print("Categorizing test articles:")
        for article in test_articles:
            print(f"  Before: '{article.title[:50]}...' -> sport: '{article.sport}'")
            parser._categorize_article(article)
            print(f"  After:  '{article.title[:50]}...' -> sport: '{article.sport}'")
            print()
        
        print("\n=== Test Complete ===\n")
        
        if len(all_articles) == 0:
            print("⚠️  WARNING: No articles were fetched. This could indicate:")
            print("   1. RSS feeds are not accessible")
            print("   2. Network connectivity issues")
            print("   3. RSS feed URLs are invalid")
            print("   4. Content filtering is too restrictive")
        else:
            print(f"✓ Successfully fetched {len(all_articles)} articles")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)