#!/usr/bin/env python3
"""
Test script to verify the team matching fix in RSS news provider.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load configuration first
from formfinder.config import load_config
print("Loading configuration...")
try:
    config = load_config("config.yaml")
    print("✓ Configuration loaded successfully")
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    print("Proceeding with default configuration...")

from formfinder.rss_news_provider import RSSNewsProvider, RSSProviderConfig
from formfinder.rss_feed_manager import RSSConfig, RSSFeed
from formfinder.rss_content_parser import RSSContentParser, ParsingConfig
from formfinder.rss_cache_pg import create_default_rss_cache
from formfinder.rss_monitoring_pg import create_default_rss_monitor

def test_team_matching():
    """Test team matching with actual RSS feeds."""
    print("Testing team matching fix...")
    
    # Create configuration
    parsing_config = ParsingConfig(
        user_agent="FormFinder/1.0",
        min_content_length=50,
        max_content_length=10000,
        extract_full_content=False,
        clean_html=True,
        normalize_text=True,
        filter_duplicates=True,
        max_articles_per_feed=50,
        content_timeout=10
    )
    
    rss_config = RSSConfig(
        feeds=[
            RSSFeed(
                url='https://www.bbc.co.uk/sport/football/rss.xml',
                name='BBC Football',
                sport='football',
                priority=1
            )
        ],
        cache_dir="data/rss_cache",
        max_cache_age_hours=48,
        health_check_interval=300,
        max_consecutive_failures=5,
        request_timeout=30,
        user_agent="FormFinder RSS Reader 1.0",
        enable_health_monitoring=True
    )
    
    provider_config = RSSProviderConfig(
        enabled=True,
        name="Test RSS Provider",
        priority=5,
        max_articles_per_query=10,
        rss_config=rss_config,
        parsing_config=parsing_config
    )
    
    # Create provider
    cache = create_default_rss_cache()
    monitor = create_default_rss_monitor()
    provider = RSSNewsProvider(provider_config, cache, monitor)
    
    # Test queries
    test_queries = [
        "football",  # Generic query first
        "Sparta Prague",
        "Real Madrid", 
        "Manchester United",
        "Barcelona",
        "Liverpool"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        try:
            articles = provider.fetch_articles(query, max_results=5)
            print(f"Found {len(articles)} articles")
            
            if articles:
                for i, article in enumerate(articles[:3]):
                    title = article.get('title', 'No title')[:60]
                    print(f"  {i+1}. {title}...")
            else:
                print("  No articles found")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nTeam matching test completed.")

if __name__ == "__main__":
    test_team_matching()