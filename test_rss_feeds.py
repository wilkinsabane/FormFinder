#!/usr/bin/env python3
"""
Test script to verify RSS feeds are fetching articles correctly
after updating sport_keywords configuration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.rss_news_provider import create_default_rss_provider

def test_rss_feeds():
    """Test RSS feeds to see which ones are fetching articles."""
    print("Testing RSS feeds for article fetching...\n")
    
    # Create RSS provider
    provider = create_default_rss_provider()
    
    # Get feeds from the provider's feed manager
    feeds = list(provider.feed_manager.feeds.values())
    
    print(f"Total feeds configured: {len(feeds)}\n")
    
    # Test each feed individually
    for i, feed in enumerate(feeds[:10], 1):  # Test first 10 feeds
        print(f"{i}. Testing feed: {feed.name} (sport: {feed.sport})")
        print(f"   URL: {feed.url}")
        
        try:
            # Fetch articles from this specific feed
            articles = provider._fetch_from_feed(feed, "football")
            print(f"   Articles fetched: {len(articles)}")
            
            if articles:
                # Show first article details
                first_article = articles[0]
                print(f"   Sample article: '{first_article.title[:60]}...'")
                print(f"   Sport categorized as: '{first_article.sport}'")
                print(f"   Published: {first_article.published_date}")
            else:
                print(f"   ❌ No articles found")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()
    
    # Test overall provider with football query
    print("\n" + "="*60)
    print("Testing overall provider with 'football' query...")
    
    try:
        all_articles = provider.fetch_articles("football", max_results=20, sport="football")
        print(f"Total articles returned: {len(all_articles)}")
        
        if all_articles:
            print("\nSample articles:")
            for i, article in enumerate(all_articles[:5], 1):
                print(f"  {i}. {article['title'][:60]}...")
                print(f"     Source: {article['source']}")
                print(f"     Sport: {article.get('sport', 'N/A')}")
                print()
        else:
            print("❌ No articles returned from provider")
            
    except Exception as e:
        print(f"❌ Error testing provider: {str(e)}")

if __name__ == "__main__":
    test_rss_feeds()