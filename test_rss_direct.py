#!/usr/bin/env python3
"""
Direct test of RSS feed fetching to diagnose the 0 articles issue.
"""

import requests
import feedparser
from datetime import datetime

def test_rss_feed_direct():
    """Test RSS feed fetching directly without the framework."""
    print("Testing RSS feeds directly...\n")
    
    # Test feeds
    test_feeds = [
        "https://www.bbc.co.uk/sport/football/rss.xml",
        "http://feeds.bbci.co.uk/sport/football/rss.xml",
        "https://feeds.skysports.com/feeds/12040",
        "https://www.espn.com/espn/rss/soccer/news"
    ]
    
    for feed_url in test_feeds:
        print(f"--- Testing: {feed_url} ---")
        try:
            # Test with requests first
            print("1. Testing with requests...")
            response = requests.get(feed_url, timeout=10, headers={
                'User-Agent': 'FormFinder RSS Reader 1.0'
            })
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('content-type', 'Unknown')}")
            print(f"   Content length: {len(response.content)} bytes")
            
            if response.status_code == 200:
                # Test with feedparser
                print("2. Testing with feedparser...")
                feed = feedparser.parse(response.content)
                print(f"   Feed title: {feed.feed.get('title', 'No title')}")
                print(f"   Number of entries: {len(feed.entries)}")
                
                if feed.entries:
                    print("   Sample entries:")
                    for i, entry in enumerate(feed.entries[:3]):
                        title = entry.get('title', 'No title')[:60]
                        pub_date = entry.get('published', 'No date')
                        print(f"     {i+1}. {title}... ({pub_date})")
                else:
                    print("   ⚠️  No entries found in feed")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_rss_feed_direct()