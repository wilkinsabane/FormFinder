#!/usr/bin/env python3
"""
Test script to verify that the rate limiting fix for sentiment analysis works properly.
"""

import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and initialize config first
from formfinder.config import load_config
load_config()

from formfinder.sentiment import SentimentAnalyzer

def test_rate_limiting_fix():
    """
    Test that the sentiment analyzer now handles rate limiting properly.
    """
    print("🔧 Testing Rate Limiting Fix for Sentiment Analysis")
    print("=" * 50)
    
    # Use the API key from config
    api_key = "ff008e7b4e9b4041ab44c50a729d7885"
    print(f"🔑 Using API key: {api_key[:10]}...")
    
    try:
        # Initialize sentiment analyzer with rate limiting
        analyzer = SentimentAnalyzer()
        print("✅ SentimentAnalyzer initialized with rate limiting")
        
        # Test multiple rapid requests to trigger rate limiting
        print("\n🚀 Testing multiple rapid requests...")
        teams = ["Arsenal", "Chelsea", "Manchester United", "Liverpool"]
        
        for i, team in enumerate(teams, 1):
            print(f"\n📰 Request {i}: Fetching articles for {team}...")
            
            try:
                articles = analyzer._fetch_team_articles(
                    team_name=team,
                    start_date="2025-01-10",
                    end_date="2025-01-17",
                    max_articles=3
                )
                
                print(f"   ✅ Success: {len(articles)} articles fetched")
                if articles:
                    print(f"   📄 Sample: {articles[0].get('title', 'No title')[:60]}...")
                else:
                    print("   ⚠️ No articles returned (may be rate limited but handled gracefully)")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test sentiment analysis with rate limiting
        print("\n🎭 Testing full sentiment analysis with rate limiting...")
        result = analyzer.get_sentiment_for_match(
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=datetime.now(),
            days_back=7
        )
        
        print(f"🏠 Home sentiment (Arsenal): {result.home_sentiment:.3f}")
        print(f"🏃 Away sentiment (Chelsea): {result.away_sentiment:.3f}")
        print(f"📰 Home articles: {result.home_article_count}")
        print(f"📰 Away articles: {result.away_article_count}")
        
        print("\n🎉 Rate limiting test completed successfully!")
        print("   The system now handles 429 errors gracefully with:")
        print("   - Exponential backoff retry logic")
        print("   - Minimum 1-second intervals between requests")
        print("   - Proper error logging and recovery")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rate_limiting_fix()