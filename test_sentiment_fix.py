#!/usr/bin/env python3
"""Test script to verify sentiment analysis is working after the fix."""

import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and initialize config first
from formfinder.config import load_config, get_config
load_config()

# Now import sentiment analyzer
from formfinder.sentiment import SentimentAnalyzer

def test_sentiment_fix():
    """Test that sentiment analysis now works with the valid API key."""
    print("🔧 Testing Sentiment Analysis Fix")
    print("=" * 40)
    
    # Test with the previously rejected but valid API key
    api_key = "ff008e7b4e9b4041ab44c50a729d7885"
    print(f"🔑 Testing with API key: {api_key[:10]}...")
    
    try:
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        print("✅ SentimentAnalyzer initialized successfully")
        
        # Test article fetching
        print("\n📰 Testing article fetching...")
        articles = analyzer._fetch_team_articles(
            team_name="Arsenal",
            start_date="2025-08-17",
            end_date="2025-08-24",
            max_articles=3
        )
        
        print(f"📊 Articles fetched: {len(articles)}")
        
        if articles:
            print("✅ Article fetching is working!")
            print(f"📄 Sample article: {articles[0].get('title', 'No title')[:80]}...")
            
            # Test sentiment analysis
            print("\n🎭 Testing sentiment analysis...")
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
            
            if result.home_sentiment != 0.0 or result.away_sentiment != 0.0:
                print("\n🎉 SUCCESS! Sentiment analysis is now working!")
            else:
                print("\n⚠️ Sentiment values are still 0.0 - may need further investigation")
                
        else:
            print("❌ No articles fetched - API key validation may still be failing")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sentiment_fix()