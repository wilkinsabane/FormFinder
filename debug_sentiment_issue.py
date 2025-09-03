#!/usr/bin/env python3
"""Debug script to identify why sentiment analysis is returning 0.0 values."""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and initialize config first
from formfinder.config import load_config, get_config
from formfinder.sentiment import SentimentAnalyzer
from formfinder.logger import get_logger

logger = get_logger(__name__)

def debug_sentiment_configuration():
    """Debug sentiment analyzer configuration and functionality."""
    print("🔍 Debugging Sentiment Analysis Configuration")
    print("=" * 50)
    
    try:
        # Load configuration
        load_config()
        config = get_config()
        
        # Check sentiment configuration
        sentiment_config = config.sentiment_analysis
        print(f"Sentiment analysis enabled: {sentiment_config.enabled}")
        print(f"News API key: {sentiment_config.news_api_key}")
        print(f"Cache hours: {sentiment_config.cache_hours}")
        
        # Test sentiment analyzer initialization
        if sentiment_config.news_api_key:
            print(f"\n📰 Testing SentimentAnalyzer with API key: {sentiment_config.news_api_key[:10]}...")
            
            # Initialize sentiment analyzer
            analyzer = SentimentAnalyzer(
                api_key=sentiment_config.news_api_key,
                cache_hours=sentiment_config.cache_hours
            )
            
            # Test with sample teams
            test_teams = [
                ("Arsenal", "Chelsea"),
                ("Manchester United", "Liverpool"),
                ("Barcelona", "Real Madrid")
            ]
            
            for home_team, away_team in test_teams:
                print(f"\n🏆 Testing sentiment for {home_team} vs {away_team}")
                
                try:
                    # Get sentiment for match
                    result = analyzer.get_sentiment_for_match(
                        home_team=home_team,
                        away_team=away_team,
                        match_date=datetime.now(),
                        days_back=7
                    )
                    
                    print(f"  Home sentiment ({home_team}): {result.home_sentiment:.3f}")
                    print(f"  Away sentiment ({away_team}): {result.away_sentiment:.3f}")
                    print(f"  Home articles: {result.home_article_count}")
                    print(f"  Away articles: {result.away_article_count}")
                    
                    # Show sample articles if available
                    if result.home_articles:
                        print(f"  Sample home article: {result.home_articles[0].get('title', 'No title')[:100]}...")
                    if result.away_articles:
                        print(f"  Sample away article: {result.away_articles[0].get('title', 'No title')[:100]}...")
                        
                except Exception as e:
                    print(f"  ❌ Error getting sentiment: {e}")
                    
                # Only test one pair to avoid rate limiting
                break
                
        else:
            print("\n⚠️ No NewsAPI key configured")
            
            # Test with a dummy key to see the behavior
            print("\n🧪 Testing with dummy API key to see behavior...")
            analyzer = SentimentAnalyzer()
            
            try:
                result = analyzer.get_sentiment_for_match(
                    home_team="Arsenal",
                    away_team="Chelsea",
                    match_date=datetime.now(),
                    days_back=7
                )
                print(f"  Result with dummy key: home={result.home_sentiment}, away={result.away_sentiment}")
            except Exception as e:
                print(f"  Error with dummy key: {e}")
                
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        print(traceback.format_exc())

def test_hardcoded_api_key():
    """Test the hardcoded API key that's being rejected."""
    print("\n🔑 Testing Hardcoded API Key")
    print("=" * 30)
    
    hardcoded_key = "ff008e7b4e9b4041ab44c50a729d7885"
    print(f"Testing key: {hardcoded_key}")
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Test article fetching directly
        articles = analyzer._fetch_team_articles(
            team_name="Arsenal",
            start_date="2025-01-05",
            end_date="2025-01-12",
            max_articles=5
        )
        
        print(f"Articles fetched: {len(articles)}")
        if articles:
            print(f"Sample article: {articles[0].get('title', 'No title')}")
        else:
            print("No articles returned - likely API key issue")
            
    except Exception as e:
        print(f"Error testing hardcoded key: {e}")

if __name__ == "__main__":
    debug_sentiment_configuration()
    test_hardcoded_api_key()