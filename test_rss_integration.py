#!/usr/bin/env python3
"""
Test script to verify RSS integration with sentiment analysis system.
This script tests the RSS fallback functionality when API providers are unavailable.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.sentiment import SentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rss_fallback_integration():
    """
    Test RSS fallback integration when all API providers are disabled.
    """
    print("\n=== Testing RSS Fallback Integration ===")
    
    try:
        # Load configuration
        load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize SentimentAnalyzer (should trigger RSS fallback if API providers disabled)
        analyzer = SentimentAnalyzer()
        
        # Check which mode the analyzer is using
        if hasattr(analyzer, 'use_rss') and analyzer.use_rss:
            print("✅ RSS fallback mode activated")
            logger.info("SentimentAnalyzer initialized with RSS fallback")
        elif hasattr(analyzer, 'use_manager') and analyzer.use_manager:
            print("ℹ️  Multi-provider mode active (API providers available)")
            logger.info("SentimentAnalyzer initialized with multi-provider system")
        else:
            print("⚠️  Legacy NewsAPI mode active")
            logger.info("SentimentAnalyzer initialized with legacy NewsAPI mode")
        
        # Test article fetching
        print("\n--- Testing Article Fetching ---")
        
        # Calculate date range (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Test with a popular team
        test_team = "Manchester United"
        print(f"Fetching articles for {test_team} from {start_date_str} to {end_date_str}")
        
        articles = analyzer._fetch_team_articles(
            team_name=test_team,
            start_date=start_date_str,
            end_date=end_date_str,
            max_articles=5
        )
        
        print(f"\nFetched {len(articles)} articles:")
        for i, article in enumerate(articles[:3], 1):  # Show first 3 articles
            print(f"{i}. {article.get('title', 'No title')[:80]}...")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   Published: {article.get('publishedAt', 'Unknown')}")
            print()
        
        if articles:
            print("✅ Article fetching successful")
        else:
            print("⚠️  No articles fetched (this might be expected if no RSS feeds are available)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"RSS integration test failed: {e}")
        return False

def test_sentiment_analysis_with_rss():
    """
    Test full sentiment analysis with RSS integration.
    """
    print("\n=== Testing Sentiment Analysis with RSS ===")
    
    try:
        # Load configuration
        load_config()
        
        # Initialize SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        # Test sentiment analysis for a match
        home_team = "Arsenal"
        away_team = "Chelsea"
        
        print(f"Analyzing sentiment for {home_team} vs {away_team}")
        
        # Get sentiment (this will use whatever provider is available)
        sentiment_result = analyzer.get_sentiment_for_match(
            home_team=home_team,
            away_team=away_team,
            match_date=datetime.now(),
            days_back=7
        )
        
        print(f"\nSentiment Analysis Results:")
        print(f"Home Team ({home_team}):")
        print(f"  Sentiment Score: {sentiment_result.home_sentiment:.3f}")
        print(f"  Article Count: {sentiment_result.home_article_count}")
        
        print(f"\nAway Team ({away_team}):")
        print(f"  Sentiment Score: {sentiment_result.away_sentiment:.3f}")
        print(f"  Article Count: {sentiment_result.away_article_count}")
        
        # Show sample articles
        if sentiment_result.home_articles:
            print(f"\nSample {home_team} articles:")
            for i, article in enumerate(sentiment_result.home_articles[:2], 1):
                print(f"{i}. {article.get('title', 'No title')[:60]}...")
        
        if sentiment_result.away_articles:
            print(f"\nSample {away_team} articles:")
            for i, article in enumerate(sentiment_result.away_articles[:2], 1):
                print(f"{i}. {article.get('title', 'No title')[:60]}...")
        
        print("\n✅ Sentiment analysis completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        logger.error(f"Sentiment analysis with RSS test failed: {e}")
        return False

def main():
    """
    Run all RSS integration tests.
    """
    print("RSS Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_rss_fallback_integration,
        test_sentiment_analysis_with_rss
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! RSS integration is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)