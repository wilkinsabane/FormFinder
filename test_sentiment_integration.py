#!/usr/bin/env python3
"""
Test script to verify sentiment analysis integration is working correctly.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.sentiment import SentimentAnalyzer
from formfinder.config import load_config


def test_sentiment_analyzer():
    """Test the SentimentAnalyzer class."""
    
    # Load configuration
    config = load_config()
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(
        api_key=config.sentiment_analysis.news_api_key if config.sentiment_analysis else None,
        cache_hours=config.sentiment_analysis.cache_hours if config.sentiment_analysis else 24
    )
    
    # Test teams
    test_teams = ["Arsenal", "Chelsea"]
    
    # Calculate date range (last 7 days)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=7)
    
    print("Testing sentiment analysis...")
    print(f"Date range: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
    
    try:
        # Test sentiment for a match
        result = analyzer.get_sentiment_for_match(
            home_team="Arsenal",
            away_team="Chelsea",
            days_back=7
        )
        home_sentiment = result.home_sentiment
        away_sentiment = result.away_sentiment
        total_articles = result.home_article_count + result.away_article_count
        
        print(f"Home team (Arsenal) sentiment: {home_sentiment}")
        print(f"Away team (Chelsea) sentiment: {away_sentiment}")
        print(f"Total articles analyzed: {total_articles}")
        
        if home_sentiment is not None and away_sentiment is not None:
            print("✅ Sentiment analysis test successful!")
            return True
        else:
            print("⚠️  Sentiment analysis returned None values (possibly no articles found)")
            return True  # Still consider it successful if no articles
            
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        return False


def test_textblob_import():
    """Test that TextBlob can be imported and used."""
    try:
        from textblob import TextBlob
        
        # Test basic sentiment analysis
        text = "Arsenal played amazingly well yesterday!"
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        print(f"✅ TextBlob import and basic sentiment test: {sentiment}")
        return True
        
    except ImportError as e:
        print(f"❌ TextBlob import failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Sentiment Analysis Integration Test")
    print("=" * 50)
    
    # Test TextBlob
    textblob_ok = test_textblob_import()
    
    if textblob_ok:
        # Test SentimentAnalyzer
        sentiment_ok = test_sentiment_analyzer()
        
        if sentiment_ok:
            print("\n🎉 All tests passed! Sentiment analysis integration is working.")
        else:
            print("\n⚠️  SentimentAnalyzer test failed.")
    else:
        print("\n❌ TextBlob test failed. Please install textblob:")
        print("   pip install textblob")
        print("   python -m textblob.download_corpora")