#!/usr/bin/env python3
"""
RSS-Based Sentiment Analysis System Demonstration

This script demonstrates the complete RSS-based sentiment analysis system,
showing how it integrates as a fallback when API providers are unavailable.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.sentiment import SentimentAnalyzer
from formfinder.rss_news_provider import create_default_rss_provider
from formfinder.rss_feed_manager import RSSConfig
from formfinder.rss_content_parser import ParsingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_rss_provider():
    """
    Demonstrate the RSS provider functionality independently.
    """
    print("\n" + "="*60)
    print("RSS PROVIDER DEMONSTRATION")
    print("="*60)
    
    try:
        # Create RSS provider
        rss_provider = create_default_rss_provider()
        print(f"✅ RSS Provider initialized: {rss_provider.config.name}")
        
        # Show provider info
        info = rss_provider.get_provider_info()
        print(f"📊 Provider Status: {'Healthy' if rss_provider.is_healthy else 'Unhealthy'}")
        print(f"📈 Available Sports: {', '.join(info.get('capabilities', {}).get('sports', []))}")
        print(f"🔗 Active Feeds: {info.get('statistics', {}).get('total_feeds', 0)}")
        
        # Test article fetching
        print("\n🔍 Testing article fetching...")
        test_queries = [
            ("Manchester United", "football"),
            ("Liverpool", "football"),
            ("Arsenal", "football")
        ]
        
        for team, sport in test_queries:
            print(f"\n📰 Fetching articles for {team}...")
            articles = rss_provider.fetch_articles(
                query=f"{team} football soccer",
                max_results=5,
                sport=sport
            )
            
            if articles:
                print(f"   ✅ Found {len(articles)} articles")
                for i, article in enumerate(articles[:2], 1):
                    print(f"   {i}. {article['title'][:80]}...")
                    print(f"      Source: {article['source']} | Published: {article['publishedAt']}")
            else:
                print(f"   ⚠️  No articles found for {team}")
        
        return True
        
    except Exception as e:
        print(f"❌ RSS Provider demonstration failed: {e}")
        logger.exception("RSS Provider demonstration error")
        return False

def demonstrate_sentiment_fallback():
    """
    Demonstrate sentiment analysis with RSS fallback.
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS WITH RSS FALLBACK")
    print("="*60)
    
    try:
        # Load configuration
        load_config()
        
        # Create sentiment analyzer (this will trigger RSS fallback if APIs unavailable)
        analyzer = SentimentAnalyzer()
        print(f"✅ Sentiment Analyzer initialized")
        print(f"📊 Using RSS fallback: {getattr(analyzer, 'use_rss', False)}")
        
        # Test sentiment analysis for a match
        print("\n🏆 Analyzing sentiment for Premier League match...")
        home_team = "Manchester United"
        away_team = "Liverpool"
        match_date = datetime.now()
        
        print(f"🏠 Home Team: {home_team}")
        print(f"🚌 Away Team: {away_team}")
        print(f"📅 Match Date: {match_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Perform sentiment analysis
        result = analyzer.get_sentiment_for_match(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            days_back=7
        )
        
        # Display results
        print("\n📈 SENTIMENT ANALYSIS RESULTS:")
        print("-" * 40)
        
        print(f"🏠 {home_team}:")
        print(f"   Sentiment Score: {result.home_sentiment:.3f}")
        print(f"   Article Count: {result.home_article_count}")
        
        print(f"\n🚌 {away_team}:")
        print(f"   Sentiment Score: {result.away_sentiment:.3f}")
        print(f"   Article Count: {result.away_article_count}")
        
        # Show data source information
        data_source = "RSS Feeds" if getattr(analyzer, 'use_rss', False) else "API Providers"
        print(f"\n📊 Data Source: {data_source}")
        
        if hasattr(analyzer, 'use_rss') and analyzer.use_rss:
            print("✅ Successfully used RSS feeds as fallback data source")
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis demonstration failed: {e}")
        logger.exception("Sentiment analysis demonstration error")
        return False

def demonstrate_system_capabilities():
    """
    Demonstrate the overall system capabilities and benefits.
    """
    print("\n" + "="*60)
    print("RSS SENTIMENT SYSTEM CAPABILITIES")
    print("="*60)
    
    capabilities = {
        "🔄 Automatic Fallback": "Seamlessly switches to RSS when APIs are unavailable",
        "📰 Multi-Source RSS": "Aggregates from BBC Sport, Sky Sports, ESPN, and more",
        "🎯 Smart Filtering": "Filters articles by team, sport, and relevance",
        "🧹 Content Cleaning": "Removes ads, extracts clean text, handles duplicates",
        "⚡ Efficient Caching": "Caches articles to reduce redundant fetching",
        "📊 Quality Scoring": "Ranks articles by content quality and relevance",
        "🏥 Health Monitoring": "Monitors feed health and availability",
        "🔧 Configurable": "Customizable feeds, intervals, and filtering rules"
    }
    
    print("\n🚀 KEY FEATURES:")
    for feature, description in capabilities.items():
        print(f"   {feature}: {description}")
    
    benefits = {
        "💰 Cost Reduction": "Reduces API costs by using free RSS feeds",
        "🛡️ Rate Limit Protection": "Bypasses API rate limits and quotas",
        "📈 Higher Availability": "Maintains service when APIs are down",
        "🌐 Broader Coverage": "Access to more diverse news sources",
        "⚡ Faster Response": "Cached content provides faster responses",
        "🔒 Independence": "Reduces dependency on third-party APIs"
    }
    
    print("\n💡 BUSINESS BENEFITS:")
    for benefit, description in benefits.items():
        print(f"   {benefit}: {description}")
    
    print("\n🔧 TECHNICAL ARCHITECTURE:")
    architecture = [
        "📋 RSS Feed Manager: Discovers and manages RSS feeds",
        "🔍 Content Parser: Extracts and cleans article content",
        "🎯 News Provider: Integrates with existing provider system",
        "💾 Caching Layer: Efficient storage and retrieval",
        "🏥 Health Monitor: Tracks feed status and performance",
        "⚙️ Configuration: Flexible setup and customization"
    ]
    
    for component in architecture:
        print(f"   {component}")

def main():
    """
    Main demonstration function.
    """
    print("🚀 RSS-BASED SENTIMENT ANALYSIS SYSTEM DEMO")
    print("=" * 60)
    print("This demonstration shows how RSS feeds provide a robust")
    print("fallback solution for sentiment analysis when API providers")
    print("are unavailable or rate-limited.")
    
    # Track results
    results = {
        'rss_provider': False,
        'sentiment_fallback': False
    }
    
    # Run demonstrations
    results['rss_provider'] = demonstrate_rss_provider()
    results['sentiment_fallback'] = demonstrate_sentiment_fallback()
    demonstrate_system_capabilities()
    
    # Summary
    print("\n" + "="*60)
    print("DEMONSTRATION SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All demonstrations completed successfully!")
        print("✅ RSS-based sentiment analysis system is fully operational")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} demonstration(s) failed")
        print("🔧 Check logs for detailed error information")
    
    print("\n💡 The RSS fallback system provides a reliable alternative")
    print("   to API-based news providers, ensuring continuous service")
    print("   availability and reducing operational costs.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)