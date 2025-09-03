#!/usr/bin/env python3
"""
Fix for provider availability issue by implementing singleton pattern for NewsProviderMonitor.

The issue: Each SentimentAnalyzer creates its own NewsProviderManager with its own 
NewsProviderMonitor, so rate limiting state is not shared between instances.

The solution: Use a singleton pattern for NewsProviderMonitor to ensure all instances
share the same monitoring state.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.news_monitoring import NewsProviderMonitor
from formfinder.sentiment import SentimentAnalyzer
from formfinder.config import load_config, get_config
from datetime import datetime


class SingletonNewsProviderMonitor:
    """
    Singleton wrapper for NewsProviderMonitor to ensure shared state.
    """
    _instance = None
    _monitor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._monitor = NewsProviderMonitor(log_dir="data/logs/news_monitoring")
        return cls._instance
    
    def get_monitor(self):
        """Get the singleton monitor instance."""
        return self._monitor


def create_modified_sentiment_analyzer():
    """
    Create a SentimentAnalyzer that uses the singleton monitor.
    """
    # Load configuration
    load_config()
    config = get_config()
    
    # Get the singleton monitor
    singleton_monitor = SingletonNewsProviderMonitor()
    monitor = singleton_monitor.get_monitor()
    
    # Create analyzer with shared monitor
    analyzer = SentimentAnalyzer()
    
    # Replace the monitor with our singleton
    if analyzer.news_manager:
        analyzer.news_manager.monitor = monitor
        analyzer.monitor = monitor
    
    return analyzer


def test_singleton_fix():
    """
    Test that the singleton fix resolves the provider availability issue.
    """
    print("=== Testing Singleton Monitor Fix ===")
    print(f"Test started at: {datetime.now()}\n")
    
    # Test 1: Create first analyzer and trigger rate limit
    print("1. Creating first SentimentAnalyzer with singleton monitor...")
    analyzer1 = create_modified_sentiment_analyzer()
    
    print("   Making sentiment analysis request to trigger rate limit...")
    try:
        result1 = analyzer1.get_sentiment_for_match(
            home_team="Hradec Kralove",
            away_team="Sigma Olomouc", 
            match_date=datetime.now()
        )
        print(f"   Result: {result1}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Create second analyzer and check if it sees the rate limit
    print("\n2. Creating second SentimentAnalyzer with singleton monitor...")
    analyzer2 = create_modified_sentiment_analyzer()
    
    print("   Checking provider status in second analyzer...")
    if analyzer2.news_manager:
        status = analyzer2.news_manager.get_provider_status()
        for name, info in status.items():
            print(f"   {name}: available={info['available']}, in_cooldown={info['is_in_cooldown']}")
            if info['rate_limit_reset_time'] > 0:
                reset_time = datetime.fromtimestamp(info['rate_limit_reset_time'])
                print(f"     Rate limit until: {reset_time}")
    
    # Test 3: Verify both analyzers share the same monitor
    print("\n3. Verifying monitor sharing...")
    if analyzer1.monitor is analyzer2.monitor:
        print("   ✅ SUCCESS: Both analyzers share the same monitor instance")
    else:
        print("   ❌ FAILURE: Analyzers have different monitor instances")
    
    if analyzer1.news_manager and analyzer2.news_manager:
        if analyzer1.news_manager.monitor is analyzer2.news_manager.monitor:
            print("   ✅ SUCCESS: Both managers share the same monitor instance")
        else:
            print("   ❌ FAILURE: Managers have different monitor instances")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_singleton_fix()