#!/usr/bin/env python
"""
Test script to demonstrate and debug the rate limiting issue.
This script shows how rate limiting affects provider availability.
"""

import time
from datetime import datetime
from formfinder.config import load_config, get_config
from formfinder.sentiment import SentimentAnalyzer
from formfinder.news_manager import NewsProviderManager
from formfinder.news_monitoring import NewsProviderMonitor
from formfinder.news_manager import ProviderConfig

def test_rate_limit_behavior():
    """Test rate limiting behavior and provider availability."""
    print("=== Rate Limit Issue Debug Test ===")
    print(f"Test started at: {datetime.now()}\n")
    
    # Load configuration
    load_config()
    config = get_config()
    
    # Test 1: Create SentimentAnalyzer and trigger rate limiting
    print("1. Testing SentimentAnalyzer rate limiting...")
    analyzer = SentimentAnalyzer(
        api_key=config.sentiment_analysis.news_api_key if config.sentiment_analysis else None,
        cache_hours=24
    )
    
    # Make a request that should trigger rate limiting
    print("   Making sentiment analysis request...")
    result = analyzer.get_sentiment_for_match(
        home_team="Hradec Kralove",
        away_team="Sigma Olomouc",
        match_date=datetime.now()
    )
    print(f"   Result: {result}")
    
    # Test 2: Check provider status in the same analyzer instance
    print("\n2. Checking provider status in same analyzer instance...")
    if hasattr(analyzer, 'manager') and analyzer.manager:
        status = analyzer.manager.get_provider_status()
        for name, info in status.items():
            print(f"   {name}: available={info['available']}, in_cooldown={info['is_in_cooldown']}")
            if info['rate_limit_reset_time'] > 0:
                reset_time = datetime.fromtimestamp(info['rate_limit_reset_time'])
                print(f"     Rate limit until: {reset_time}")
    
    # Test 3: Create a NEW analyzer instance and check availability
    print("\n3. Creating NEW SentimentAnalyzer instance...")
    analyzer2 = SentimentAnalyzer(
        api_key=config.sentiment_analysis.news_api_key if config.sentiment_analysis else None,
        cache_hours=24
    )
    
    print("   Checking provider status in new analyzer instance...")
    if hasattr(analyzer2, 'manager') and analyzer2.manager:
        status2 = analyzer2.manager.get_provider_status()
        for name, info in status2.items():
            print(f"   {name}: available={info['available']}, in_cooldown={info['is_in_cooldown']}")
            if info['rate_limit_reset_time'] > 0:
                reset_time = datetime.fromtimestamp(info['rate_limit_reset_time'])
                print(f"     Rate limit until: {reset_time}")
    
    # Test 4: Create a standalone NewsProviderManager
    print("\n4. Creating standalone NewsProviderManager...")
    provider_configs = []
    for name, provider_config in config.sentiment_analysis.providers.items():
        if provider_config.enabled:
            provider_configs.append(ProviderConfig(
                name=name,
                api_key=provider_config.api_key,
                enabled=provider_config.enabled,
                priority=provider_config.priority,
                max_retries=provider_config.max_retries
            ))
    
    monitor = NewsProviderMonitor()
    manager = NewsProviderManager(provider_configs, monitor=monitor)
    
    print("   Checking provider status in standalone manager...")
    status3 = manager.get_provider_status()
    for name, info in status3.items():
        print(f"   {name}: available={info['available']}, in_cooldown={info['is_in_cooldown']}")
        if info['rate_limit_reset_time'] > 0:
            reset_time = datetime.fromtimestamp(info['rate_limit_reset_time'])
            print(f"     Rate limit until: {reset_time}")
    
    # Test 5: Try to get available providers from each instance
    print("\n5. Testing _get_available_providers from each instance...")
    
    if hasattr(analyzer, 'manager') and analyzer.manager:
        available1 = analyzer.manager._get_available_providers()
        print(f"   Analyzer 1 available providers: {available1}")
    
    if hasattr(analyzer2, 'manager') and analyzer2.manager:
        available2 = analyzer2.manager._get_available_providers()
        print(f"   Analyzer 2 available providers: {available2}")
    
    available3 = manager._get_available_providers()
    print(f"   Standalone manager available providers: {available3}")
    
    # Test 6: Show the issue - different instances have different states
    print("\n6. ISSUE SUMMARY:")
    print("   - Each SentimentAnalyzer creates its own NewsProviderManager instance")
    print("   - Each NewsProviderManager creates its own NewsProviderMonitor instance")
    print("   - Rate limiting state is not shared between instances")
    print("   - This causes the 'No providers available' error when all providers are rate limited")
    
    return analyzer, analyzer2, manager

if __name__ == "__main__":
    test_rate_limit_behavior()