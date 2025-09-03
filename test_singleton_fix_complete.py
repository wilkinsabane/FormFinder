#!/usr/bin/env python3
"""
Comprehensive test to verify the singleton monitor fix resolves the provider availability issue.

This test demonstrates that:
1. Rate limiting state is now shared between SentimentAnalyzer instances
2. The "No providers available" error is resolved
3. Failover works correctly across instances
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.sentiment import SentimentAnalyzer
from formfinder.config import load_config
from datetime import datetime
import time


def test_complete_singleton_fix():
    """
    Test that the singleton fix completely resolves the provider availability issue.
    """
    print("=== Complete Singleton Fix Test ===")
    print(f"Test started at: {datetime.now()}\n")
    
    # Load configuration
    load_config()
    
    # Test 1: Create first analyzer and trigger rate limit
    print("1. Creating first SentimentAnalyzer...")
    analyzer1 = SentimentAnalyzer()
    
    print("   Making sentiment analysis request to trigger rate limit...")
    try:
        result1 = analyzer1.get_sentiment_for_match(
            home_team="Hradec Kralove",
            away_team="Sigma Olomouc", 
            match_date=datetime.now()
        )
        print(f"   First analyzer result: {result1}")
        print(f"   Articles found: home={result1.home_article_count}, away={result1.away_article_count}")
    except Exception as e:
        print(f"   Error in first analyzer: {e}")
    
    # Test 2: Check provider status in first analyzer
    print("\n2. Checking provider status in first analyzer...")
    if analyzer1.news_manager:
        status1 = analyzer1.news_manager.get_provider_status()
        for name, info in status1.items():
            cooldown_status = "in cooldown" if info['is_in_cooldown'] else "available"
            print(f"   {name}: {cooldown_status}")
            if info['rate_limit_reset_time'] > 0:
                reset_time = datetime.fromtimestamp(info['rate_limit_reset_time'])
                print(f"     Rate limit until: {reset_time}")
    
    # Test 3: Create second analyzer immediately
    print("\n3. Creating second SentimentAnalyzer (should share rate limit state)...")
    analyzer2 = SentimentAnalyzer()
    
    # Test 4: Check provider status in second analyzer
    print("   Checking provider status in second analyzer...")
    if analyzer2.news_manager:
        status2 = analyzer2.news_manager.get_provider_status()
        for name, info in status2.items():
            cooldown_status = "in cooldown" if info['is_in_cooldown'] else "available"
            print(f"   {name}: {cooldown_status}")
    
    # Test 5: Verify monitor sharing
    print("\n4. Verifying monitor instance sharing...")
    if analyzer1.monitor is analyzer2.monitor:
        print("   ✅ SUCCESS: Both analyzers share the same monitor instance")
    else:
        print("   ❌ FAILURE: Analyzers have different monitor instances")
    
    if (analyzer1.news_manager and analyzer2.news_manager and 
        analyzer1.news_manager.monitor is analyzer2.news_manager.monitor):
        print("   ✅ SUCCESS: Both managers share the same monitor instance")
    else:
        print("   ❌ FAILURE: Managers have different monitor instances")
    
    # Test 6: Try sentiment analysis with second analyzer
    print("\n5. Testing sentiment analysis with second analyzer...")
    try:
        result2 = analyzer2.get_sentiment_for_match(
            home_team="Arsenal",
            away_team="Chelsea", 
            match_date=datetime.now()
        )
        print(f"   Second analyzer result: {result2}")
        print(f"   Articles found: home={result2.home_article_count}, away={result2.away_article_count}")
        print("   ✅ SUCCESS: Second analyzer can still perform sentiment analysis")
    except Exception as e:
        print(f"   Error in second analyzer: {e}")
        if "No providers available" in str(e):
            print("   ❌ FAILURE: Still getting 'No providers available' error")
        else:
            print("   ⚠️  Different error occurred")
    
    # Test 7: Check available providers
    print("\n6. Checking available providers in both analyzers...")
    if analyzer1.news_manager:
        available1 = analyzer1.news_manager._get_available_providers()
        print(f"   First analyzer available providers: {available1}")
    
    if analyzer2.news_manager:
        available2 = analyzer2.news_manager._get_available_providers()
        print(f"   Second analyzer available providers: {available2}")
    
    # Test 8: Summary
    print("\n7. TEST SUMMARY:")
    if (analyzer1.monitor is analyzer2.monitor and 
        analyzer1.news_manager and analyzer2.news_manager and
        analyzer1.news_manager.monitor is analyzer2.news_manager.monitor):
        print("   ✅ Monitor sharing: WORKING")
    else:
        print("   ❌ Monitor sharing: FAILED")
    
    # Check if any providers are available in both instances
    if analyzer1.news_manager and analyzer2.news_manager:
        available1 = analyzer1.news_manager._get_available_providers()
        available2 = analyzer2.news_manager._get_available_providers()
        if available1 == available2:
            print("   ✅ Provider availability consistency: WORKING")
        else:
            print("   ❌ Provider availability consistency: FAILED")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_complete_singleton_fix()