#!/usr/bin/env python3
"""
Debug script to investigate the "No providers available" issue.

This script will:
1. Check provider initialization and configuration
2. Test provider availability logic step by step
3. Examine monitoring health status
4. Identify why providers are marked as unavailable
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.sentiment import SentimentAnalyzer
from formfinder.news_manager import NewsProviderManager
from formfinder.news_monitoring import NewsProviderMonitor
from formfinder.news_providers import ProviderStatus
import time

def debug_provider_availability():
    """Debug provider availability issues."""
    print("=== Provider Availability Debug ===")
    
    # Load configuration
    config = load_config()
    print(f"\n1. Configuration loaded:")
    print(f"   Sentiment analysis enabled: {config.sentiment_analysis.enabled}")
    print(f"   Configured providers: {list(config.sentiment_analysis.providers.keys())}")
    
    for name, provider_config in config.sentiment_analysis.providers.items():
        print(f"   {name}: enabled={provider_config.enabled}, priority={provider_config.priority}")
        if provider_config.api_key:
            print(f"     API key: {provider_config.api_key[:10]}...")
        else:
            print(f"     API key: NOT SET")
    
    # Create SentimentAnalyzer (which creates NewsProviderManager)
    print(f"\n2. Creating SentimentAnalyzer...")
    analyzer = SentimentAnalyzer()  # No parameters needed, it loads config internally
    
    if not analyzer.news_manager:
        print("   ERROR: No news_manager created!")
        return
    
    manager = analyzer.news_manager
    monitor = manager.monitor
    
    print(f"   NewsProviderManager created with {len(manager.providers)} providers")
    print(f"   Provider names: {list(manager.providers.keys())}")
    
    # Check individual provider status
    print(f"\n3. Individual Provider Analysis:")
    for name, provider in manager.providers.items():
        print(f"\n   Provider: {name}")
        print(f"     Status: {provider.status.value}")
        print(f"     Rate limit reset time: {provider.rate_limit_reset_time}")
        print(f"     Current time: {time.time()}")
        print(f"     is_available(): {provider.is_available()}")
        
        # Check monitoring health
        health_info = monitor.get_provider_health(name)
        print(f"     Monitor status: {health_info['status']}")
        print(f"     Is in cooldown: {health_info['is_in_cooldown']}")
        print(f"     Total requests: {health_info['total_requests']}")
        print(f"     Success rate: {health_info['success_rate']:.1f}%")
        print(f"     Consecutive failures: {health_info['consecutive_failures']}")
        
        # Check if provider would be included in available list
        provider_available = provider.is_available()
        health_ok = health_info['status'] in ['healthy', 'unknown']
        not_in_cooldown = not health_info['is_in_cooldown']
        
        print(f"     Would be available: {provider_available and health_ok and not_in_cooldown}")
        print(f"       - provider.is_available(): {provider_available}")
        print(f"       - health status OK: {health_ok} (status: {health_info['status']})")
        print(f"       - not in cooldown: {not_in_cooldown}")
    
    # Check _get_available_providers result
    print(f"\n4. Available Providers Check:")
    available = manager._get_available_providers()
    print(f"   Available providers: {available}")
    print(f"   Number available: {len(available)}")
    
    if not available:
        print(f"   \n   ISSUE IDENTIFIED: No providers available!")
        print(f"   Checking why each provider is excluded:")
        
        for name, provider in manager.providers.items():
            health_info = monitor.get_provider_health(name)
            
            print(f"\n   {name}:")
            if not provider.is_available():
                print(f"     ❌ Provider not available: status={provider.status.value}")
                if provider.status == ProviderStatus.RATE_LIMITED:
                    print(f"        Rate limited until: {provider.rate_limit_reset_time}")
                    print(f"        Current time: {time.time()}")
                    print(f"        Time remaining: {provider.rate_limit_reset_time - time.time():.1f}s")
            elif health_info['status'] not in ['healthy', 'unknown']:
                print(f"     ❌ Health status not OK: {health_info['status']}")
            elif health_info['is_in_cooldown']:
                print(f"     ❌ In cooldown: {health_info.get('cooldown_until', 'unknown')}")
            else:
                print(f"     ✅ Should be available but isn't - this is the bug!")
    
    # Test a simple fetch to see what happens
    print(f"\n5. Testing Article Fetch:")
    try:
        response = manager.fetch_articles("test query", max_articles=1)
        print(f"   Fetch result: success={response.success}")
        if not response.success:
            print(f"   Error: {response.error_message}")
        else:
            print(f"   Articles found: {len(response.articles)}")
    except Exception as e:
        print(f"   Exception during fetch: {e}")
    
    # Check provider status after fetch attempt
    print(f"\n6. Provider Status After Fetch Attempt:")
    available_after = manager._get_available_providers()
    print(f"   Available providers after fetch: {available_after}")
    
    for name, provider in manager.providers.items():
        health_info = monitor.get_provider_health(name)
        print(f"   {name}: status={provider.status.value}, health={health_info['status']}, cooldown={health_info['is_in_cooldown']}")
    
    return manager, monitor

if __name__ == "__main__":
    debug_provider_availability()