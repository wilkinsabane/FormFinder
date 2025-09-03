#!/usr/bin/env python3
"""
Direct test of NewsProviderManager to check for "No providers available" error.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.sentiment import SentimentAnalyzer
import logging

# Set up logging to see all messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_news_manager_direct():
    """Test NewsProviderManager directly to reproduce the error."""
    print("=== Direct NewsProviderManager Test ===")
    
    # Load configuration
    load_config()
    
    # Create SentimentAnalyzer (which creates NewsProviderManager)
    print("\n1. Creating SentimentAnalyzer...")
    analyzer = SentimentAnalyzer()
    
    if not analyzer.news_manager:
        print("   ERROR: No news_manager created!")
        return
    
    manager = analyzer.news_manager
    print(f"   NewsProviderManager created with {len(manager.providers)} providers")
    
    # Check available providers
    print("\n2. Checking available providers...")
    available = manager._get_available_providers()
    print(f"   Available providers: {available}")
    print(f"   Number available: {len(available)}")
    
    if len(available) == 0:
        print("   ERROR: No providers available! This reproduces the issue.")
        
        # Debug each provider
        print("\n   Debugging individual providers:")
        for name, provider in manager.providers.items():
            print(f"     {name}:")
            print(f"       is_available(): {provider.is_available()}")
            health = manager.monitor.get_provider_health(name)
            print(f"       health: {health}")
    else:
        print("   SUCCESS: Providers are available!")
    
    # Try to fetch articles
    print("\n3. Testing article fetch...")
    try:
        response = manager.fetch_articles("test query")
        print(f"   Fetch result: success={response.success}")
        if response.success:
            print(f"   Articles found: {len(response.articles)}")
        else:
            print(f"   Error: {response.error_message}")
    except Exception as e:
        print(f"   Exception during fetch: {e}")
    
    # Check available providers again
    print("\n4. Checking available providers after fetch...")
    available_after = manager._get_available_providers()
    print(f"   Available providers: {available_after}")
    print(f"   Number available: {len(available_after)}")

if __name__ == "__main__":
    test_news_manager_direct()