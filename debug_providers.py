#!/usr/bin/env python3
"""Debug script to investigate provider availability issues."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import get_config, load_config
from formfinder.sentiment import SentimentAnalyzer
from formfinder.news_manager import NewsProviderManager
from formfinder.news_monitoring import NewsProviderMonitor
from formfinder.news_manager import ProviderConfig

def debug_providers():
    """Debug provider availability issues."""
    print("=== Provider Debug Information ===")
    
    # Load configuration
    load_config()
    config = get_config()
    print(f"\nSentiment analysis enabled: {config.sentiment_analysis.enabled}")
    print(f"Failover enabled: {config.sentiment_analysis.enable_failover}")
    
    # Check provider configurations
    print("\n=== Provider Configurations ===")
    for name, provider_config in config.sentiment_analysis.providers.items():
        print(f"{name}:")
        print(f"  Enabled: {provider_config.enabled}")
        print(f"  Has API key: {bool(provider_config.api_key)}")
        print(f"  Priority: {provider_config.priority}")
    
    # Create provider configs for manager
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
    
    print(f"\n=== Provider Manager Initialization ===")
    print(f"Number of enabled provider configs: {len(provider_configs)}")
    
    # Create monitor and manager
    monitor = NewsProviderMonitor()
    manager = NewsProviderManager(provider_configs, monitor=monitor)
    
    print(f"Number of initialized providers: {len(manager.providers)}")
    print(f"Provider names: {list(manager.providers.keys())}")
    
    # Check individual provider status
    print("\n=== Individual Provider Status ===")
    for name, provider in manager.providers.items():
        print(f"{name}:")
        print(f"  Status: {provider.status.value}")
        print(f"  Is available: {provider.is_available()}")
        
        # Check monitor health (before any requests)
        health_info = monitor.get_provider_health(name)
        print(f"  Monitor status: {health_info['status']}")
        print(f"  Is in cooldown: {health_info['is_in_cooldown']}")
        print(f"  Total requests: {health_info['total_requests']}")
    
    # Check available providers
    print("\n=== Available Providers Check ===")
    available = manager._get_available_providers()
    print(f"Available providers: {available}")
    print(f"Number of available providers: {len(available)}")
    
    # Initialize provider health by recording a dummy successful request
    print("\n=== Initializing Provider Health ===")
    for name in manager.providers.keys():
        monitor.record_request(name, "initialization", True, 1.0)
        health_info = monitor.get_provider_health(name)
        print(f"{name} health after initialization: {health_info['status']}")
    
    # Check available providers again
    available_after_init = manager._get_available_providers()
    print(f"\nAvailable providers after health init: {available_after_init}")
    print(f"Number of available providers after init: {len(available_after_init)}")
    
    # Test SentimentAnalyzer initialization
    print("\n=== SentimentAnalyzer Test ===")
    try:
        analyzer = SentimentAnalyzer()
        print(f"SentimentAnalyzer initialized successfully")
        print(f"Using manager: {analyzer.use_manager}")
        print(f"Monitor enabled: {analyzer.monitor_enabled}")
        
        if analyzer.news_manager:
            available_in_analyzer = analyzer.news_manager._get_available_providers()
            print(f"Available providers in analyzer: {available_in_analyzer}")
    except Exception as e:
        print(f"SentimentAnalyzer initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_providers()