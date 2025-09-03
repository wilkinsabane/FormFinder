#!/usr/bin/env python3
"""Debug script to check provider status and cooldown state."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.news_manager import NewsProviderManager, ProviderConfig
from formfinder.news_monitoring import NewsProviderMonitor
from datetime import datetime
import time

def check_provider_status():
    """Check detailed provider status and cooldown information."""
    print("=== Provider Status Debug ===\n")
    
    # Load configuration
    load_config()
    config = get_config()
    
    # Create provider configs
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
    
    # Create monitor and manager
    monitor = NewsProviderMonitor()
    manager = NewsProviderManager(provider_configs, monitor=monitor)
    
    print(f"Current time: {datetime.now()}\n")
    
    # Check each provider in detail
    for name, provider in manager.providers.items():
        print(f"=== {name.upper()} ===")
        print(f"Provider status: {provider.status.value}")
        print(f"Is available (provider): {provider.is_available()}")
        print(f"Rate limit reset time: {provider.rate_limit_reset_time}")
        
        if provider.rate_limit_reset_time > 0:
            reset_datetime = datetime.fromtimestamp(provider.rate_limit_reset_time)
            print(f"Rate limit reset datetime: {reset_datetime}")
            time_until_reset = provider.rate_limit_reset_time - time.time()
            print(f"Time until reset: {time_until_reset:.2f} seconds")
        
        # Check monitoring health
        health_info = monitor.get_provider_health(name)
        print(f"Monitor status: {health_info['status']}")
        print(f"Is in cooldown (monitor): {health_info['is_in_cooldown']}")
        print(f"Total requests: {health_info['total_requests']}")
        print(f"Success rate: {health_info['success_rate']:.1f}%")
        print(f"Consecutive failures: {health_info['consecutive_failures']}")
        
        if 'cooldown_until' in health_info:
            print(f"Cooldown until: {health_info['cooldown_until']}")
        
        print()
    
    # Check available providers
    available = manager._get_available_providers()
    print(f"Available providers: {available}")
    print(f"Number of available: {len(available)}")
    
    if not available:
        print("\n=== NO PROVIDERS AVAILABLE - DETAILED ANALYSIS ===")
        for name, provider in manager.providers.items():
            print(f"\n{name}:")
            print(f"  Provider available: {provider.is_available()}")
            health_info = monitor.get_provider_health(name)
            print(f"  Monitor healthy: {health_info['status'] in ['healthy', 'unknown']}")
            print(f"  Not in cooldown: {not health_info['is_in_cooldown']}")
            
            # Check why it's not available
            if not provider.is_available():
                print(f"  REASON: Provider status is {provider.status.value}")
                if provider.status.value == 'RATE_LIMITED':
                    print(f"  Rate limit until: {datetime.fromtimestamp(provider.rate_limit_reset_time)}")
            
            if health_info['status'] not in ['healthy', 'unknown']:
                print(f"  REASON: Monitor status is {health_info['status']}")
            
            if health_info['is_in_cooldown']:
                print(f"  REASON: In monitor cooldown until {health_info.get('cooldown_until', 'unknown')}")
    
    return manager, monitor

if __name__ == "__main__":
    check_provider_status()