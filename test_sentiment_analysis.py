#!/usr/bin/env python3
"""Test script to simulate sentiment analysis and debug provider failures."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.sentiment import SentimentAnalyzer
from datetime import datetime as dt
import time

def test_sentiment_analysis():
    """Test sentiment analysis with a real team name."""
    print("=== Sentiment Analysis Test ===\n")
    
    # Load configuration
    load_config()
    config = get_config()
    
    print(f"Sentiment analysis enabled: {config.sentiment_analysis.enabled}")
    print(f"Current time: {dt.now()}\n")
    
    # Initialize sentiment analyzer
    try:
        analyzer = SentimentAnalyzer()
        print(f"SentimentAnalyzer initialized successfully")
        print(f"Using manager: {analyzer.use_manager}")
        print(f"Monitor enabled: {analyzer.monitor_enabled}")
        
        if hasattr(analyzer, 'manager') and analyzer.manager:
            available = analyzer.manager._get_available_providers()
            print(f"Available providers: {available}")
            print(f"Number of available: {len(available)}\n")
        
    except Exception as e:
        print(f"Failed to initialize SentimentAnalyzer: {e}")
        return
    
    # Test with a real team name that should have news
    test_team = "Hradec Kralove"
    print(f"Testing sentiment analysis for: {test_team}")
    
    try:
        # Perform sentiment analysis
        print("\nPerforming sentiment analysis for Hradec Kralove vs Sigma Olomouc...")
        result = analyzer.get_sentiment_for_match(
            home_team="Hradec Kralove",
            away_team="Sigma Olomouc", 
            match_date=dt.now()
        )
        print(f"Sentiment result: {result}")
        
        # Check provider status after the request
        if hasattr(analyzer, 'manager') and analyzer.manager:
            print("\n=== Provider Status After Request ===")
            available = analyzer.manager._get_available_providers()
            print(f"Available providers: {available}")
            
            for name, provider in analyzer.manager.providers.items():
                print(f"\n{name}:")
                print(f"  Status: {provider.status.value}")
                print(f"  Available: {provider.is_available()}")
                print(f"  Rate limit reset: {provider.rate_limit_reset_time}")
                
                if provider.rate_limit_reset_time > 0:
                    reset_datetime = datetime.fromtimestamp(provider.rate_limit_reset_time)
                    print(f"  Reset datetime: {reset_datetime}")
                    time_until_reset = provider.rate_limit_reset_time - time.time()
                    print(f"  Time until reset: {time_until_reset:.2f} seconds")
                
                # Check monitoring health
                health_info = analyzer.manager.monitor.get_provider_health(name)
                print(f"  Monitor status: {health_info['status']}")
                print(f"  In cooldown: {health_info['is_in_cooldown']}")
                print(f"  Total requests: {health_info['total_requests']}")
                print(f"  Success rate: {health_info['success_rate']:.1f}%")
                
                if 'cooldown_until' in health_info:
                    print(f"  Cooldown until: {health_info['cooldown_until']}")
        
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Check provider status after error
        if hasattr(analyzer, 'manager') and analyzer.manager:
            print("\n=== Provider Status After Error ===")
            available = analyzer.manager._get_available_providers()
            print(f"Available providers: {available}")
            
            for name, provider in analyzer.manager.providers.items():
                print(f"\n{name}:")
                print(f"  Status: {provider.status.value}")
                print(f"  Available: {provider.is_available()}")
                
                health_info = analyzer.manager.monitor.get_provider_health(name)
                print(f"  Monitor status: {health_info['status']}")
                print(f"  In cooldown: {health_info['is_in_cooldown']}")

if __name__ == "__main__":
    test_sentiment_analysis()