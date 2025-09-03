#!/usr/bin/env python3
"""
Fix for provider health recovery after rate limiting.

The issue is that providers get marked as unhealthy due to consecutive failures
from rate limiting, but don't automatically recover their health status when
the rate limit period expires.
"""

import logging
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.news_manager import NewsProviderManager, ProviderConfig
from formfinder.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_provider_recovery():
    """Test provider recovery after rate limiting."""
    try:
        # Load configuration
        config = load_config()
        
        # Convert sentiment config to provider configs
        provider_configs = []
        for name, provider_config in config.sentiment_analysis.providers.items():
            if provider_config.enabled:
                provider_configs.append(ProviderConfig(
                    name=name,
                    api_key=provider_config.api_key,
                    priority=provider_config.priority,
                    enabled=provider_config.enabled,
                    max_retries=provider_config.max_retries
                ))
        
        # Initialize news manager
        news_manager = NewsProviderManager(provider_configs)
        
        logger.info("Testing provider recovery...")
        
        # Check initial provider status
        available_providers = news_manager._get_available_providers()
        logger.info(f"Initially available providers: {available_providers}")
        
        # Get health status for all providers
        for provider_name in news_manager.providers.keys():
            health = news_manager.monitor.get_provider_health(provider_name)
            provider_available = news_manager.providers[provider_name].is_available()
            logger.info(f"Provider {provider_name}:")
            logger.info(f"  - Health status: {health['status']}")
            logger.info(f"  - In cooldown: {health['is_in_cooldown']}")
            logger.info(f"  - Provider available: {provider_available}")
            logger.info(f"  - Consecutive failures: {health['consecutive_failures']}")
            logger.info(f"  - Success rate: {health['success_rate']:.1f}%")
            
            # If provider is available but marked unhealthy due to rate limiting,
            # reset its health status
            if (provider_available and 
                health['status'] == 'unhealthy' and 
                not health['is_in_cooldown'] and
                health['consecutive_failures'] > 0):
                
                logger.info(f"Resetting health for recovered provider: {provider_name}")
                
                # Reset consecutive failures to allow provider to be considered healthy
                metrics = news_manager.monitor.provider_metrics[provider_name]
                metrics.consecutive_failures = 0
                metrics.is_healthy = True
                
                logger.info(f"Provider {provider_name} health status reset")
        
        # Check available providers after recovery
        available_providers_after = news_manager._get_available_providers()
        logger.info(f"Available providers after recovery: {available_providers_after}")
        
        # Test fetching articles
        if available_providers_after:
            logger.info("Testing article fetch with recovered providers...")
            response = news_manager.fetch_articles(
                query='football test',
                page_size=5
            )
            
            if response.success:
                logger.info(f"Successfully fetched {len(response.articles)} articles")
            else:
                logger.error(f"Failed to fetch articles: {response.error_message}")
        else:
            logger.warning("No providers available for testing")
            
    except Exception as e:
        logger.error(f"Error during provider recovery test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_provider_recovery()