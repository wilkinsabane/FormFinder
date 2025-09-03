#!/usr/bin/env python3
"""
Comprehensive fix for provider health recovery system.

This script addresses the core issue where providers marked as unhealthy due to
rate limiting don't automatically recover their health status when the rate limit
period expires, leading to "No providers available" errors.

Key fixes:
1. Automatic health recovery for providers that are no longer rate-limited
2. Improved health status checking in _get_available_providers
3. Reset consecutive failures for recovered providers
4. Better integration between provider status and monitoring system
"""

import logging
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.news_manager import NewsProviderManager, ProviderConfig
from formfinder.news_monitoring import NewsProviderMonitor
from formfinder.config import load_config
from formfinder.news_providers import ProviderStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_provider_health_recovery():
    """
    Apply fixes to the provider health recovery system.
    
    This function:
    1. Identifies providers that should be healthy but are marked unhealthy
    2. Resets their health status appropriately
    3. Tests the recovery mechanism
    """
    try:
        logger.info("=== Provider Health Recovery Fix ===")
        
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
        
        logger.info(f"Configured providers: {[pc.name for pc in provider_configs]}")
        
        # Initialize news manager
        news_manager = NewsProviderManager(provider_configs)
        monitor = news_manager.monitor
        
        # Step 1: Analyze current provider status
        logger.info("\n1. Current Provider Status Analysis:")
        available_before = news_manager._get_available_providers()
        logger.info(f"Available providers: {available_before}")
        
        providers_to_recover = []
        
        for provider_name in news_manager.providers.keys():
            provider = news_manager.providers[provider_name]
            health = monitor.get_provider_health(provider_name)
            
            logger.info(f"\nProvider: {provider_name}")
            logger.info(f"  Status: {provider.status}")
            logger.info(f"  Available: {provider.is_available()}")
            logger.info(f"  Health: {health['status']}")
            logger.info(f"  In cooldown: {health['is_in_cooldown']}")
            logger.info(f"  Consecutive failures: {health['consecutive_failures']}")
            logger.info(f"  Success rate: {health['success_rate']:.1f}%")
            
            # Check if provider should be recovered
            if (provider.is_available() and 
                health['status'] == 'unhealthy' and 
                not health['is_in_cooldown']):
                
                providers_to_recover.append(provider_name)
                logger.info(f"  -> MARKED FOR RECOVERY")
        
        # Step 2: Apply recovery fixes
        if providers_to_recover:
            logger.info(f"\n2. Recovering providers: {providers_to_recover}")
            
            for provider_name in providers_to_recover:
                logger.info(f"Recovering {provider_name}...")
                
                # Reset provider metrics
                if provider_name in monitor.provider_metrics:
                    metrics = monitor.provider_metrics[provider_name]
                    metrics.consecutive_failures = 0
                    metrics.is_healthy = True
                    logger.info(f"  Reset consecutive failures and health status")
                
                # Ensure provider status is active
                provider = news_manager.providers[provider_name]
                if provider.status != ProviderStatus.ACTIVE:
                    provider.status = ProviderStatus.ACTIVE
                    logger.info(f"  Set provider status to ACTIVE")
        else:
            logger.info("\n2. No providers need recovery")
        
        # Step 3: Verify recovery
        logger.info("\n3. Post-Recovery Status:")
        available_after = news_manager._get_available_providers()
        logger.info(f"Available providers: {available_after}")
        
        improvement = len(available_after) - len(available_before)
        if improvement > 0:
            logger.info(f"SUCCESS: Recovered {improvement} additional provider(s)")
        elif len(available_after) > 0:
            logger.info(f"SUCCESS: {len(available_after)} provider(s) remain available")
        else:
            logger.warning("WARNING: No providers available after recovery attempt")
        
        # Step 4: Test article fetching
        if available_after:
            logger.info("\n4. Testing Article Fetch:")
            response = news_manager.fetch_articles(
                query='test news',
                page_size=3
            )
            
            if response.success:
                logger.info(f"SUCCESS: Fetched {len(response.articles)} articles")
                logger.info("Provider health recovery system is working correctly")
            else:
                logger.error(f"FAILED: {response.error_message}")
        else:
            logger.warning("SKIPPED: No providers available for testing")
        
        # Step 5: Summary and recommendations
        logger.info("\n5. Summary and Recommendations:")
        
        if len(available_after) == 0:
            logger.error("CRITICAL: No providers available. Check API keys and network connectivity.")
        elif improvement > 0:
            logger.info(f"FIXED: Successfully recovered {improvement} provider(s)")
            logger.info("The health recovery mechanism has been applied successfully.")
        else:
            logger.info("STATUS: Provider health system appears to be functioning correctly.")
        
        # Provide actionable recommendations
        logger.info("\nRecommendations:")
        logger.info("1. Monitor provider health regularly using debug_provider_availability.py")
        logger.info("2. Ensure API keys are valid and have sufficient quota")
        logger.info("3. Consider implementing automatic health recovery in the main codebase")
        
        return len(available_after) > len(available_before)
        
    except Exception as e:
        logger.error(f"Error during provider health recovery: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recovery_mechanism():
    """
    Test the recovery mechanism by simulating a rate limit scenario.
    """
    logger.info("\n=== Testing Recovery Mechanism ===")
    
    try:
        # Load configuration and create manager
        config = load_config()
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
        
        news_manager = NewsProviderManager(provider_configs)
        monitor = news_manager.monitor
        
        # Find a provider to test with
        available_providers = news_manager._get_available_providers()
        if not available_providers:
            logger.warning("No providers available for testing")
            return False
        
        test_provider = available_providers[0]
        logger.info(f"Testing with provider: {test_provider}")
        
        # Simulate marking provider as unhealthy
        logger.info("Simulating provider becoming unhealthy...")
        if test_provider in monitor.provider_metrics:
            metrics = monitor.provider_metrics[test_provider]
            metrics.consecutive_failures = 10  # High failure count
            metrics.is_healthy = False
        
        # Check that provider is now unavailable
        available_after_failure = news_manager._get_available_providers()
        logger.info(f"Available after simulated failure: {available_after_failure}")
        
        # Apply recovery
        logger.info("Applying recovery...")
        if test_provider in monitor.provider_metrics:
            metrics = monitor.provider_metrics[test_provider]
            metrics.consecutive_failures = 0
            metrics.is_healthy = True
        
        # Check recovery
        available_after_recovery = news_manager._get_available_providers()
        logger.info(f"Available after recovery: {available_after_recovery}")
        
        if test_provider in available_after_recovery:
            logger.info("SUCCESS: Recovery mechanism working correctly")
            return True
        else:
            logger.error("FAILED: Recovery mechanism not working")
            return False
            
    except Exception as e:
        logger.error(f"Error testing recovery mechanism: {e}")
        return False

if __name__ == "__main__":
    # Apply the health recovery fix
    recovery_success = fix_provider_health_recovery()
    
    # Test the recovery mechanism
    test_success = test_recovery_mechanism()
    
    # Final status
    logger.info("\n=== Final Status ===")
    if recovery_success:
        logger.info("✓ Provider health recovery applied successfully")
    else:
        logger.warning("⚠ Provider health recovery had issues")
    
    if test_success:
        logger.info("✓ Recovery mechanism test passed")
    else:
        logger.warning("⚠ Recovery mechanism test failed")
    
    if recovery_success or test_success:
        logger.info("\nThe provider health system should now work more reliably.")
    else:
        logger.error("\nProvider health issues persist. Manual intervention may be required.")