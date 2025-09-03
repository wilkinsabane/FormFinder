#!/usr/bin/env python3
"""
Test script to verify the provider health recovery fix.

This script simulates the original "No providers available" issue and verifies
that the automatic recovery mechanism now works correctly.
"""

import logging
from pathlib import Path
import sys
import time

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.news_manager import NewsProviderManager, ProviderConfig
from formfinder.news_monitoring import NewsProviderMonitor
from formfinder.config import load_config
from formfinder.news_providers import ProviderStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_provider_recovery_fix():
    """
    Test the provider recovery fix by simulating the original issue.
    """
    logger.info("=== Testing Provider Recovery Fix ===")
    
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
        
        logger.info(f"Testing with providers: {[pc.name for pc in provider_configs]}")
        
        # Create news manager
        news_manager = NewsProviderManager(provider_configs)
        monitor = news_manager.monitor
        
        # Step 1: Check initial state
        logger.info("\n1. Initial State:")
        initial_available = news_manager._get_available_providers()
        logger.info(f"Available providers: {initial_available}")
        
        if not initial_available:
            logger.error("No providers available initially - check API keys")
            return False
        
        # Step 2: Simulate the "No providers available" issue
        logger.info("\n2. Simulating 'No providers available' issue:")
        logger.info("Marking all providers as unhealthy due to consecutive failures...")
        
        for provider_name in news_manager.providers.keys():
            # Simulate consecutive failures that would mark provider as unhealthy
            if provider_name not in monitor.provider_metrics:
                # Initialize metrics if not exists
                monitor.record_request(provider_name, "test", False, 1.0)
            
            metrics = monitor.provider_metrics[provider_name]
            metrics.consecutive_failures = 10  # High failure count
            metrics.is_healthy = False
            logger.info(f"  {provider_name}: marked unhealthy (consecutive_failures=10)")
        
        # Check that no providers are available (reproducing the original issue)
        available_after_failure = news_manager._get_available_providers()
        logger.info(f"Available providers after simulated failures: {available_after_failure}")
        
        if len(available_after_failure) == 0:
            logger.info("✓ Successfully reproduced 'No providers available' issue")
        else:
            logger.warning(f"⚠ Expected no providers, but got: {available_after_failure}")
        
        # Step 3: Test automatic recovery
        logger.info("\n3. Testing Automatic Recovery:")
        logger.info("Calling _get_available_providers() again to trigger auto-recovery...")
        
        # The fix should automatically recover providers when _get_available_providers is called
        recovered_available = news_manager._get_available_providers()
        logger.info(f"Available providers after recovery attempt: {recovered_available}")
        
        # Step 4: Verify recovery
        logger.info("\n4. Recovery Verification:")
        
        if len(recovered_available) > len(available_after_failure):
            logger.info(f"✓ SUCCESS: Recovered {len(recovered_available)} provider(s)")
            logger.info("✓ Automatic recovery mechanism is working correctly")
            recovery_success = True
        else:
            logger.error("✗ FAILED: No providers recovered automatically")
            recovery_success = False
        
        # Step 5: Test article fetching
        if recovered_available:
            logger.info("\n5. Testing Article Fetch After Recovery:")
            response = news_manager.fetch_articles(
                query='test recovery',
                page_size=2
            )
            
            if response.success:
                logger.info(f"✓ Successfully fetched {len(response.articles)} articles")
                fetch_success = True
            else:
                logger.error(f"✗ Failed to fetch articles: {response.error_message}")
                fetch_success = False
        else:
            logger.warning("⚠ Skipping article fetch test - no providers available")
            fetch_success = False
        
        # Step 6: Final verification
        logger.info("\n6. Final Verification:")
        final_available = news_manager._get_available_providers()
        logger.info(f"Final available providers: {final_available}")
        
        # Check provider health status
        for provider_name in news_manager.providers.keys():
            health = monitor.get_provider_health(provider_name)
            logger.info(f"  {provider_name}: {health['status']} (failures: {health['consecutive_failures']})")
        
        return recovery_success and (fetch_success or len(final_available) > 0)
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_recovery_cycles():
    """
    Test multiple recovery cycles to ensure the fix is robust.
    """
    logger.info("\n=== Testing Multiple Recovery Cycles ===")
    
    try:
        # Load configuration
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
        
        success_count = 0
        total_cycles = 3
        
        for cycle in range(1, total_cycles + 1):
            logger.info(f"\nCycle {cycle}/{total_cycles}:")
            
            # Mark providers as unhealthy
            for provider_name in news_manager.providers.keys():
                if provider_name in monitor.provider_metrics:
                    metrics = monitor.provider_metrics[provider_name]
                    metrics.consecutive_failures = 5
                    metrics.is_healthy = False
            
            # Test recovery
            available = news_manager._get_available_providers()
            if available:
                logger.info(f"  ✓ Cycle {cycle}: {len(available)} provider(s) recovered")
                success_count += 1
            else:
                logger.error(f"  ✗ Cycle {cycle}: No providers recovered")
        
        logger.info(f"\nMultiple cycle test: {success_count}/{total_cycles} cycles successful")
        return success_count == total_cycles
        
    except Exception as e:
        logger.error(f"Error during multiple cycle test: {e}")
        return False

if __name__ == "__main__":
    # Test the provider recovery fix
    logger.info("Starting provider recovery fix tests...")
    
    # Test 1: Basic recovery functionality
    basic_test_success = test_provider_recovery_fix()
    
    # Test 2: Multiple recovery cycles
    cycle_test_success = test_multiple_recovery_cycles()
    
    # Final results
    logger.info("\n" + "="*50)
    logger.info("FINAL TEST RESULTS:")
    logger.info("="*50)
    
    if basic_test_success:
        logger.info("✓ Basic recovery test: PASSED")
    else:
        logger.error("✗ Basic recovery test: FAILED")
    
    if cycle_test_success:
        logger.info("✓ Multiple cycle test: PASSED")
    else:
        logger.error("✗ Multiple cycle test: FAILED")
    
    if basic_test_success and cycle_test_success:
        logger.info("\n🎉 ALL TESTS PASSED! The provider recovery fix is working correctly.")
        logger.info("The 'No providers available' issue should now be resolved.")
    else:
        logger.error("\n❌ SOME TESTS FAILED. The fix may need additional work.")
    
    logger.info("\nNext steps:")
    logger.info("1. Run your original sentiment analysis to verify the fix")
    logger.info("2. Monitor the logs for any remaining 'No providers available' errors")
    logger.info("3. Consider implementing additional monitoring and alerting")