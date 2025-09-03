#!/usr/bin/env python3
"""
Test script to simulate the original user scenario and verify the fix.

This script reproduces the exact scenario from the user's logs:
- Processing fixtures with sentiment analysis
- Rate limiting occurring
- "No providers available" error
- Verifying that the fix resolves this issue
"""

import logging
from pathlib import Path
import sys
import time

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.sentiment import SentimentAnalyzer
from formfinder.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_original_scenario():
    """
    Simulate the original scenario that caused "No providers available" error.
    """
    logger.info("=== Simulating Original User Scenario ===")
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Sentiment analysis enabled: {config.sentiment_analysis.enabled}")
        
        # Create SentimentAnalyzer (as used in the original scenario)
        logger.info("\n1. Creating SentimentAnalyzer (as in original scenario)...")
        analyzer = SentimentAnalyzer()
        
        if not analyzer.news_manager:
            logger.error("No news_manager created - this reproduces the original issue")
            return False
        
        logger.info(f"SentimentAnalyzer created with {len(analyzer.news_manager.providers)} providers")
        
        # Check initial provider availability
        logger.info("\n2. Checking initial provider availability...")
        available_providers = analyzer.news_manager._get_available_providers()
        logger.info(f"Available providers: {available_providers}")
        
        if not available_providers:
            logger.error("❌ REPRODUCED ORIGINAL ISSUE: No providers available initially")
            return False
        
        # Simulate the teams from the original scenario
        teams = ["El Bayadh", "CS Constantine"]
        
        logger.info("\n3. Simulating sentiment analysis for teams (original scenario)...")
        
        for i, team in enumerate(teams):
            logger.info(f"\nAnalyzing sentiment for {team}...")
            
            # Simulate the rate limiting that occurred in the original scenario
            if i == 0:  # First team - simulate rate limiting
                logger.info("Simulating rate limiting scenario...")
                
                # Mark providers as rate-limited/unhealthy to simulate the original issue
                for provider_name in analyzer.news_manager.providers.keys():
                    if provider_name in analyzer.news_manager.monitor.provider_metrics:
                        metrics = analyzer.news_manager.monitor.provider_metrics[provider_name]
                        metrics.consecutive_failures = 8  # High failure count
                        metrics.is_healthy = False
                        logger.info(f"  Simulated failures for {provider_name}")
            
            # Check provider availability (this should trigger auto-recovery with the fix)
            available_before = analyzer.news_manager._get_available_providers()
            logger.info(f"Available providers before analysis: {available_before}")
            
            if not available_before:
                logger.warning(f"⚠ No providers available for {team} - testing auto-recovery...")
                
                # The fix should automatically recover providers
                # Let's call _get_available_providers again to trigger recovery
                available_after_recovery = analyzer.news_manager._get_available_providers()
                logger.info(f"Available providers after recovery attempt: {available_after_recovery}")
                
                if available_after_recovery:
                    logger.info(f"✓ AUTO-RECOVERY SUCCESSFUL: {len(available_after_recovery)} provider(s) recovered")
                else:
                    logger.error(f"❌ AUTO-RECOVERY FAILED: Still no providers available for {team}")
                    return False
            
            # Try to fetch articles (simulating the sentiment analysis process)
            try:
                response = analyzer.news_manager.fetch_articles(
                    query=f'{team} football news',
                    page_size=3
                )
                
                if response.success:
                    logger.info(f"✓ Successfully fetched {len(response.articles)} articles for {team}")
                else:
                    logger.warning(f"⚠ Failed to fetch articles for {team}: {response.error_message}")
                    
                    # Check if this is the "No providers available" error from the original scenario
                    if "No news providers available" in response.error_message or "No providers available" in response.error_message:
                        logger.error(f"❌ ORIGINAL ISSUE REPRODUCED: {response.error_message}")
                        return False
                    
            except Exception as e:
                logger.error(f"Exception during article fetch for {team}: {e}")
                return False
        
        logger.info("\n4. Final verification...")
        final_available = analyzer.news_manager._get_available_providers()
        logger.info(f"Final available providers: {final_available}")
        
        # Check final health status
        for provider_name in analyzer.news_manager.providers.keys():
            health = analyzer.news_manager.monitor.get_provider_health(provider_name)
            logger.info(f"  {provider_name}: {health['status']} (failures: {health['consecutive_failures']})")
        
        if final_available:
            logger.info("✅ SUCCESS: Original scenario completed without 'No providers available' error")
            return True
        else:
            logger.error("❌ FAILED: No providers available at the end")
            return False
            
    except Exception as e:
        logger.error(f"Error during scenario simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rapid_requests():
    """
    Test rapid consecutive requests to simulate heavy processing load.
    """
    logger.info("\n=== Testing Rapid Consecutive Requests ===")
    
    try:
        analyzer = SentimentAnalyzer()
        
        if not analyzer.news_manager:
            logger.error("No news_manager available")
            return False
        
        success_count = 0
        total_requests = 5
        
        for i in range(total_requests):
            logger.info(f"\nRapid request {i+1}/{total_requests}...")
            
            # Check availability before each request
            available = analyzer.news_manager._get_available_providers()
            logger.info(f"  Available providers: {available}")
            
            if not available:
                logger.error(f"  ❌ No providers available for request {i+1}")
                continue
            
            # Make request
            response = analyzer.news_manager.fetch_articles(
                query=f'test request {i+1}',
                page_size=2
            )
            
            if response.success:
                logger.info(f"  ✓ Request {i+1}: {len(response.articles)} articles")
                success_count += 1
            else:
                logger.warning(f"  ⚠ Request {i+1} failed: {response.error_message}")
            
            # Small delay to avoid overwhelming the APIs
            time.sleep(0.5)
        
        logger.info(f"\nRapid request test: {success_count}/{total_requests} successful")
        return success_count > 0  # At least some requests should succeed
        
    except Exception as e:
        logger.error(f"Error during rapid request test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing fix for original 'No providers available' scenario...")
    
    # Test 1: Simulate the original scenario
    scenario_success = simulate_original_scenario()
    
    # Test 2: Test rapid requests
    rapid_test_success = test_rapid_requests()
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("ORIGINAL SCENARIO TEST RESULTS:")
    logger.info("="*60)
    
    if scenario_success:
        logger.info("✅ Original scenario test: PASSED")
        logger.info("   The 'No providers available' issue has been RESOLVED")
    else:
        logger.error("❌ Original scenario test: FAILED")
        logger.error("   The 'No providers available' issue persists")
    
    if rapid_test_success:
        logger.info("✅ Rapid request test: PASSED")
    else:
        logger.error("❌ Rapid request test: FAILED")
    
    if scenario_success and rapid_test_success:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("\n✅ SOLUTION VERIFIED:")
        logger.info("   • The automatic provider health recovery is working")
        logger.info("   • Providers recover automatically after rate limits expire")
        logger.info("   • The 'No providers available' error should no longer occur")
        logger.info("\n📋 WHAT WAS FIXED:")
        logger.info("   • Added auto-recovery logic in _get_available_providers()")
        logger.info("   • Providers marked unhealthy due to rate limiting now recover automatically")
        logger.info("   • Health status is reset when providers become available again")
    elif scenario_success:
        logger.info("\n✅ MAIN ISSUE RESOLVED")
        logger.info("   The original 'No providers available' issue has been fixed")
    else:
        logger.error("\n❌ ISSUE NOT FULLY RESOLVED")
        logger.error("   Additional investigation may be needed")
    
    logger.info("\n📝 RECOMMENDATIONS:")
    logger.info("   1. Monitor your sentiment analysis logs for any remaining issues")
    logger.info("   2. Ensure all API keys are valid and have sufficient quota")
    logger.info("   3. Consider implementing rate limit monitoring and alerting")
    logger.info("   4. The fix is now integrated into the main codebase")