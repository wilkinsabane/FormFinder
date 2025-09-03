#!/usr/bin/env python3
"""
Demonstration script showing the performance difference between:
1. Multi-provider mode (with enabled providers)
2. Legacy fallback mode (all providers disabled)

This script measures initialization time and processing overhead.
"""

import time
import logging
from formfinder.sentiment import SentimentAnalyzer
from formfinder.config import get_config, load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
load_config()

def measure_initialization_time(mode_name: str, enable_providers: bool = True):
    """
    Measure the time it takes to initialize SentimentAnalyzer.
    
    Args:
        mode_name: Description of the mode being tested
        enable_providers: Whether to enable providers in config
    """
    logger.info(f"\n=== Testing {mode_name} ===")
    
    # Temporarily modify config if needed
    original_config = None
    if not enable_providers:
        config = get_config()
        original_config = {
            'newsapi': config.sentiment_analysis.providers['newsapi'].enabled,
            'newsdata_io': config.sentiment_analysis.providers['newsdata_io'].enabled,
            'thenewsapi': config.sentiment_analysis.providers['thenewsapi'].enabled
        }
        # Disable all providers
        config.sentiment_analysis.providers['newsapi'].enabled = False
        config.sentiment_analysis.providers['newsdata_io'].enabled = False
        config.sentiment_analysis.providers['thenewsapi'].enabled = False
    
    try:
        # Measure initialization time
        start_time = time.time()
        analyzer = SentimentAnalyzer()
        init_time = time.time() - start_time
        
        logger.info(f"Initialization time: {init_time:.4f} seconds")
        logger.info(f"Using manager: {analyzer.use_manager}")
        logger.info(f"News manager: {'Active' if analyzer.news_manager else 'None (Legacy mode)'}")
        logger.info(f"Monitor enabled: {analyzer.monitor_enabled}")
        
        if analyzer.news_manager:
            available_providers = analyzer.news_manager._get_available_providers()
            logger.info(f"Available providers: {len(available_providers)}")
        else:
            logger.info("Legacy mode: Direct NewsAPI access")
            
        return init_time, analyzer
        
    finally:
        # Restore original config if modified
        if original_config:
            config = get_config()
            config.sentiment_analysis.providers['newsapi'].enabled = original_config['newsapi']
            config.sentiment_analysis.providers['newsdata_io'].enabled = original_config['newsdata_io']
            config.sentiment_analysis.providers['thenewsapi'].enabled = original_config['thenewsapi']

def main():
    """
    Main demonstration function.
    """
    logger.info("🚀 Performance Comparison: Multi-provider vs Legacy Fallback")
    logger.info("=" * 70)
    
    # Test 1: Multi-provider mode (with enabled providers)
    multi_time, multi_analyzer = measure_initialization_time(
        "Multi-provider Mode (Normal Operation)", 
        enable_providers=True
    )
    
    # Test 2: Legacy fallback mode (all providers disabled)
    legacy_time, legacy_analyzer = measure_initialization_time(
        "Legacy Fallback Mode (All Providers Disabled)", 
        enable_providers=False
    )
    
    # Performance comparison
    logger.info(f"\n📊 PERFORMANCE COMPARISON")
    logger.info("=" * 50)
    logger.info(f"Multi-provider initialization: {multi_time:.4f}s")
    logger.info(f"Legacy fallback initialization: {legacy_time:.4f}s")
    
    if legacy_time < multi_time:
        speedup = ((multi_time - legacy_time) / multi_time) * 100
        logger.info(f"✅ Legacy mode is {speedup:.1f}% faster for initialization")
    else:
        slowdown = ((legacy_time - multi_time) / multi_time) * 100
        logger.info(f"⚠️ Legacy mode is {slowdown:.1f}% slower for initialization")
    
    logger.info(f"\n🎯 KEY BENEFITS OF FALLBACK MODE:")
    logger.info("   • No provider selection overhead")
    logger.info("   • No health monitoring system")
    logger.info("   • No failover logic complexity")
    logger.info("   • Direct API access (single point of failure but faster)")
    logger.info("   • Simplified error handling")
    logger.info("   • Reduced memory footprint")
    
    logger.info(f"\n⚡ PROCESSING SPEED IMPROVEMENTS:")
    logger.info("   • Bypasses multi-provider selection algorithm")
    logger.info("   • No provider health checks during requests")
    logger.info("   • No monitoring data collection overhead")
    logger.info("   • Direct HTTP requests without abstraction layers")

if __name__ == "__main__":
    main()