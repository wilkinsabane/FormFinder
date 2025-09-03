#!/usr/bin/env python3
"""
Test script to validate API keys for all news providers.
This will help identify which API keys are valid and which need to be updated.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from formfinder.news_providers import NewsAPIProvider, NewsDataIOProvider, TheNewsAPIProvider
from formfinder.news_manager import ProviderConfig
from formfinder.config import load_config

def test_provider(provider_class, config, provider_name):
    """
    Test a single provider with its API key.
    
    Args:
        provider_class: The provider class to test
        config: ProviderConfig for the provider
        provider_name: Name of the provider for logging
    
    Returns:
        tuple: (success, message)
    """
    try:
        provider = provider_class(config.api_key)
        
        # Test with a simple search query
        test_query = "football"
        print(f"\n=== Testing {provider_name} ===")
        print(f"API Key: {config.api_key[:10]}...{config.api_key[-4:]}")
        print(f"Testing with query: '{test_query}'")
        
        response = provider.fetch_articles(test_query, limit=1)
        
        if response.success:
            article_count = len(response.articles) if response.articles else 0
            print(f"✅ SUCCESS: {provider_name} returned {article_count} articles")
            if response.articles and len(response.articles) > 0:
                 print(f"Sample article: {response.articles[0].title[:50]}...")
            return True, f"{provider_name} is working correctly"
        else:
            print(f"❌ ERROR: {provider_name} failed - {response.error_message}")
            
            # Provide specific guidance based on error type
            error_msg = response.error_message or ""
            if "402" in error_msg or "Payment Required" in error_msg:
                guidance = f"{provider_name}: API key quota exceeded or payment required. Please update your API key or upgrade your plan."
            elif "401" in error_msg or "Unauthorized" in error_msg:
                guidance = f"{provider_name}: Invalid API key. Please check and update your API key."
            elif "403" in error_msg or "Forbidden" in error_msg:
                guidance = f"{provider_name}: Access forbidden. Check API key permissions or account status."
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                guidance = f"{provider_name}: Rate limit exceeded. Wait before retrying or upgrade your plan."
            else:
                guidance = f"{provider_name}: {error_msg}"
                
            return False, guidance
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: {provider_name} failed - {error_msg}")
        
        # Provide specific guidance based on error type
        if "402" in error_msg or "Payment Required" in error_msg:
            guidance = f"{provider_name}: API key quota exceeded or payment required. Please update your API key or upgrade your plan."
        elif "401" in error_msg or "Unauthorized" in error_msg:
            guidance = f"{provider_name}: Invalid API key. Please check and update your API key."
        elif "403" in error_msg or "Forbidden" in error_msg:
            guidance = f"{provider_name}: Access forbidden. Check API key permissions or account status."
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            guidance = f"{provider_name}: Rate limit exceeded. Wait before retrying or upgrade your plan."
        else:
            guidance = f"{provider_name}: Unexpected error - {error_msg}"
            
        return False, guidance

def main():
    """
    Test all configured news providers.
    """
    print("=== News Provider API Key Validation ===")
    print("Testing all configured providers...\n")
    
    try:
        # Load configuration
        config = load_config()
        sentiment_config = config.sentiment_analysis
        
        # Test each provider
        results = []
        
        # NewsAPI
        if 'newsapi' in sentiment_config.providers and sentiment_config.providers['newsapi'].enabled:
            newsapi_provider = sentiment_config.providers['newsapi']
            newsapi_config = ProviderConfig(
                name="newsapi",
                api_key=newsapi_provider.api_key,
                enabled=newsapi_provider.enabled,
                priority=newsapi_provider.priority,
                max_retries=newsapi_provider.max_retries
            )
            success, message = test_provider(NewsAPIProvider, newsapi_config, "NewsAPI")
            results.append(("NewsAPI", success, message))
        
        # NewsData.io
        if 'newsdata_io' in sentiment_config.providers and sentiment_config.providers['newsdata_io'].enabled:
            newsdata_provider = sentiment_config.providers['newsdata_io']
            newsdata_config = ProviderConfig(
                name="newsdata_io",
                api_key=newsdata_provider.api_key,
                enabled=newsdata_provider.enabled,
                priority=newsdata_provider.priority,
                max_retries=newsdata_provider.max_retries
            )
            success, message = test_provider(NewsDataIOProvider, newsdata_config, "NewsData.io")
            results.append(("NewsData.io", success, message))
        
        # TheNewsAPI
        if 'thenewsapi' in sentiment_config.providers and sentiment_config.providers['thenewsapi'].enabled:
            thenewsapi_provider = sentiment_config.providers['thenewsapi']
            thenewsapi_config = ProviderConfig(
                name="thenewsapi",
                api_key=thenewsapi_provider.api_key,
                enabled=thenewsapi_provider.enabled,
                priority=thenewsapi_provider.priority,
                max_retries=thenewsapi_provider.max_retries
            )
            success, message = test_provider(TheNewsAPIProvider, thenewsapi_config, "TheNewsAPI")
            results.append(("TheNewsAPI", success, message))
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        working_providers = []
        failed_providers = []
        
        for provider_name, success, message in results:
            status = "✅ WORKING" if success else "❌ FAILED"
            print(f"{status}: {message}")
            
            if success:
                working_providers.append(provider_name)
            else:
                failed_providers.append(provider_name)
        
        print(f"\nWorking providers: {len(working_providers)}/{len(results)}")
        if working_providers:
            print(f"Available: {', '.join(working_providers)}")
        
        if failed_providers:
            print(f"\n⚠️  ATTENTION REQUIRED:")
            print(f"Failed providers: {', '.join(failed_providers)}")
            print(f"\nTo fix the issues:")
            print(f"1. Visit the provider websites to get new API keys")
            print(f"2. Update the API keys in config.yaml")
            print(f"3. Ensure your accounts have sufficient quota/credits")
            
            print(f"\nProvider websites:")
            print(f"- NewsAPI: https://newsapi.org/")
            print(f"- NewsData.io: https://newsdata.io/")
            print(f"- TheNewsAPI: https://www.thenewsapi.com/")
        else:
            print(f"\n🎉 All providers are working correctly!")
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return 1
    
    return 0 if not failed_providers else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)