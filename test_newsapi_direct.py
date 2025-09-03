#!/usr/bin/env python3
"""Direct test of NewsAPI to check if the API key is working properly."""

import requests
from datetime import datetime, timedelta

def test_newsapi_direct():
    """Test NewsAPI directly to check for issues."""
    api_key = "ff008e7b4e9b4041ab44c50a729d7885"
    
    print("Testing NewsAPI Direct Access")
    print("=" * 40)
    
    # Test 1: General everything endpoint
    print("\n1. Testing general everything endpoint...")
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'Arsenal football',
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'to': datetime.now().strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 5,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total Results: {data.get('totalResults', 0)}")
            print(f"Articles Returned: {len(data.get('articles', []))}")
            
            if data.get('articles'):
                print("\nFirst article:")
                article = data['articles'][0]
                print(f"  Title: {article.get('title', 'N/A')[:100]}...")
                print(f"  Source: {article.get('source', {}).get('name', 'N/A')}")
                print(f"  Published: {article.get('publishedAt', 'N/A')}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 2: Top headlines endpoint
    print("\n2. Testing top headlines endpoint...")
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'q': 'football',
        'country': 'gb',
        'category': 'sports',
        'pageSize': 5,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total Results: {data.get('totalResults', 0)}")
            print(f"Articles Returned: {len(data.get('articles', []))}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 3: Check API key status
    print("\n3. Testing API key status...")
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'country': 'us',
        'pageSize': 1,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API key is valid and working")
        elif response.status_code == 401:
            print("❌ API key is invalid or expired")
        elif response.status_code == 426:
            print("⚠️ API requires upgrade (rate limit or plan issue)")
        elif response.status_code == 429:
            print("⚠️ Rate limit exceeded")
        else:
            print(f"❓ Unexpected status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_newsapi_direct()