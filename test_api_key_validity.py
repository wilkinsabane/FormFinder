#!/usr/bin/env python3
"""Test script to check if the NewsAPI key is actually valid."""

import requests
import json
from datetime import datetime, timedelta

def test_api_key_validity():
    """Test if the NewsAPI key is actually valid by making a direct API call."""
    api_key = 'ff008e7b4e9b4041ab44c50a729d7885'
    
    print(f"🔑 Testing NewsAPI key: {api_key}")
    print("=" * 50)
    
    # Test with a simple query
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'Arsenal football',
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'to': datetime.now().strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 5,
        'apiKey': api_key
    }
    
    try:
        print(f"📡 Making API request to: {url}")
        print(f"📋 Query parameters: {json.dumps(params, indent=2)}")
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ API Key is VALID!")
            print(f"📰 Total Results: {data.get('totalResults', 'N/A')}")
            print(f"📰 Articles Returned: {len(data.get('articles', []))}")
            
            if data.get('articles'):
                print(f"\n📄 Sample Article:")
                article = data['articles'][0]
                print(f"   Title: {article.get('title', 'No title')[:100]}...")
                print(f"   Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"   Published: {article.get('publishedAt', 'Unknown')}")
            else:
                print("\n⚠️ No articles found for this query")
                
        elif response.status_code == 401:
            print(f"\n❌ API Key is INVALID - Unauthorized (401)")
            print(f"Response: {response.text}")
            
        elif response.status_code == 429:
            print(f"\n⚠️ Rate limit exceeded (429)")
            print(f"Response: {response.text}")
            
        else:
            print(f"\n❌ API Error ({response.status_code})")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("\n⏰ Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_api_key_validity()