#!/usr/bin/env python3
"""Test script to verify NewsAPI functionality."""

import requests
import json

def test_newsapi():
    """Test NewsAPI with a simple query."""
    api_key = 'ff008e7b4e9b4041ab44c50a729d7885'
    query = 'Arsenal AND (football OR soccer)'
    
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': '2025-08-04',
        'to': '2025-08-11',
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 10,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total Results: {data.get('totalResults', 'N/A')}")
            print(f"Articles Found: {len(data.get('articles', []))}")
            
            if data.get('articles'):
                print("\nFirst article sample:")
                article = data['articles'][0]
                print(f"Title: {article.get('title', 'No title')}")
                print(f"Description: {article.get('description', 'No description')}")
            else:
                print("No articles found")
                
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_newsapi()