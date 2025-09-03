#!/usr/bin/env python3
"""Test script to check NewsAPI with different team names."""

import requests
import json

def test_team_search():
    """Test NewsAPI with different team names and search strategies."""
    api_key = 'ff008e7b4e9b4041ab44c50a729d7885'
    
    teams = [
        'Arsenal',
        'Chelsea',
        'Manchester United',
        'Liverpool',
        'Barcelona',
        'Real Madrid',
        'Ararat-Armenia',
        'Alashkert',
        'Dinamo Minsk',
        'Partizan'
    ]
    
    url = 'https://newsapi.org/v2/everything'
    
    print("Testing different team search strategies:")
    print("=" * 50)
    
    for team in teams:
        # Test with exact team name in quotes
        exact_query = f'"{team}" AND (football OR soccer)'
        params = {
            'q': exact_query,
            'from': '2025-08-04',
            'to': '2025-08-11',
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 10,
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                exact_count = data.get('totalResults', 0)
            else:
                exact_count = 0
                print(f"Error for {team}: {response.status_code}")
        except Exception as e:
            exact_count = 0
            print(f"Error for {team}: {e}")
        
        # Test with broader search
        broad_query = f'{team} AND (football OR soccer)'
        params['q'] = broad_query
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                broad_count = data.get('totalResults', 0)
            else:
                broad_count = 0
        except:
            broad_count = 0
        
        print(f"{team}: {exact_count} (exact) | {broad_count} (broad) articles")
    
    # Test date range issue
    print("\nTesting date range issue:")
    print("=" * 30)
    
    # Test with very recent date
    params = {
        'q': 'Arsenal AND (football OR soccer)',
        'from': '2025-08-10',
        'to': '2025-08-11',
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 10,
        'apiKey': api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        print(f"Recent Arsenal articles (Aug 10-11): {data.get('totalResults', 0)}")
    
    # Test with much older date
    params.update({
        'from': '2024-08-01',
        'to': '2024-08-31'
    })
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        print(f"Older Arsenal articles (Aug 2024): {data.get('totalResults', 0)}")

if __name__ == "__main__":
    test_team_search()