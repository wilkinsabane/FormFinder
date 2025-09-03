#!/usr/bin/env python3
"""
Test API connection and authentication.
"""

import requests
import yaml
from pathlib import Path

def test_api_connection():
    """Test basic API connection and authentication."""
    
    # Load config
    config_path = Path("config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    api_config = config['api']
    base_url = api_config['base_url']
    auth_token = api_config['auth_token']
    
    print(f"Testing API connection to: {base_url}")
    print(f"Using auth token: {auth_token[:10]}...{auth_token[-10:]}")
    
    # Test basic endpoint
    test_url = f"{base_url}/leagues"
    headers = {
        'Authorization': f'Bearer {auth_token}'
    }
    
    try:
        response = requests.get(test_url, headers=headers, timeout=10)
        print(f"\nResponse status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nSuccess! Received {len(data.get('data', []))} leagues")
            if data.get('data'):
                print("First few leagues:")
                for league in data['data'][:3]:
                    print(f"  - {league.get('name', 'Unknown')} (ID: {league.get('id', 'Unknown')})")
        else:
            print(f"\nError response: {response.text}")
            
    except Exception as e:
        print(f"\nError making request: {e}")
    
    # Test historical matches endpoint for a specific league
    print("\n" + "="*50)
    print("Testing historical matches endpoint...")
    
    test_url = f"{base_url}/historical/"
    params = {
        'league_id': 204,  # Premier League
        'season': '2023-2024'
    }
    
    try:
        response = requests.get(test_url, headers=headers, params=params, timeout=10)
        print(f"\nHistorical matches response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Received {len(data.get('data', []))} matches")
            if data.get('data'):
                print("First match:")
                match = data['data'][0]
                print(f"  - {match.get('home_team', {}).get('name', 'Unknown')} vs {match.get('away_team', {}).get('name', 'Unknown')}")
                print(f"  - Date: {match.get('date', 'Unknown')}")
                print(f"  - ID: {match.get('id', 'Unknown')}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    test_api_connection()