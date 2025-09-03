import requests
import json

# Test the API directly
url = "https://api.soccerdataapi.com/matches/"
params = {
    'league_id': 228,
    'auth_token': '320cae54d49a09f11c5cd23da43204a5543fb394',
    'season': '2025-2026'
}

print("Testing API endpoint...")
print(f"URL: {url}")
print(f"Params: {params}")

response = requests.get(url, params=params)
print(f"Status code: {response.status_code}")
print(f"Response length: {len(response.text)}")

if response.status_code == 200:
    data = response.json()
    print("\nResponse structure:")
    print(json.dumps(data[:2] if isinstance(data, list) else data, indent=2))
    
    if isinstance(data, list) and len(data) > 0:
        league_data = data[0]
        print(f"\nLeague ID: {league_data.get('league_id')}")
        print(f"Has 'stage': {'stage' in league_data}")
        
        if 'stage' in league_data:
            for stage in league_data['stage']:
                print(f"\nStage: {stage.get('stage_name', 'N/A')}")
                print(f"Matches count: {len(stage.get('matches', []))}")
                
                # Show first few matches
                matches = stage.get('matches', [])[:3]
                for match in matches:
                    print(f"  Match: {match.get('home_team', {}).get('name')} vs {match.get('away_team', {}).get('name')} - Status: {match.get('status')}")
else:
    print(f"Error: {response.text}")