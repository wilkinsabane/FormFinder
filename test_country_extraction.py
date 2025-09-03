import json

# Load league data from leagues.json
with open('leagues.json', 'r') as f:
    data = json.load(f)

# Extract and print country information for specific leagues
league_ids = [203, 204]  # Superliga and Ligue 1
leagues_data = {}

for league in data['results']:
    if league['id'] in league_ids:
        country = league['country']['name'].title() if 'country' in league else 'Unknown'
        leagues_data[league['id']] = {
            'name': league['name'],
            'country': country
        }
        print(f"League ID: {league['id']}, Name: {league['name']}, Country: {country}")