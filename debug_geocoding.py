#!/usr/bin/env python3
"""Debug geocoding API responses."""

import asyncio
import aiohttp
import json

async def test_geocoding():
    """Test geocoding for problematic cities."""
    
    cities = [
        "Leeds, West Yorkshire",
        "Coventry, West Midlands", 
        "Blackburn, Lancashire",
        "Leeds",
        "Coventry",
        "Blackburn"
    ]
    
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    
    async with aiohttp.ClientSession() as session:
        for city in cities:
            print(f"\nTesting geocoding for: {city}")
            
            params = {
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            try:
                async with session.get(geocoding_url, params=params) as response:
                    print(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"Response: {json.dumps(data, indent=2)}")
                        
                        if data.get("results"):
                            result = data["results"][0]
                            lat = result.get("latitude")
                            lon = result.get("longitude")
                            print(f"✅ Found coordinates: {lat}, {lon}")
                        else:
                            print("❌ No results found")
                    else:
                        text = await response.text()
                        print(f"Error response: {text}")
                        
            except Exception as e:
                print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_geocoding())