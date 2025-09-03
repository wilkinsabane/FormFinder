"""Debug script to examine weather API responses."""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from formfinder.config import load_config
from formfinder.database import get_db_session, Fixture

async def debug_weather_api():
    """Debug weather API responses to understand data structure."""
    
    try:
        # Load config
        config = load_config()
        
        with get_db_session() as session:
            # Get a recent fixture
            recent_fixture = session.query(Fixture).filter(
                Fixture.stadium_city.isnot(None),
                Fixture.match_date.isnot(None)
            ).order_by(Fixture.match_date.desc()).first()
            
            if not recent_fixture:
                print("No fixtures found with stadium city data")
                return
            
            print(f"Testing with fixture: {recent_fixture.home_team} vs {recent_fixture.away_team}")
            print(f"Stadium city: {recent_fixture.stadium_city}")
            print(f"Match date: {recent_fixture.match_date}")
            
            # First test geocoding
            geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
            geocoding_params = {
                "name": recent_fixture.stadium_city,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            async with aiohttp.ClientSession() as session_http:
                print("\n=== Testing Geocoding ===")
                async with session_http.get(geocoding_url, params=geocoding_params) as response:
                    if response.status == 200:
                        geo_data = await response.json()
                        print(f"Geocoding response: {json.dumps(geo_data, indent=2)}")
                        
                        if geo_data.get("results"):
                            location = geo_data["results"][0]
                            lat = location.get("latitude")
                            lon = location.get("longitude")
                            print(f"Coordinates: {lat}, {lon}")
                            
                            # Now test weather API
                            print("\n=== Testing Weather API ===")
                            
                            # Determine if historical or forecast
                            now = datetime.utcnow()
                            match_time = recent_fixture.match_date
                            
                            if match_time < now:
                                # Historical weather
                                weather_url = "https://archive-api.open-meteo.com/v1/archive"
                                start_date = (match_time - timedelta(days=1)).strftime('%Y-%m-%d')
                                end_date = (match_time + timedelta(days=1)).strftime('%Y-%m-%d')
                                
                                weather_params = {
                                    "latitude": lat,
                                    "longitude": lon,
                                    "start_date": start_date,
                                    "end_date": end_date,
                                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                                    "timezone": "UTC"
                                }
                                
                                print(f"Historical weather request: {weather_url}")
                                print(f"Params: {json.dumps(weather_params, indent=2)}")
                                
                            else:
                                # Forecast weather
                                weather_url = "https://api.open-meteo.com/v1/forecast"
                                
                                weather_params = {
                                    "latitude": lat,
                                    "longitude": lon,
                                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                                    "timezone": "UTC",
                                    "forecast_days": 7
                                }
                                
                                print(f"Forecast weather request: {weather_url}")
                                print(f"Params: {json.dumps(weather_params, indent=2)}")
                            
                            async with session_http.get(weather_url, params=weather_params) as weather_response:
                                if weather_response.status == 200:
                                    weather_data = await weather_response.json()
                                    print(f"\nWeather API response status: {weather_response.status}")
                                    print(f"Response keys: {list(weather_data.keys())}")
                                    
                                    if "hourly" in weather_data:
                                        hourly = weather_data["hourly"]
                                        print(f"\nHourly data keys: {list(hourly.keys())}")
                                        
                                        # Show first few entries
                                        for key in ["time", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "weather_code"]:
                                            if key in hourly:
                                                values = hourly[key][:5]  # First 5 values
                                                print(f"{key}: {values}")
                                        
                                        # Find match time data
                                        times = hourly.get("time", [])
                                        match_hour = match_time.replace(minute=0, second=0, microsecond=0)
                                        target_str = match_hour.isoformat()
                                        
                                        print(f"\nLooking for match time: {target_str}")
                                        
                                        closest_index = None
                                        for i, time_str in enumerate(times):
                                            if time_str.startswith(target_str[:13]):  # Match to hour
                                                closest_index = i
                                                break
                                        
                                        if closest_index is not None:
                                            print(f"Found match time at index {closest_index}: {times[closest_index]}")
                                            
                                            # Extract values at that index
                                            for key in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "weather_code"]:
                                                if key in hourly and len(hourly[key]) > closest_index:
                                                    value = hourly[key][closest_index]
                                                    print(f"{key} at match time: {value}")
                                                else:
                                                    print(f"{key}: NOT FOUND or INDEX OUT OF RANGE")
                                        else:
                                            print("Could not find match time in weather data")
                                            print(f"Available times: {times[:10]}...")  # Show first 10 times
                                    else:
                                        print("No hourly data in response")
                                        print(f"Full response: {json.dumps(weather_data, indent=2)}")
                                else:
                                    print(f"Weather API error: {weather_response.status}")
                                    error_text = await weather_response.text()
                                    print(f"Error response: {error_text}")
                        else:
                            print("No geocoding results found")
                    else:
                        print(f"Geocoding error: {response.status}")
                        error_text = await response.text()
                        print(f"Error response: {error_text}")
                        
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_weather_api())