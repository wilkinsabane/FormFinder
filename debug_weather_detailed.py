#!/usr/bin/env python3
"""Detailed debug script to investigate weather data extraction and saving."""

import asyncio
import logging
from datetime import datetime
from formfinder.config import load_config
from formfinder.database import get_db_session, Fixture, WeatherData
from formfinder.weather_fetcher import WeatherFetcher

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_weather_extraction():
    """Debug weather data extraction in detail."""
    
    # Load config and get database session
    config = load_config()
    
    with get_db_session() as session:
        # Get a recent fixture with stadium city
        fixture = session.query(Fixture).filter(
            Fixture.stadium_city.isnot(None),
            Fixture.match_date.isnot(None)
        ).order_by(Fixture.match_date.desc()).first()
        
        if not fixture:
            print("No fixtures found with stadium city data")
            return
        
        print(f"\nTesting with fixture:")
        print(f"  ID: {fixture.id}")
        print(f"  Stadium City: {fixture.stadium_city}")
        print(f"  Match Date: {fixture.match_date}")
        print(f"  Status: {fixture.status}")
        
        # Create WeatherFetcher instance
        weather_fetcher = WeatherFetcher(session)
        
        # Get coordinates
        lat, lon = await weather_fetcher._get_coordinates(fixture.stadium_city)
        print(f"\nCoordinates: {lat}, {lon}")
        
        if not lat or not lon:
            print("Failed to get coordinates")
            return
        
        # Determine data type
        now = datetime.utcnow()
        match_time = fixture.match_date
        data_type = 'historical' if match_time < now else 'forecast'
        
        print(f"\nData type: {data_type}")
        print(f"Match time: {match_time}")
        print(f"Current time: {now}")
        
        # Test the actual API call and data extraction
        if data_type == 'historical':
            await debug_historical_extraction(weather_fetcher, fixture, lat, lon, match_time)
        else:
            await debug_forecast_extraction(weather_fetcher, fixture, lat, lon, match_time)

async def debug_historical_extraction(weather_fetcher, fixture, lat, lon, match_time):
    """Debug historical weather data extraction."""
    import aiohttp
    
    print("\n=== DEBUGGING HISTORICAL WEATHER EXTRACTION ===")
    
    # Format date for API
    date_str = match_time.strftime("%Y-%m-%d")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(weather_fetcher.weather_vars),
        "timezone": "UTC"
    }
    
    print(f"\nAPI Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(weather_fetcher.historical_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                print(f"\nAPI Response Status: {response.status}")
                print(f"Response keys: {list(data.keys())}")
                
                hourly_data = data.get("hourly", {})
                print(f"\nHourly data keys: {list(hourly_data.keys())}")
                
                times = hourly_data.get("time", [])
                print(f"\nNumber of time entries: {len(times)}")
                if times:
                    print(f"First time: {times[0]}")
                    print(f"Last time: {times[-1]}")
                
                # Find closest time index
                match_hour = match_time.replace(minute=0, second=0, microsecond=0)
                print(f"\nLooking for match hour: {match_hour}")
                
                closest_index = weather_fetcher._find_closest_time_index(times, match_hour)
                print(f"Closest index found: {closest_index}")
                
                if closest_index is not None:
                    print(f"\nWeather data at index {closest_index}:")
                    for var in weather_fetcher.weather_vars:
                        if var in hourly_data and len(hourly_data[var]) > closest_index:
                            value = hourly_data[var][closest_index]
                            print(f"  {var}: {value}")
                        else:
                            print(f"  {var}: NOT FOUND or INDEX OUT OF RANGE")
                    
                    # Test creating WeatherData object
                    print(f"\n=== TESTING WEATHERDATA OBJECT CREATION ===")
                    weather_data = WeatherData(
                        fixture_id=fixture.id,
                        latitude=lat,
                        longitude=lon,
                        weather_datetime=match_hour,
                        data_type='historical',
                        api_response_time_ms=100.0
                    )
                    
                    print(f"Created WeatherData object: {weather_data}")
                    
                    # Test setting attributes
                    print(f"\nTesting setattr for each weather variable:")
                    for var in weather_fetcher.weather_vars:
                        if var in hourly_data and len(hourly_data[var]) > closest_index:
                            value = hourly_data[var][closest_index]
                            try:
                                setattr(weather_data, var, value)
                                retrieved_value = getattr(weather_data, var)
                                print(f"  {var}: {value} -> {retrieved_value} (SUCCESS)")
                            except Exception as e:
                                print(f"  {var}: {value} -> ERROR: {e}")
                        else:
                            print(f"  {var}: SKIPPED (no data)")
                    
                    # Check final object state
                    print(f"\nFinal WeatherData object state:")
                    for var in weather_fetcher.weather_vars:
                        value = getattr(weather_data, var, 'NOT_SET')
                        print(f"  {var}: {value}")
                
                else:
                    print("Could not find matching time index")
            else:
                print(f"API request failed with status: {response.status}")

async def debug_forecast_extraction(weather_fetcher, fixture, lat, lon, match_time):
    """Debug forecast weather data extraction."""
    print("\n=== DEBUGGING FORECAST WEATHER EXTRACTION ===")
    print("Forecast debugging not implemented yet")

if __name__ == "__main__":
    asyncio.run(debug_weather_extraction())