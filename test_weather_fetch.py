#!/usr/bin/env python3
"""
Test script to manually fetch weather data for a recent fixture
to diagnose why weather data is showing as NULL in the database.
"""

import asyncio
import sys
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import get_db_session, Fixture, WeatherData
from formfinder.weather_fetcher import WeatherFetcher
from datetime import datetime, timedelta

async def test_weather_fetch():
    """Test weather fetching for a recent completed fixture."""
    
    # Load config and get database session
    config = load_config()
    
    with get_db_session() as session:
        try:
            # Get a recent fixture with stadium city data
            recent_fixture = session.query(Fixture).filter(
                Fixture.match_date.isnot(None),
                Fixture.stadium_city.isnot(None)
            ).order_by(Fixture.match_date.desc()).first()
            
            if not recent_fixture:
                print("No recent completed fixtures found")
                return
            
            print(f"Testing weather fetch for fixture {recent_fixture.id}")
            print(f"Match: {recent_fixture.home_team} vs {recent_fixture.away_team}")
            print(f"Date: {recent_fixture.match_date}")
            print(f"Stadium City: {recent_fixture.stadium_city}")
            print("\n" + "="*50 + "\n")
            
            # Initialize weather fetcher
            weather_fetcher = WeatherFetcher(session)
            
            # Test geocoding first
            print(f"Testing geocoding for {recent_fixture.stadium_city}...")
            lat, lon = await weather_fetcher._get_coordinates(recent_fixture.stadium_city)
            
            if lat and lon:
                print(f"✓ Geocoding successful: {lat}, {lon}")
            else:
                print(f"✗ Geocoding failed for {recent_fixture.stadium_city}")
                return
            
            print("\nTesting weather data fetch...")
            
            # Test weather fetch
            success = await weather_fetcher.fetch_weather_for_fixture(recent_fixture)
            
            if success:
                print("✓ Weather fetch completed successfully")
                
                # Check what was actually saved
                weather_record = session.query(WeatherData).filter_by(
                    fixture_id=recent_fixture.id
                ).first()
                
                if weather_record:
                    print("\nWeather data saved:")
                    print(f"  Temperature: {weather_record.temperature_2m}°C")
                    print(f"  Humidity: {weather_record.relative_humidity_2m}%")
                    print(f"  Wind Speed: {weather_record.wind_speed_10m} km/h")
                    print(f"  Weather Code: {weather_record.weather_code}")
                    print(f"  Data Type: {weather_record.data_type}")
                    print(f"  API Response Time: {weather_record.api_response_time_ms}ms")
                else:
                    print("✗ No weather record found in database")
            else:
                print("✗ Weather fetch failed")
                
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_weather_fetch())