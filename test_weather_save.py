#!/usr/bin/env python3
"""Test script to verify weather data is actually saved to database."""

import asyncio
import logging
from datetime import datetime
from formfinder.config import load_config
from formfinder.database import get_db_session, Fixture, WeatherData
from formfinder.weather_fetcher import WeatherFetcher

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_weather_save():
    """Test that weather data is actually saved to database."""
    
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
        
        # Check if weather data already exists
        existing_weather = session.query(WeatherData).filter(
            WeatherData.fixture_id == fixture.id
        ).first()
        
        if existing_weather:
            print(f"\nExisting weather data found:")
            print(f"  Temperature: {existing_weather.temperature_2m}")
            print(f"  Humidity: {existing_weather.relative_humidity_2m}")
            print(f"  Wind Speed: {existing_weather.wind_speed_10m}")
            print(f"  Weather Code: {existing_weather.weather_code}")
            
            # Delete existing data to test fresh save
            print("\nDeleting existing weather data to test fresh save...")
            session.delete(existing_weather)
            session.commit()
        
        # Create WeatherFetcher and fetch weather
        weather_fetcher = WeatherFetcher(session)
        
        print(f"\nFetching weather data...")
        success = await weather_fetcher.fetch_weather_for_fixture(fixture)
        
        print(f"Weather fetch result: {success}")
        
        # Check what was actually saved
        saved_weather = session.query(WeatherData).filter(
            WeatherData.fixture_id == fixture.id
        ).first()
        
        if saved_weather:
            print(f"\nSaved weather data:")
            print(f"  Temperature: {saved_weather.temperature_2m}")
            print(f"  Humidity: {saved_weather.relative_humidity_2m}")
            print(f"  Wind Speed: {saved_weather.wind_speed_10m}")
            print(f"  Weather Code: {saved_weather.weather_code}")
            print(f"  Precipitation: {saved_weather.precipitation}")
            print(f"  Cloud Cover: {saved_weather.cloud_cover}")
            print(f"  Data Type: {saved_weather.data_type}")
            print(f"  Weather DateTime: {saved_weather.weather_datetime}")
        else:
            print("\nNo weather data was saved to database!")
            
            # Let's check if there were any errors in the logs
            print("\nChecking for any WeatherData records for this fixture...")
            all_weather = session.query(WeatherData).filter(
                WeatherData.fixture_id == fixture.id
            ).all()
            print(f"Total WeatherData records for fixture {fixture.id}: {len(all_weather)}")
            
            for wd in all_weather:
                print(f"  Record: temp={wd.temperature_2m}, humidity={wd.relative_humidity_2m}, datetime={wd.weather_datetime}")

if __name__ == "__main__":
    asyncio.run(test_weather_save())