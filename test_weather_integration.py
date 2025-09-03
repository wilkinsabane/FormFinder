#!/usr/bin/env python3
"""
Test Weather Integration

This script tests the weather integration components:
1. WeatherFetcher class functionality
2. Database schema for weather data
3. API connectivity to Open-Meteo

Author: FormFinder2 Team
Created: 2025-01-22
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from formfinder.config import load_config, get_config
from formfinder.database import get_db_session, Fixture, WeatherData, WeatherForecast
from formfinder.weather_fetcher import WeatherFetcher
from formfinder.feature_precomputer import FeaturePrecomputer
from sqlalchemy import select

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_schema():
    """Test that the database schema supports weather data."""
    logger.info("🗄️ Testing database schema for weather data...")
    
    try:
        with get_db_session() as session:
            # Test WeatherData model exists
            weather_count = session.execute(select(WeatherData)).scalars().all()
            logger.info(f"📊 WeatherData table accessible, found {len(weather_count)} records")
            
            # Test WeatherForecast model exists
            forecast_count = session.execute(select(WeatherForecast)).scalars().all()
            logger.info(f"📊 WeatherForecast table accessible, found {len(forecast_count)} records")
            
            # Test Fixture table exists
            fixture_count = session.execute(select(Fixture)).scalars().all()
            logger.info(f"📊 Fixture table accessible, found {len(fixture_count)} records")
            
            logger.info("✅ Database schema test passed")
            return True
                
    except Exception as e:
        logger.error(f"❌ Database schema test failed: {e}")
        return False


async def test_weather_fetcher_api():
    """Test WeatherFetcher API connectivity."""
    logger.info("🌐 Testing WeatherFetcher API connectivity...")
    
    try:
        with get_db_session() as session:
            weather_fetcher = WeatherFetcher(session)
            
            # Test geocoding
            logger.info("Testing geocoding for London...")
            coords = await weather_fetcher._get_coordinates("London")
            
            if coords:
                lat, lon = coords
                logger.info(f"✅ Geocoding successful: London -> ({lat}, {lon})")
                
                # Test weather API call
                logger.info("Testing weather API call...")
                test_date = datetime.now() - timedelta(days=1)  # Yesterday
                
                # Get a real fixture from the database for testing
                fixture_query = select(Fixture).where(
                    Fixture.stadium_city.isnot(None)
                ).limit(1)
                fixture = session.execute(fixture_query).scalar_one_or_none()
                
                if not fixture:
                    logger.error("❌ No fixtures with stadium city found for testing")
                    return False
                
                success = await weather_fetcher._fetch_historical_weather(
                    fixture, lat, lon, test_date
                )
                
                if success:
                    logger.info("✅ Weather API call successful")
                    return True
                else:
                    logger.error("❌ Weather API call failed")
                    return False
            else:
                logger.error("❌ Geocoding failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ Weather fetcher API test failed: {e}")
        return False


def test_create_sample_fixture():
    """Create a sample fixture for testing weather integration."""
    logger.info("🏟️ Creating sample fixture for testing...")
    
    try:
        with get_db_session() as session:
            # Check if we already have any fixture with stadium city
            existing_fixture = session.execute(
                select(Fixture).where(Fixture.stadium_city.isnot(None)).limit(1)
            ).scalar_one_or_none()
            
            if existing_fixture:
                logger.info(f"✅ Found existing fixture with stadium info (ID: {existing_fixture.id})")
                logger.info(f"Stadium: {existing_fixture.stadium_name}, City: {existing_fixture.stadium_city}")
                return existing_fixture.id
            
            logger.info("No existing fixtures with stadium info found")
            logger.info("Weather integration requires fixtures with stadium_city information")
            logger.info("Please run your normal data fetching process first to populate fixtures")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to check fixtures: {e}")
        return None


async def test_weather_integration_full():
    """Test full weather integration with a sample fixture."""
    logger.info("🔄 Testing full weather integration...")
    
    try:
        # Create sample fixture
        fixture_id = test_create_sample_fixture()
        if not fixture_id:
            return False
        
        with get_db_session() as session:
            # Get the fixture
            fixture = session.get(Fixture, fixture_id)
            if not fixture:
                logger.error("❌ Could not retrieve test fixture")
                return False
            
            # Test weather fetching
            weather_fetcher = WeatherFetcher(session)
            success = await weather_fetcher.fetch_weather_for_fixture(fixture)
            
            if success:
                logger.info("✅ Weather data fetched successfully")
                
                # Check if weather data was saved
                weather_query = select(WeatherData).where(WeatherData.fixture_id == fixture.id)
                weather_data = session.execute(weather_query).scalars().all()
                
                logger.info(f"📊 Found {len(weather_data)} weather records")
                
                if weather_data:
                    sample_weather = weather_data[0]
                    logger.info(f"Sample weather data:")
                    logger.info(f"  Temperature: {sample_weather.temperature_2m}°C")
                    logger.info(f"  Humidity: {sample_weather.relative_humidity_2m}%")
                    logger.info(f"  Wind Speed: {sample_weather.wind_speed_10m} km/h")
                    logger.info(f"  Precipitation: {sample_weather.precipitation} mm")
                    logger.info(f"  Weather Code: {sample_weather.weather_code}")
                    logger.info(f"  Cloud Cover: {sample_weather.cloud_cover}%")
                    
                    # Test feature computation
                    logger.info("Testing feature computation...")
                    precomputer = FeaturePrecomputer(session)
                    result = precomputer.compute_preview_features(fixture.id)
                    
                    if result.get('success'):
                        logger.info("✅ Feature computation successful")
                        preview_features = result.get('preview_features', {})
                        logger.info(f"Weather features:")
                        logger.info(f"  Temperature: {preview_features.get('weather_temp_c')}°C")
                        logger.info(f"  Humidity: {preview_features.get('weather_humidity')}%")
                        logger.info(f"  Wind Speed: {preview_features.get('weather_wind_speed')} km/h")
                        return True
                    else:
                        logger.error(f"❌ Feature computation failed: {result.get('error')}")
                        return False
                else:
                    logger.error("❌ No weather data found after fetching")
                    return False
            else:
                logger.error("❌ Weather data fetching failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ Full integration test failed: {e}")
        return False


async def main():
    """Run all weather integration tests."""
    logger.info("🚀 Starting Weather Integration Tests")
    logger.info("=" * 50)
    
    # Load configuration
    try:
        load_config()
        config = get_config()
        logger.info("✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return False
    
    tests = [
        ("Database Schema", test_database_schema),
        ("Weather API Connectivity", test_weather_fetcher_api),
        ("Full Weather Integration", test_weather_integration_full),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} Test: PASSED")
            else:
                logger.error(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} Test: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 Test Results Summary:")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All weather integration tests passed!")
        logger.info("Weather data integration is working correctly.")
        logger.info("\n📝 Next steps:")
        logger.info("  1. Run your normal data fetching process")
        logger.info("  2. Weather data will be automatically collected for new fixtures")
        logger.info("  3. Prediction models will now include weather features")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())