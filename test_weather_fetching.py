#!/usr/bin/env python3
"""Test weather fetching functionality."""

import asyncio
import logging
from datetime import datetime

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.weather_fetcher import WeatherFetcher
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_weather_fetching():
    """Test weather fetching for fixtures without weather data."""
    
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Initialize weather fetcher
        weather_fetcher = WeatherFetcher(session)
        
        # Get fixtures without weather data
        query = text("""
            SELECT f.id, f.stadium_city, f.match_date, f.home_team_id, f.away_team_id
            FROM fixtures f
            LEFT JOIN weather_data wd ON f.id = wd.fixture_id
            WHERE f.stadium_city IS NOT NULL 
                AND f.match_date IS NOT NULL
                AND f.status = 'finished'
                AND wd.fixture_id IS NULL
            ORDER BY f.match_date DESC
            LIMIT 5
        """)
        
        fixtures = session.execute(query).fetchall()
        
        if not fixtures:
            logger.info("No fixtures found without weather data")
            return
        
        logger.info(f"Found {len(fixtures)} fixtures without weather data")
        
        for fixture_row in fixtures:
            fixture_id, stadium_city, match_date, home_team_id, away_team_id = fixture_row
            
            logger.info(f"\nTesting weather fetch for fixture {fixture_id}:")
            logger.info(f"  Stadium: {stadium_city}")
            logger.info(f"  Match Date: {match_date}")
            
            # Create a minimal fixture object
            class MinimalFixture:
                def __init__(self, fixture_id, stadium_city, match_date):
                    self.id = fixture_id
                    self.stadium_city = stadium_city
                    self.match_date = match_date
            
            minimal_fixture = MinimalFixture(fixture_id, stadium_city, match_date)
            
            try:
                success = await weather_fetcher.fetch_weather_for_fixture(minimal_fixture)
                
                if success:
                    logger.info(f"  ✅ Successfully fetched weather data")
                    
                    # Check what was saved
                    check_query = text("""
                        SELECT temperature_2m, relative_humidity_2m, wind_speed_10m, 
                               precipitation, weather_code
                        FROM weather_data 
                        WHERE fixture_id = :fixture_id
                        ORDER BY weather_datetime DESC
                        LIMIT 1
                    """)
                    
                    result = session.execute(check_query, {'fixture_id': fixture_id}).fetchone()
                    if result:
                        logger.info(f"  Temperature: {result[0]}°C")
                        logger.info(f"  Humidity: {result[1]}%")
                        logger.info(f"  Wind Speed: {result[2]} km/h")
                        logger.info(f"  Precipitation: {result[3]} mm")
                        logger.info(f"  Weather Code: {result[4]}")
                    else:
                        logger.warning(f"  ⚠️ No weather data found after fetch")
                else:
                    logger.warning(f"  ❌ Failed to fetch weather data")
                    
            except Exception as e:
                logger.error(f"  ❌ Error fetching weather: {e}")
        
        # Check overall weather data coverage
        coverage_query = text("""
            SELECT 
                COUNT(DISTINCT f.id) as total_fixtures,
                COUNT(DISTINCT wd.fixture_id) as fixtures_with_weather,
                ROUND(COUNT(DISTINCT wd.fixture_id) * 100.0 / COUNT(DISTINCT f.id), 2) as coverage_pct
            FROM fixtures f
            LEFT JOIN weather_data wd ON f.id = wd.fixture_id
            WHERE f.status = 'finished'
                AND f.stadium_city IS NOT NULL
        """)
        
        coverage = session.execute(coverage_query).fetchone()
        logger.info(f"\nWeather Data Coverage:")
        logger.info(f"  Total fixtures: {coverage[0]}")
        logger.info(f"  Fixtures with weather: {coverage[1]}")
        logger.info(f"  Coverage: {coverage[2]}%")

if __name__ == "__main__":
    asyncio.run(test_weather_fetching())