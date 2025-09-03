#!/usr/bin/env python3
"""Re-fetch weather data for fixtures with NULL or missing weather data."""

import asyncio
import logging
from datetime import datetime
from formfinder.config import load_config
from formfinder.database import get_db_session, Fixture, WeatherData
from formfinder.weather_fetcher import WeatherFetcher
from sqlalchemy import and_, or_

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def refetch_weather_data(batch_size=10, delay_between_batches=2):
    """Re-fetch weather data for fixtures with NULL or missing data.
    
    Args:
        batch_size: Number of fixtures to process in each batch
        delay_between_batches: Seconds to wait between batches
    """
    
    config = load_config()
    
    with get_db_session() as session:
        # Get fixtures with NULL weather data
        fixtures_with_null_weather = session.query(Fixture).join(
            WeatherData, Fixture.id == WeatherData.fixture_id
        ).filter(
            and_(
                Fixture.stadium_city.isnot(None),
                Fixture.match_date.isnot(None),
                WeatherData.temperature_2m.is_(None)
            )
        ).all()
        
        # Get fixtures without any weather data
        fixtures_without_weather = session.query(Fixture).outerjoin(
            WeatherData, Fixture.id == WeatherData.fixture_id
        ).filter(
            and_(
                Fixture.stadium_city.isnot(None),
                Fixture.match_date.isnot(None),
                WeatherData.fixture_id.is_(None)
            )
        ).all()
        
        # Combine all fixtures that need weather data
        all_fixtures = fixtures_with_null_weather + fixtures_without_weather
        
        logger.info(f"Found {len(fixtures_with_null_weather)} fixtures with NULL weather data")
        logger.info(f"Found {len(fixtures_without_weather)} fixtures without weather data")
        logger.info(f"Total fixtures to process: {len(all_fixtures)}")
        
        if not all_fixtures:
            logger.info("No fixtures need weather data re-fetching")
            return
        
        # Create WeatherFetcher
        weather_fetcher = WeatherFetcher(session)
        
        # Process fixtures in batches
        successful_fetches = 0
        failed_fetches = 0
        
        for i in range(0, len(all_fixtures), batch_size):
            batch = all_fixtures[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_fixtures) + batch_size - 1) // batch_size
            
            logger.info(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} fixtures)")
            
            for fixture in batch:
                try:
                    logger.info(f"Fetching weather for fixture {fixture.id} ({fixture.stadium_city}, {fixture.match_date})")
                    
                    # Delete existing NULL weather data if it exists
                    existing_weather = session.query(WeatherData).filter(
                        WeatherData.fixture_id == fixture.id
                    ).first()
                    
                    if existing_weather and existing_weather.temperature_2m is None:
                        logger.info(f"Deleting existing NULL weather data for fixture {fixture.id}")
                        session.delete(existing_weather)
                        session.commit()
                    
                    # Fetch new weather data
                    success = await weather_fetcher.fetch_weather_for_fixture(fixture)
                    
                    if success:
                        successful_fetches += 1
                        logger.info(f"✓ Successfully fetched weather for fixture {fixture.id}")
                    else:
                        failed_fetches += 1
                        logger.warning(f"✗ Failed to fetch weather for fixture {fixture.id}")
                        
                except Exception as e:
                    failed_fetches += 1
                    logger.error(f"✗ Error fetching weather for fixture {fixture.id}: {e}")
                    
                # Small delay between individual requests
                await asyncio.sleep(0.5)
            
            # Delay between batches to be respectful to the API
            if i + batch_size < len(all_fixtures):
                logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                await asyncio.sleep(delay_between_batches)
        
        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"Total fixtures processed: {len(all_fixtures)}")
        logger.info(f"Successful fetches: {successful_fetches}")
        logger.info(f"Failed fetches: {failed_fetches}")
        logger.info(f"Success rate: {(successful_fetches / len(all_fixtures) * 100):.1f}%")
        
        # Verify the results
        logger.info(f"\n=== VERIFICATION ===")
        remaining_null = session.query(WeatherData).filter(
            WeatherData.temperature_2m.is_(None)
        ).count()
        logger.info(f"Remaining weather records with NULL temperature: {remaining_null}")
        
        remaining_without_weather = session.query(Fixture).outerjoin(
            WeatherData, Fixture.id == WeatherData.fixture_id
        ).filter(
            and_(
                Fixture.stadium_city.isnot(None),
                Fixture.match_date.isnot(None),
                WeatherData.fixture_id.is_(None)
            )
        ).count()
        logger.info(f"Remaining fixtures without weather data: {remaining_without_weather}")

if __name__ == "__main__":
    asyncio.run(refetch_weather_data())