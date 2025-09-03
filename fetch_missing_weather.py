#!/usr/bin/env python3
"""Fetch missing weather data for fixtures."""

import asyncio
import argparse
import logging
from datetime import datetime
from typing import List, Tuple

from rich.console import Console
from rich.progress import Progress, TaskID
from sqlalchemy import text

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.weather_fetcher import WeatherFetcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class MissingWeatherFetcher:
    """Fetches missing weather data for fixtures."""
    
    def __init__(self):
        self.weather_fetcher = None
    
    async def initialize_weather_fetcher(self, session):
        """Initialize the weather fetcher."""
        try:
            self.weather_fetcher = WeatherFetcher(session)
            logger.info("Weather fetcher initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize weather fetcher: {e}")
            return False
    
    def get_fixtures_without_weather(self, session, limit: int = None) -> List[Tuple]:
        """Get fixtures that don't have weather data."""
        query = text("""
            SELECT f.id, f.stadium_city, f.match_date, f.home_team_id, f.away_team_id,
                   ht.name as home_team, at.name as away_team
            FROM fixtures f
            LEFT JOIN weather_data wd ON f.id = wd.fixture_id
            LEFT JOIN teams ht ON f.home_team_id = ht.id
            LEFT JOIN teams at ON f.away_team_id = at.id
            WHERE f.stadium_city IS NOT NULL 
                AND f.match_date IS NOT NULL
                AND f.status = 'finished'
                AND wd.fixture_id IS NULL
            ORDER BY f.match_date DESC
            {}
        """.format(f"LIMIT {limit}" if limit else ""))
        
        return session.execute(query).fetchall()
    
    def get_fixtures_with_default_weather(self, session, limit: int = None) -> List[Tuple]:
        """Get fixtures that have weather data but it's all defaults."""
        query = text("""
            SELECT DISTINCT f.id, f.stadium_city, f.match_date, f.home_team_id, f.away_team_id,
                   ht.name as home_team, at.name as away_team
            FROM fixtures f
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            LEFT JOIN teams ht ON f.home_team_id = ht.id
            LEFT JOIN teams at ON f.away_team_id = at.id
            WHERE f.match_date IS NOT NULL
                AND f.status = 'finished'
                AND pcf.weather_temp_c = 21.0  -- Default temperature
                AND pcf.weather_humidity = 50.0  -- Default humidity
            ORDER BY f.match_date DESC
            {}
        """.format(f"LIMIT {limit}" if limit else ""))
        
        return session.execute(query).fetchall()
    
    async def fetch_weather_for_fixture(self, fixture_data: Tuple) -> bool:
        """Fetch weather data for a single fixture with multiple city name attempts."""
        fixture_id, stadium_city, match_date, home_team_id, away_team_id, home_team, away_team = fixture_data
        
        if not self.weather_fetcher:
            return False
        
        # Skip fixtures without stadium_city
        if not stadium_city:
            logger.debug(f"Skipping fixture {fixture_id} ({home_team} vs {away_team}) - no stadium city")
            return False
        
        # Create a minimal fixture object
        class MinimalFixture:
            def __init__(self, fixture_id, stadium_city, match_date):
                self.id = fixture_id
                self.stadium_city = stadium_city
                self.match_date = match_date
        
        # Try multiple city name variations
        city_variations = [stadium_city]
        
        # Add simplified city name (remove county/region)
        if ',' in stadium_city:
            simplified_city = stadium_city.split(',')[0].strip()
            city_variations.append(simplified_city)
        
        # Try each city variation
        for city_variant in city_variations:
            try:
                minimal_fixture = MinimalFixture(fixture_id, city_variant, match_date)
                success = await self.weather_fetcher.fetch_weather_for_fixture(minimal_fixture)
                
                if success:
                    logger.info(f"✅ Fetched weather for fixture {fixture_id} ({home_team} vs {away_team}) using city: {city_variant}")
                    return True
                else:
                    logger.debug(f"❌ Failed to fetch weather for fixture {fixture_id} with city: {city_variant}")
                    
            except Exception as e:
                logger.debug(f"❌ Error fetching weather for fixture {fixture_id} with city {city_variant}: {e}")
                continue
        
        logger.warning(f"❌ Failed to fetch weather for fixture {fixture_id} ({home_team} vs {away_team}) with all city variants: {city_variations}")
        return False
    
    async def fetch_missing_weather(self, limit: int = None, include_defaults: bool = False):
        """Fetch missing weather data for fixtures."""
        load_config()
        
        with get_db_session() as session:
            # Initialize weather fetcher
            if not await self.initialize_weather_fetcher(session):
                console.print("❌ Failed to initialize weather fetcher")
                return
            
            fixtures_to_process = []
            
            if include_defaults:
                console.print("🔍 Finding fixtures with default weather data...")
                fixtures_with_defaults = self.get_fixtures_with_default_weather(session, limit)
                fixtures_to_process.extend(fixtures_with_defaults)
                console.print(f"Found {len(fixtures_with_defaults)} fixtures with default weather")
            
            # Get fixtures without weather data (only if we haven't reached the limit)
            remaining_limit = limit - len(fixtures_to_process) if limit else None
            if not limit or remaining_limit > 0:
                console.print("🔍 Finding fixtures without weather data...")
                fixtures_without_weather = self.get_fixtures_without_weather(session, remaining_limit)
                console.print(f"Found {len(fixtures_without_weather)} fixtures without weather data")
                
                # Remove duplicates by fixture_id
                existing_ids = {f[0] for f in fixtures_to_process}
                new_fixtures = [f for f in fixtures_without_weather if f[0] not in existing_ids]
                fixtures_to_process.extend(new_fixtures)
            
            if not fixtures_to_process:
                console.print("✅ No fixtures found that need weather data")
                return
            
            console.print(f"📊 Found {len(fixtures_to_process)} fixtures that need weather data")
            
            success_count = 0
            error_count = 0
            
            with Progress() as progress:
                task = progress.add_task("Fetching weather data...", total=len(fixtures_to_process))
                
                for fixture_data in fixtures_to_process:
                    try:
                        success = await self.fetch_weather_for_fixture(fixture_data)
                        if success:
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Error processing fixture {fixture_data[0]}: {e}")
                        error_count += 1
                    
                    progress.update(task, advance=1)
            
            console.print(f"\n📈 Results:")
            console.print(f"  ✅ Successfully fetched: {success_count}")
            console.print(f"  ❌ Failed: {error_count}")
            
            # Show updated coverage
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
            console.print(f"\n📊 Weather Data Coverage:")
            console.print(f"  Total fixtures: {coverage[0]}")
            console.print(f"  Fixtures with weather: {coverage[1]}")
            console.print(f"  Coverage: {coverage[2]}%")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fetch missing weather data for fixtures')
    parser.add_argument('--limit', type=int, help='Limit number of fixtures to process')
    parser.add_argument('--include-defaults', action='store_true', 
                       help='Also fetch weather for fixtures that currently have default values')
    
    args = parser.parse_args()
    
    try:
        fetcher = MissingWeatherFetcher()
        await fetcher.fetch_missing_weather(limit=args.limit, include_defaults=args.include_defaults)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        console.print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())