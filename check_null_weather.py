#!/usr/bin/env python3
"""Check how many fixtures have NULL weather data."""

from formfinder.config import load_config
from formfinder.database import get_db_session, Fixture, WeatherData
from sqlalchemy import and_, or_

def check_null_weather():
    """Check fixtures with NULL weather data."""
    
    config = load_config()
    
    with get_db_session() as session:
        # Count total fixtures with weather data
        total_weather_records = session.query(WeatherData).count()
        print(f"Total weather data records: {total_weather_records}")
        
        # Count weather records with NULL temperature (main indicator)
        null_temp_records = session.query(WeatherData).filter(
            WeatherData.temperature_2m.is_(None)
        ).count()
        print(f"Weather records with NULL temperature: {null_temp_records}")
        
        # Count weather records with valid temperature
        valid_temp_records = session.query(WeatherData).filter(
            WeatherData.temperature_2m.isnot(None)
        ).count()
        print(f"Weather records with valid temperature: {valid_temp_records}")
        
        # Get some examples of NULL weather data
        print(f"\nExamples of NULL weather data:")
        null_examples = session.query(WeatherData).filter(
            WeatherData.temperature_2m.is_(None)
        ).limit(5).all()
        
        for wd in null_examples:
            print(f"  Fixture {wd.fixture_id}: temp={wd.temperature_2m}, humidity={wd.relative_humidity_2m}, wind={wd.wind_speed_10m}, datetime={wd.weather_datetime}")
        
        # Get some examples of valid weather data
        print(f"\nExamples of valid weather data:")
        valid_examples = session.query(WeatherData).filter(
            WeatherData.temperature_2m.isnot(None)
        ).limit(5).all()
        
        for wd in valid_examples:
            print(f"  Fixture {wd.fixture_id}: temp={wd.temperature_2m}, humidity={wd.relative_humidity_2m}, wind={wd.wind_speed_10m}, datetime={wd.weather_datetime}")
        
        # Check fixtures that need weather data re-fetching
        fixtures_needing_refetch = session.query(Fixture).join(
            WeatherData, Fixture.id == WeatherData.fixture_id
        ).filter(
            and_(
                Fixture.stadium_city.isnot(None),
                Fixture.match_date.isnot(None),
                WeatherData.temperature_2m.is_(None)
            )
        ).count()
        
        print(f"\nFixtures with NULL weather data that can be re-fetched: {fixtures_needing_refetch}")
        
        # Check fixtures without any weather data
        fixtures_without_weather = session.query(Fixture).outerjoin(
            WeatherData, Fixture.id == WeatherData.fixture_id
        ).filter(
            and_(
                Fixture.stadium_city.isnot(None),
                Fixture.match_date.isnot(None),
                WeatherData.fixture_id.is_(None)
            )
        ).count()
        
        print(f"Fixtures without any weather data: {fixtures_without_weather}")
        
        # Total fixtures that need weather data
        total_needing_weather = fixtures_needing_refetch + fixtures_without_weather
        print(f"\nTotal fixtures needing weather data: {total_needing_weather}")

if __name__ == "__main__":
    check_null_weather()