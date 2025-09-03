#!/usr/bin/env python3
"""Check weather data coverage for fixtures in pre_computed_features."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

# Load configuration
load_config()

def check_weather_coverage():
    """Check weather data coverage."""
    with get_db_session() as session:
        # Check fixtures with weather data vs those using defaults
        result = session.execute(text('''
            SELECT 
                COUNT(*) as total_fixtures,
                COUNT(CASE WHEN pcf.weather_temp_c = 21.0 AND pcf.weather_condition = 'Clear' THEN 1 END) as default_weather,
                COUNT(CASE WHEN pcf.weather_temp_c != 21.0 OR pcf.weather_condition != 'Clear' THEN 1 END) as actual_weather
            FROM pre_computed_features pcf
        '''))
        
        row = result.fetchone()
        total = row[0]
        defaults = row[1]
        actual = row[2]
        
        print(f"Weather Data Coverage:")
        print(f"  Total fixtures: {total}")
        print(f"  Using default weather: {defaults} ({defaults/total*100:.1f}%)")
        print(f"  Using actual weather: {actual} ({actual/total*100:.1f}%)")
        
        # Check which fixtures have weather data in weather_data table
        result = session.execute(text('''
            SELECT 
                COUNT(DISTINCT pcf.fixture_id) as pcf_fixtures,
                COUNT(DISTINCT wd.fixture_id) as weather_fixtures,
                COUNT(DISTINCT CASE WHEN wd.fixture_id IS NOT NULL THEN pcf.fixture_id END) as matched_fixtures
            FROM pre_computed_features pcf
            LEFT JOIN weather_data wd ON pcf.fixture_id = wd.fixture_id
        '''))
        
        row = result.fetchone()
        pcf_fixtures = row[0]
        weather_fixtures = row[1]
        matched = row[2]
        
        print(f"\nWeather Data Availability:")
        print(f"  Fixtures in pre_computed_features: {pcf_fixtures}")
        print(f"  Fixtures with weather data: {weather_fixtures}")
        print(f"  Matched fixtures: {matched} ({matched/pcf_fixtures*100:.1f}%)")
        
        # Sample fixtures without weather data
        result = session.execute(text('''
            SELECT pcf.fixture_id, pcf.match_date, pcf.weather_temp_c, pcf.weather_condition
            FROM pre_computed_features pcf
            LEFT JOIN weather_data wd ON pcf.fixture_id = wd.fixture_id
            WHERE wd.fixture_id IS NULL
            ORDER BY pcf.match_date DESC
            LIMIT 10
        '''))
        
        print(f"\nSample fixtures without weather data:")
        for row in result.fetchall():
            print(f"  Fixture {row[0]} ({row[1]}): {row[2]}°C, {row[3]}")
        
        # Sample fixtures with weather data
        result = session.execute(text('''
            SELECT pcf.fixture_id, pcf.match_date, pcf.weather_temp_c, pcf.weather_condition,
                   wd.temperature_2m, wd.weather_code
            FROM pre_computed_features pcf
            INNER JOIN weather_data wd ON pcf.fixture_id = wd.fixture_id
            ORDER BY pcf.match_date DESC
            LIMIT 10
        '''))
        
        print(f"\nSample fixtures with weather data:")
        for row in result.fetchall():
            print(f"  Fixture {row[0]} ({row[1]}): PCF={row[2]}°C/{row[3]}, WD={row[4]}°C/code{row[5]}")

if __name__ == '__main__':
    check_weather_coverage()