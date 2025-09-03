#!/usr/bin/env python3
"""Debug the default weather query."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def debug_default_weather_query():
    """Debug what the default weather query returns."""
    load_config()
    
    with get_db_session() as session:
        # This is the exact query from fetch_missing_weather.py
        query = text("""
            SELECT DISTINCT f.id, f.stadium_city, f.match_date, f.home_team_id, f.away_team_id,
                   ht.name as home_team, at.name as away_team
            FROM fixtures f
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            LEFT JOIN teams ht ON f.home_team_id = ht.id
            LEFT JOIN teams at ON f.away_team_id = at.id
            WHERE f.stadium_city IS NOT NULL 
                AND f.match_date IS NOT NULL
                AND f.status = 'finished'
                AND pcf.weather_temp_c = 21.0  -- Default temperature
                AND pcf.weather_humidity = 50.0  -- Default humidity
            ORDER BY f.match_date DESC
            LIMIT 10
        """)
        
        results = session.execute(query).fetchall()
        print(f"Query returned {len(results)} fixtures with default weather:")
        for result in results:
            print(f"  {result.id}: {result.home_team} vs {result.away_team} ({result.stadium_city}) - {result.match_date}")
        
        # Let's also check how many fixtures have default weather in total
        count_query = text("""
            SELECT COUNT(DISTINCT f.id)
            FROM fixtures f
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.stadium_city IS NOT NULL 
                AND f.match_date IS NOT NULL
                AND f.status = 'finished'
                AND pcf.weather_temp_c = 21.0
                AND pcf.weather_humidity = 50.0
        """)
        
        total_count = session.execute(count_query).scalar()
        print(f"\nTotal fixtures with default weather: {total_count}")
        
        # Check if there are any fixtures with default weather but different condition
        condition_query = text("""
            SELECT DISTINCT pcf.weather_condition, COUNT(*)
            FROM fixtures f
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.stadium_city IS NOT NULL 
                AND f.match_date IS NOT NULL
                AND f.status = 'finished'
                AND pcf.weather_temp_c = 21.0
                AND pcf.weather_humidity = 50.0
            GROUP BY pcf.weather_condition
        """)
        
        conditions = session.execute(condition_query).fetchall()
        print(f"\nWeather conditions for default weather:")
        for condition in conditions:
            print(f"  {condition[0]}: {condition[1]} fixtures")

if __name__ == "__main__":
    debug_default_weather_query()