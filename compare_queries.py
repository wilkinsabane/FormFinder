#!/usr/bin/env python3
"""Compare the different queries to understand the discrepancy."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def compare_queries():
    """Compare different queries for default weather."""
    load_config()
    
    with get_db_session() as session:
        # Query 1: From check_weather_coverage.py (shows 3828)
        query1 = text("""
            SELECT COUNT(*) FROM fixtures f 
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id 
            WHERE f.status = 'finished' 
                AND pcf.weather_temp_c = 21.0 
                AND pcf.weather_humidity = 50.0
        """)
        count1 = session.execute(query1).scalar()
        print(f"Query 1 (check_weather_coverage): {count1} fixtures")
        
        # Query 2: From fetch_missing_weather.py (shows 2)
        query2 = text("""
            SELECT COUNT(DISTINCT f.id)
            FROM fixtures f
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.stadium_city IS NOT NULL 
                AND f.match_date IS NOT NULL
                AND f.status = 'finished'
                AND pcf.weather_temp_c = 21.0
                AND pcf.weather_humidity = 50.0
        """)
        count2 = session.execute(query2).scalar()
        print(f"Query 2 (fetch_missing_weather): {count2} fixtures")
        
        # Let's check what's different - stadium_city and match_date filters
        query3 = text("""
            SELECT COUNT(*) FROM fixtures f 
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id 
            WHERE f.status = 'finished' 
                AND pcf.weather_temp_c = 21.0 
                AND pcf.weather_humidity = 50.0
                AND f.stadium_city IS NOT NULL
        """)
        count3 = session.execute(query3).scalar()
        print(f"Query 3 (with stadium_city filter): {count3} fixtures")
        
        query4 = text("""
            SELECT COUNT(*) FROM fixtures f 
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id 
            WHERE f.status = 'finished' 
                AND pcf.weather_temp_c = 21.0 
                AND pcf.weather_humidity = 50.0
                AND f.stadium_city IS NOT NULL
                AND f.match_date IS NOT NULL
        """)
        count4 = session.execute(query4).scalar()
        print(f"Query 4 (with both filters): {count4} fixtures")
        
        # Check how many fixtures have NULL stadium_city or match_date
        null_city_query = text("""
            SELECT COUNT(*) FROM fixtures f 
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id 
            WHERE f.status = 'finished' 
                AND pcf.weather_temp_c = 21.0 
                AND pcf.weather_humidity = 50.0
                AND f.stadium_city IS NULL
        """)
        null_city_count = session.execute(null_city_query).scalar()
        print(f"Fixtures with NULL stadium_city: {null_city_count}")
        
        null_date_query = text("""
            SELECT COUNT(*) FROM fixtures f 
            JOIN pre_computed_features pcf ON f.id = pcf.fixture_id 
            WHERE f.status = 'finished' 
                AND pcf.weather_temp_c = 21.0 
                AND pcf.weather_humidity = 50.0
                AND f.match_date IS NULL
        """)
        null_date_count = session.execute(null_date_query).scalar()
        print(f"Fixtures with NULL match_date: {null_date_count}")

if __name__ == "__main__":
    compare_queries()