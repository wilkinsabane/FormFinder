#!/usr/bin/env python3
"""Check default weather values in the database."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_default_weather():
    """Check what the default weather values are."""
    load_config()
    
    with get_db_session() as session:
        # Check distinct weather values for fixtures with 21.0°C
        query = text("""
            SELECT DISTINCT weather_temp_c, weather_humidity, weather_condition, COUNT(*) as count
            FROM pre_computed_features 
            WHERE weather_temp_c = 21.0
            GROUP BY weather_temp_c, weather_humidity, weather_condition
            ORDER BY count DESC
        """)
        
        results = session.execute(query).fetchall()
        print("Default weather combinations (temp=21.0°C):")
        for result in results:
            print(f"  {result.weather_temp_c}°C, {result.weather_humidity}%, {result.weather_condition} - {result.count} fixtures")
        
        # Check what humidity value is most common with 21.0°C
        humidity_query = text("""
            SELECT weather_humidity, COUNT(*) as count
            FROM pre_computed_features 
            WHERE weather_temp_c = 21.0
            GROUP BY weather_humidity
            ORDER BY count DESC
            LIMIT 5
        """)
        
        humidity_results = session.execute(humidity_query).fetchall()
        print("\nMost common humidity values with 21.0°C:")
        for result in humidity_results:
            print(f"  {result.weather_humidity}% - {result.count} fixtures")

if __name__ == "__main__":
    check_default_weather()