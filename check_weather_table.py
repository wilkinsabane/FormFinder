#!/usr/bin/env python3
"""Check weather data table contents."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

# Load configuration
load_config()

def check_weather_table():
    """Check weather data table."""
    with get_db_session() as session:
        # First check table structure
        result = session.execute(text('''
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'weather_data' 
            ORDER BY ordinal_position
        '''))
        print("Weather data table columns:")
        columns = []
        for row in result.fetchall():
            columns.append(row[0])
            print(f"  {row[0]}: {row[1]}")
        
        # Count total weather records
        result = session.execute(text('SELECT COUNT(*) FROM weather_data'))
        total_records = result.scalar()
        print(f'\nTotal weather data records: {total_records}')
        
        # Count fixtures with weather data
        result = session.execute(text('SELECT COUNT(DISTINCT fixture_id) FROM weather_data'))
        fixtures_with_weather = result.scalar()
        print(f'Fixtures with weather data: {fixtures_with_weather}')
        
        # Sample weather data using correct column names
        if total_records > 0:
            result = session.execute(text('SELECT * FROM weather_data ORDER BY created_at DESC LIMIT 5'))
            print("\nSample weather data:")
            for row in result.fetchall():
                print(f"  Row: {dict(zip(columns, row))}")
        else:
            print("No weather data found in the table.")

if __name__ == '__main__':
    check_weather_table()