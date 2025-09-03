#!/usr/bin/env python3
"""Check pre_computed_features table schema for H2H and weather columns."""

import sys
sys.path.append('.')

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    """Check H2H and weather columns in pre_computed_features table."""
    load_config()
    
    with get_db_session() as db_session:
        # Get H2H and weather columns
        query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'pre_computed_features' 
            AND (column_name LIKE '%h2h%' OR column_name LIKE '%weather%')
            ORDER BY column_name
        """)
        
        result = db_session.execute(query).fetchall()
        
        print("H2H and Weather columns in pre_computed_features:")
        print("=" * 50)
        for row in result:
            print(f"{row[0]}: {row[1]}")
        
        print(f"\nTotal H2H/Weather columns found: {len(result)}")
        
        # Check if we have the expected columns
        expected_h2h = ['h2h_total_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_avg_goals']
        expected_weather = ['weather_temp_c', 'weather_humidity', 'weather_wind_speed', 'weather_condition']
        
        found_columns = [row[0] for row in result]
        
        print("\nExpected H2H columns:")
        for col in expected_h2h:
            status = "✓" if col in found_columns else "✗"
            print(f"  {status} {col}")
            
        print("\nExpected Weather columns:")
        for col in expected_weather:
            status = "✓" if col in found_columns else "✗"
            print(f"  {status} {col}")

if __name__ == "__main__":
    main()