#!/usr/bin/env python3
"""
Script to check data quality of pre_computed_features table.
"""

import sys
import os
from sqlalchemy import text

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_session
from formfinder.config import load_config

def check_data_quality():
    """Check the data quality of pre_computed_features table."""
    # Load configuration first
    load_config()
    with get_db_session() as session:
        # Total records
        result = session.execute(text('SELECT COUNT(*) as total FROM pre_computed_features')).fetchone()
        total_records = result[0]
        print(f"Total records: {total_records}")
        
        if total_records == 0:
            print("No records found in pre_computed_features table")
            return
        
        # Check for zero/null strength values
        result = session.execute(text(
            'SELECT COUNT(*) as zero_strength FROM pre_computed_features WHERE home_team_strength = 0 OR home_team_strength IS NULL'
        )).fetchone()
        zero_strength = result[0]
        print(f"Zero/null home_team_strength: {zero_strength} ({zero_strength/total_records*100:.1f}%)")
        
        # Check for zero/null momentum values
        result = session.execute(text(
            'SELECT COUNT(*) as zero_momentum FROM pre_computed_features WHERE home_team_momentum = 0 OR home_team_momentum IS NULL'
        )).fetchone()
        zero_momentum = result[0]
        print(f"Zero/null home_team_momentum: {zero_momentum} ({zero_momentum/total_records*100:.1f}%)")
        
        # Check for zero/null sentiment values
        result = session.execute(text(
            'SELECT COUNT(*) as zero_sentiment FROM pre_computed_features WHERE home_team_sentiment = 0 OR home_team_sentiment IS NULL'
        )).fetchone()
        zero_sentiment = result[0]
        print(f"Zero/null home_team_sentiment: {zero_sentiment} ({zero_sentiment/total_records*100:.1f}%)")
        
        # Check for default weather values (21.0 is the default temperature)
        result = session.execute(text(
            'SELECT COUNT(*) as default_weather FROM pre_computed_features WHERE weather_temp_c = 21.0 OR weather_temp_c IS NULL'
        )).fetchone()
        default_weather = result[0]
        print(f"Default/null weather_temp_c: {default_weather} ({default_weather/total_records*100:.1f}%)")
        
        # Check for null xG values
        result = session.execute(text(
            'SELECT COUNT(*) as null_xg FROM pre_computed_features WHERE home_xg IS NULL OR away_xg IS NULL'
        )).fetchone()
        null_xg = result[0]
        print(f"Null xG values: {null_xg} ({null_xg/total_records*100:.1f}%)")
        
        # Sample some records to see actual values
        print("\nSample records:")
        result = session.execute(text(
            'SELECT fixture_id, home_team_strength, home_team_momentum, home_team_sentiment, weather_temp_c, home_xg '
            'FROM pre_computed_features LIMIT 5'
        )).fetchall()
        
        for row in result:
            print(f"Fixture {row[0]}: strength={row[1]}, momentum={row[2]}, sentiment={row[3]}, temp={row[4]}, xg={row[5]}")

if __name__ == "__main__":
    check_data_quality()