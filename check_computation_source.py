#!/usr/bin/env python3
"""Check which system populated the pre_computed_features data."""

import sys
sys.path.append('.')

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
from collections import Counter

def main():
    """Check computation sources and data patterns."""
    load_config()
    
    with get_db_session() as db_session:
        # Check computation sources
        source_query = text("""
            SELECT computation_source, COUNT(*) as count
            FROM pre_computed_features 
            WHERE computation_source IS NOT NULL
            GROUP BY computation_source
            ORDER BY count DESC
        """)
        
        source_result = db_session.execute(source_query).fetchall()
        
        print("Computation Sources:")
        print("=" * 30)
        for row in source_result:
            print(f"{row[0]}: {row[1]} records")
        
        # Check which H2H columns have data
        h2h_query = text("""
            SELECT 
                COUNT(h2h_total_matches) as new_h2h_total,
                COUNT(h2h_overall_games) as old_h2h_total,
                COUNT(h2h_home_wins) as new_h2h_home_wins,
                COUNT(h2h_team1_wins) as old_h2h_team1_wins,
                COUNT(h2h_away_wins) as new_h2h_away_wins,
                COUNT(h2h_team2_wins) as old_h2h_team2_wins
            FROM pre_computed_features
        """)
        
        h2h_result = db_session.execute(h2h_query).fetchone()
        
        print("\nH2H Column Data Population:")
        print("=" * 35)
        print(f"New H2H columns:")
        print(f"  h2h_total_matches: {h2h_result[0]} non-NULL")
        print(f"  h2h_home_wins: {h2h_result[2]} non-NULL")
        print(f"  h2h_away_wins: {h2h_result[4]} non-NULL")
        print(f"\nOld H2H columns:")
        print(f"  h2h_overall_games: {h2h_result[1]} non-NULL")
        print(f"  h2h_team1_wins: {h2h_result[3]} non-NULL")
        print(f"  h2h_team2_wins: {h2h_result[5]} non-NULL")
        
        # Check weather data patterns
        weather_query = text("""
            SELECT 
                weather_temp_c,
                weather_humidity,
                weather_wind_speed,
                weather_condition,
                COUNT(*) as count
            FROM pre_computed_features 
            WHERE weather_temp_c IS NOT NULL
            GROUP BY weather_temp_c, weather_humidity, weather_wind_speed, weather_condition
            ORDER BY count DESC
            LIMIT 10
        """)
        
        weather_result = db_session.execute(weather_query).fetchall()
        
        print("\nWeather Data Patterns:")
        print("=" * 25)
        for row in weather_result:
            print(f"Temp: {row[0]}°C, Humidity: {row[1]}%, Wind: {row[2]}, Condition: {row[3]} - {row[4]} records")
        
        # Check recent records to see which system is being used
        recent_query = text("""
            SELECT 
                fixture_id,
                computation_source,
                features_computed_at,
                h2h_total_matches,
                h2h_overall_games,
                weather_temp_c
            FROM pre_computed_features 
            ORDER BY features_computed_at DESC NULLS LAST
            LIMIT 5
        """)
        
        recent_result = db_session.execute(recent_query).fetchall()
        
        print("\nMost Recent Records:")
        print("=" * 25)
        for row in recent_result:
            print(f"Fixture {row[0]}: source={row[1]}, computed={row[2]}, new_h2h={row[3]}, old_h2h={row[4]}, weather={row[5]}°C")

if __name__ == "__main__":
    main()