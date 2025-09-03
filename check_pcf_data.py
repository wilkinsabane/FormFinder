#!/usr/bin/env python3
"""Check actual H2H and weather data variation in pre_computed_features."""

import sys
sys.path.append('.')

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
from collections import Counter

def main():
    """Check H2H and weather data variation in pre_computed_features table."""
    load_config()
    
    with get_db_session() as db_session:
        # Get sample H2H and weather data
        query = text("""
            SELECT fixture_id, 
                   h2h_total_matches, h2h_home_wins, h2h_away_wins, h2h_draws, h2h_avg_goals,
                   h2h_overall_games, h2h_team1_wins, h2h_team2_wins,
                   weather_temp_c, weather_humidity, weather_wind_speed, weather_condition
            FROM pre_computed_features 
            WHERE fixture_id IS NOT NULL
            ORDER BY fixture_id DESC
            LIMIT 20
        """)
        
        result = db_session.execute(query).fetchall()
        
        print("Sample H2H and Weather data from pre_computed_features:")
        print("=" * 70)
        
        h2h_total_values = []
        h2h_home_wins_values = []
        weather_temp_values = []
        weather_condition_values = []
        
        for i, row in enumerate(result):
            print(f"Fixture {row[0]}:")
            print(f"  H2H (new): matches={row[1]}, home_wins={row[2]}, away_wins={row[3]}, draws={row[4]}, avg_goals={row[5]}")
            print(f"  H2H (old): games={row[6]}, team1_wins={row[7]}, team2_wins={row[8]}")
            print(f"  Weather: temp={row[9]}°C, humidity={row[10]}%, wind={row[11]}, condition={row[12]}")
            print()
            
            # Collect values for variation analysis
            if row[1] is not None:
                h2h_total_values.append(row[1])
            if row[2] is not None:
                h2h_home_wins_values.append(row[2])
            if row[9] is not None:
                weather_temp_values.append(float(row[9]))
            if row[12] is not None:
                weather_condition_values.append(row[12])
        
        print("\nData Variation Analysis:")
        print("=" * 30)
        
        print(f"H2H Total Matches - Unique values: {len(set(h2h_total_values))} out of {len(h2h_total_values)}")
        if h2h_total_values:
            print(f"  Values: {sorted(set(h2h_total_values))}")
        
        print(f"H2H Home Wins - Unique values: {len(set(h2h_home_wins_values))} out of {len(h2h_home_wins_values)}")
        if h2h_home_wins_values:
            print(f"  Values: {sorted(set(h2h_home_wins_values))}")
        
        print(f"Weather Temperature - Unique values: {len(set(weather_temp_values))} out of {len(weather_temp_values)}")
        if weather_temp_values:
            print(f"  Values: {sorted(set(weather_temp_values))}")
        
        print(f"Weather Condition - Unique values: {len(set(weather_condition_values))} out of {len(weather_condition_values)}")
        if weather_condition_values:
            condition_counts = Counter(weather_condition_values)
            print(f"  Values: {dict(condition_counts)}")
        
        # Check for NULL values
        print("\nNULL Value Analysis:")
        print("=" * 20)
        
        null_query = text("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(h2h_total_matches) as h2h_total_non_null,
                COUNT(h2h_home_wins) as h2h_home_wins_non_null,
                COUNT(weather_temp_c) as weather_temp_non_null,
                COUNT(weather_condition) as weather_condition_non_null
            FROM pre_computed_features
        """)
        
        null_result = db_session.execute(null_query).fetchone()
        
        print(f"Total rows: {null_result[0]}")
        print(f"H2H Total Matches non-NULL: {null_result[1]} ({null_result[1]/null_result[0]*100:.1f}%)")
        print(f"H2H Home Wins non-NULL: {null_result[2]} ({null_result[2]/null_result[0]*100:.1f}%)")
        print(f"Weather Temp non-NULL: {null_result[3]} ({null_result[3]/null_result[0]*100:.1f}%)")
        print(f"Weather Condition non-NULL: {null_result[4]} ({null_result[4]/null_result[0]*100:.1f}%)")

if __name__ == "__main__":
    main()