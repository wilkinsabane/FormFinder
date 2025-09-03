#!/usr/bin/env python3
"""Check for identical weather and transition entropy data in pre_computed_features table."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from collections import Counter

# Load configuration
load_config()

def check_weather_data():
    """Check weather data variations."""
    print("=== WEATHER DATA ANALYSIS ===")
    
    with get_db_session() as session:
        # Check distinct weather values
        result = session.execute(text("""
            SELECT DISTINCT 
                weather_temp_c, 
                weather_humidity, 
                weather_wind_speed, 
                weather_precipitation, 
                weather_condition 
            FROM pre_computed_features 
            ORDER BY weather_temp_c
        """))
        
        weather_combinations = result.fetchall()
        print(f"Found {len(weather_combinations)} distinct weather combinations:")
        
        for combo in weather_combinations[:10]:  # Show first 10
            print(f"  Temp: {combo[0]}°C, Humidity: {combo[1]}%, Wind: {combo[2]}km/h, Precip: {combo[3]}mm, Condition: {combo[4]}")
        
        if len(weather_combinations) > 10:
            print(f"  ... and {len(weather_combinations) - 10} more combinations")
        
        # Check if most records use default values
        result = session.execute(text("""
            SELECT COUNT(*) as count,
                   weather_temp_c, 
                   weather_humidity, 
                   weather_wind_speed, 
                   weather_precipitation, 
                   weather_condition
            FROM pre_computed_features 
            GROUP BY weather_temp_c, weather_humidity, weather_wind_speed, weather_precipitation, weather_condition
            ORDER BY count DESC
            LIMIT 5
        """))
        
        print("\nMost common weather combinations:")
        for row in result.fetchall():
            print(f"  {row[0]} records: Temp={row[1]}°C, Humidity={row[2]}%, Wind={row[3]}km/h, Precip={row[4]}mm, Condition={row[5]}")

def check_transition_entropy():
    """Check transition entropy variations."""
    print("\n=== TRANSITION ENTROPY ANALYSIS ===")
    
    with get_db_session() as session:
        # Check distinct entropy values
        result = session.execute(text("""
            SELECT DISTINCT 
                home_team_transition_entropy, 
                away_team_transition_entropy
            FROM pre_computed_features 
            WHERE home_team_transition_entropy IS NOT NULL 
               OR away_team_transition_entropy IS NOT NULL
            ORDER BY home_team_transition_entropy
        """))
        
        entropy_combinations = result.fetchall()
        print(f"Found {len(entropy_combinations)} distinct entropy combinations:")
        
        for combo in entropy_combinations[:10]:  # Show first 10
            print(f"  Home entropy: {combo[0]}, Away entropy: {combo[1]}")
        
        if len(entropy_combinations) > 10:
            print(f"  ... and {len(entropy_combinations) - 10} more combinations")
        
        # Check most common entropy values
        result = session.execute(text("""
            SELECT COUNT(*) as count,
                   home_team_transition_entropy, 
                   away_team_transition_entropy
            FROM pre_computed_features 
            WHERE home_team_transition_entropy IS NOT NULL 
               OR away_team_transition_entropy IS NOT NULL
            GROUP BY home_team_transition_entropy, away_team_transition_entropy
            ORDER BY count DESC
            LIMIT 5
        """))
        
        print("\nMost common entropy combinations:")
        for row in result.fetchall():
            print(f"  {row[0]} records: Home={row[1]}, Away={row[2]}")
        
        # Check for NULL values
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(home_team_transition_entropy) as home_entropy_count,
                COUNT(away_team_transition_entropy) as away_entropy_count
            FROM pre_computed_features
        """))
        
        row = result.fetchone()
        print(f"\nEntropy data coverage:")
        print(f"  Total records: {row[0]}")
        print(f"  Records with home entropy: {row[1]} ({row[1]/row[0]*100:.1f}%)")
        print(f"  Records with away entropy: {row[2]} ({row[2]/row[0]*100:.1f}%)")

def check_sample_records():
    """Check a sample of recent records."""
    print("\n=== SAMPLE RECENT RECORDS ===")
    
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                fixture_id,
                weather_temp_c,
                weather_condition,
                home_team_transition_entropy,
                away_team_transition_entropy,
                features_computed_at
            FROM pre_computed_features 
            ORDER BY id DESC 
            LIMIT 10
        """))
        
        print("Recent records:")
        for row in result.fetchall():
            print(f"  Fixture {row[0]}: Weather={row[1]}°C/{row[2]}, Entropy=H:{row[3]}/A:{row[4]}, Computed={row[5]}")

if __name__ == '__main__':
    try:
        check_weather_data()
        check_transition_entropy()
        check_sample_records()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()