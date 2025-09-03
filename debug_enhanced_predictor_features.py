#!/usr/bin/env python3
"""Debug what features the EnhancedGoalPredictor actually returns."""

import sys
sys.path.append('.')

from formfinder.database import get_db_session
from formfinder.config import load_config
from enhanced_predictor import EnhancedGoalPredictor
from sqlalchemy import text
import json

def main():
    """Debug EnhancedGoalPredictor feature output."""
    load_config()
    
    with get_db_session() as db_session:
        # Get a recent fixture
        query = text("""
            SELECT id, home_team_id, away_team_id, match_date
            FROM fixtures 
            WHERE match_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY match_date DESC
            LIMIT 1
        """)
        
        result = db_session.execute(query).fetchone()
        
        if not result:
            print("No recent fixtures found")
            return
        
        fixture_id = result[0]
        print(f"Testing fixture {fixture_id}")
        
        # Initialize EnhancedGoalPredictor
        try:
            predictor = EnhancedGoalPredictor()
            features = predictor.extract_enhanced_features(fixture_id)
            
            print("\n=== ALL FEATURES FROM ENHANCED PREDICTOR ===")
            print("=" * 50)
            
            # Group features by type
            h2h_features = {}
            weather_features = {}
            other_features = {}
            
            for key, value in features.items():
                if key.startswith('h2h_'):
                    h2h_features[key] = value
                elif key.startswith('weather_'):
                    weather_features[key] = value
                else:
                    other_features[key] = value
            
            print("\n=== H2H FEATURES ===")
            for key, value in sorted(h2h_features.items()):
                print(f"{key}: {value}")
            
            print("\n=== WEATHER FEATURES ===")
            for key, value in sorted(weather_features.items()):
                print(f"{key}: {value}")
            
            print("\n=== OTHER FEATURES (first 10) ===")
            for i, (key, value) in enumerate(sorted(other_features.items())):
                if i >= 10:
                    print(f"... and {len(other_features) - 10} more features")
                    break
                print(f"{key}: {value}")
            
            print(f"\nTotal features: {len(features)}")
            print(f"H2H features: {len(h2h_features)}")
            print(f"Weather features: {len(weather_features)}")
            print(f"Other features: {len(other_features)}")
            
            # Check specific feature names that unified script expects
            expected_h2h = ['h2h_total_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_avg_goals']
            expected_weather = ['weather_temperature', 'weather_humidity', 'weather_wind_speed', 'weather_condition']
            
            print("\n=== FEATURE NAME MAPPING CHECK ===")
            print("Expected H2H features:")
            for feature in expected_h2h:
                if feature in features:
                    print(f"  ✓ {feature}: {features[feature]}")
                else:
                    print(f"  ✗ {feature}: NOT FOUND")
            
            print("\nExpected Weather features:")
            for feature in expected_weather:
                if feature in features:
                    print(f"  ✓ {feature}: {features[feature]}")
                else:
                    print(f"  ✗ {feature}: NOT FOUND")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()