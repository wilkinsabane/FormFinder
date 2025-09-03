#!/usr/bin/env python3
"""
Test script to verify that the strength and form features are now properly populated
after fixing the mapping in populate_precomputed_features_unified.py
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check fixture 238 that we just processed
        query = text("""
            SELECT fixture_id, home_attack_strength, home_defense_strength, 
                   away_attack_strength, away_defense_strength, 
                   home_form_diff, away_form_diff, 
                   home_team_form_score, away_team_form_score
            FROM pre_computed_features 
            WHERE fixture_id = 238
        """)
        
        result = session.execute(query).fetchone()
        
        if result:
            print(f"Fixture {result[0]}:")
            print(f"  home_attack_strength: {result[1]}")
            print(f"  home_defense_strength: {result[2]}")
            print(f"  away_attack_strength: {result[3]}")
            print(f"  away_defense_strength: {result[4]}")
            print(f"  home_form_diff: {result[5]}")
            print(f"  away_form_diff: {result[6]}")
            print(f"  home_team_form_score: {result[7]}")
            print(f"  away_team_form_score: {result[8]}")
            
            # Check if any are still null
            null_features = []
            feature_names = ['home_attack_strength', 'home_defense_strength', 
                           'away_attack_strength', 'away_defense_strength',
                           'home_form_diff', 'away_form_diff',
                           'home_team_form_score', 'away_team_form_score']
            
            for i, feature_name in enumerate(feature_names, 1):
                if result[i] is None:
                    null_features.append(feature_name)
            
            if null_features:
                print(f"\n❌ Still null features: {null_features}")
            else:
                print(f"\n✅ All strength and form features are now populated!")
        else:
            print("❌ No data found for fixture 238")

if __name__ == "__main__":
    main()