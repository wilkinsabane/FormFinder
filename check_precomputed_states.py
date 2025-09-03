#!/usr/bin/env python3
"""
Check what team states are being generated for the teams in pre_computed_features.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def check_precomputed_states():
    """Check team states in pre_computed_features."""
    # Load configuration first
    load_config()
    
    with get_db_session() as session:
        try:
            # Check home and away team states
            query = """
            SELECT 
                fixture_id,
                home_team_id,
                away_team_id,
                markov_home_current_state,
                markov_away_current_state,
                markov_outcome_probabilities
            FROM pre_computed_features 
            WHERE markov_outcome_probabilities IS NOT NULL 
            ORDER BY fixture_id
            LIMIT 10
            """
            
            result = session.execute(text(query))
            rows = result.fetchall()
            
            print("Team states and outcome probabilities:")
            print("Fixture ID | Home Team | Away Team | Home State | Away State | Outcome Probabilities")
            print("-" * 100)
            
            for row in rows:
                fixture_id, home_team_id, away_team_id, home_state, away_state, outcome_probs = row
                print(f"{fixture_id:10} | {home_team_id:9} | {away_team_id:9} | {home_state:10} | {away_state:10} | {outcome_probs}")
            
            # Check distinct states
            distinct_query = """
            SELECT DISTINCT 
                markov_home_current_state,
                markov_away_current_state
            FROM pre_computed_features 
            WHERE markov_home_current_state IS NOT NULL 
               OR markov_away_current_state IS NOT NULL
            """
            
            distinct_result = session.execute(text(distinct_query))
            distinct_rows = distinct_result.fetchall()
            
            print("\nDistinct team states:")
            for row in distinct_rows:
                home_state, away_state = row
                print(f"Home: {home_state}, Away: {away_state}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_precomputed_states()