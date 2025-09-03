#!/usr/bin/env python3
"""
Test the MarkovFeatureGenerator for the specific fixture that should have non-average states.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from datetime import datetime
from sqlalchemy import text

def main():
    load_config()
    
    # Test fixture 20508: Team 2962 (home) vs Team 3031 (away) in league 224
    fixture_id = 20508
    home_team_id = 2962
    away_team_id = 3031
    league_id = 224
    match_date = datetime(2025, 8, 24, 18, 0)
    
    print(f"Testing fixture {fixture_id}: Team {home_team_id} (home) vs Team {away_team_id} (away)")
    print(f"League: {league_id}, Date: {match_date}")
    print()
    
    # Initialize MarkovFeatureGenerator
    markov_gen = MarkovFeatureGenerator()
    
    # Test home team state
    print("=== HOME TEAM (2962) ===")
    home_state_info = markov_gen.get_current_state_info(
        home_team_id, league_id, match_date, 'home'
    )
    print(f"Current state info: {home_state_info}")
    
    home_features = markov_gen.generate_team_features(
        home_team_id, league_id, match_date, 'home'
    )
    print(f"Generated features: {home_features}")
    print()
    
    # Test away team state
    print("=== AWAY TEAM (3031) ===")
    away_state_info = markov_gen.get_current_state_info(
        away_team_id, league_id, match_date, 'away'
    )
    print(f"Current state info: {away_state_info}")
    
    away_features = markov_gen.generate_team_features(
        away_team_id, league_id, match_date, 'away'
    )
    print(f"Generated features: {away_features}")
    print()
    
    # Check what's actually in the database for these teams
    print("=== DATABASE VERIFICATION ===")
    with get_db_session() as session:
        for team_id, context in [(home_team_id, 'home'), (away_team_id, 'away')]:
            query = text("""
                SELECT performance_state, state_score, state_date
                FROM team_performance_states
                WHERE team_id = :team_id
                  AND league_id = :league_id
                  AND home_away_context = :context
                  AND state_date <= :reference_date
                ORDER BY state_date DESC
                LIMIT 3
            """)
            
            results = session.execute(query, {
                'team_id': team_id,
                'league_id': league_id,
                'context': context,
                'reference_date': match_date
            }).fetchall()
            
            print(f"Team {team_id} ({context}) states in database:")
            if results:
                for result in results:
                    print(f"  {result[0]} (score: {result[1]}, date: {result[2]})")
            else:
                print(f"  No states found")
            print()

if __name__ == '__main__':
    main()