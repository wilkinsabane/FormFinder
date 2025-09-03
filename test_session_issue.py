#!/usr/bin/env python3
"""
Test if session handling is causing the issue with MarkovFeatureGenerator.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from sqlalchemy import text
from datetime import datetime

def main():
    load_config()
    
    # Test with same session
    with get_db_session() as session:
        print("=== Testing with same session ===")
        
        # Test direct query
        state_query = """
            SELECT performance_state, state_score, state_date
            FROM team_performance_states
            WHERE team_id = :team_id
              AND league_id = :league_id
              AND home_away_context = :context
              AND state_date <= :reference_date
            ORDER BY state_date DESC
            LIMIT 1
        """
        
        team_id = 2962
        league_id = 224
        context = 'away'
        reference_date = datetime(2025, 8, 1, 18, 0)
        
        result = session.execute(
            text(state_query),
            {
                'team_id': team_id,
                'league_id': league_id,
                'context': context,
                'reference_date': reference_date
            }
        ).fetchone()
        
        print(f"Direct query result: {result}")
        
        # Test with MarkovFeatureGenerator using same session
        generator = MarkovFeatureGenerator(session)
        state_info = generator.get_current_state_info(
            team_id, league_id, reference_date, context
        )
        print(f"Generator result (same session): {state_info}")
    
    # Test with different session (current behavior)
    print("\n=== Testing with different sessions ===")
    with get_db_session() as session1:
        generator = MarkovFeatureGenerator(session1)
        
        # This will create its own session internally
        state_info = generator.get_current_state_info(
            team_id, league_id, reference_date, context
        )
        print(f"Generator result (different session): {state_info}")

if __name__ == "__main__":
    main()