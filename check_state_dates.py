#!/usr/bin/env python3
"""
Check team performance state dates for debugging Markov feature generator.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime

def main():
    load_config()
    
    with get_db_session() as session:
        # Check all states for test teams
        states_query = """
            SELECT team_id, performance_state, state_date, home_away_context 
            FROM team_performance_states 
            WHERE team_id IN (2962, 3031) AND league_id = 224 
            ORDER BY state_date
        """
        
        states = session.execute(text(states_query)).fetchall()
        print("All states for teams 2962/3031:")
        for state in states:
            print(f"  Team {state[0]}: {state[1]} on {state[2]} ({state[3]})")
        
        print("\nReference date: 2025-08-01 18:00:00")
        ref_date = datetime(2025, 8, 1, 18, 0)
        
        print("\nStates before reference date:")
        before_ref = [s for s in states if s[2] <= ref_date]
        for state in before_ref:
            print(f"  Team {state[0]}: {state[1]} on {state[2]} ({state[3]})")
        
        print("\nStates after reference date:")
        after_ref = [s for s in states if s[2] > ref_date]
        for state in after_ref:
            print(f"  Team {state[0]}: {state[1]} on {state[2]} ({state[3]})")
        
        # Test the exact query used by get_current_state_info
        print("\n=== Testing get_current_state_info queries ===")
        
        for team_id in [2962, 3031]:
            for context in ['home', 'away']:
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
                
                result = session.execute(
                    text(state_query),
                    {
                        'team_id': team_id,
                        'league_id': 224,
                        'context': context,
                        'reference_date': ref_date
                    }
                ).fetchone()
                
                if result:
                    print(f"  Team {team_id}, context '{context}': {result[0]} (score: {result[1]}, date: {result[2]})")
                else:
                    print(f"  Team {team_id}, context '{context}': No state found")
                    
                    # Try fallback query (without date restriction)
                    fallback_query = """
                        SELECT performance_state, state_score, state_date
                        FROM team_performance_states
                        WHERE team_id = :team_id
                          AND league_id = :league_id
                          AND home_away_context = :context
                        ORDER BY state_date DESC
                        LIMIT 1
                    """
                    
                    fallback_result = session.execute(
                        text(fallback_query),
                        {
                            'team_id': team_id,
                            'league_id': 224,
                            'context': context
                        }
                    ).fetchone()
                    
                    if fallback_result:
                        print(f"    Fallback found: {fallback_result[0]} (score: {fallback_result[1]}, date: {fallback_result[2]})")
                    else:
                        print(f"    No fallback found either")

if __name__ == "__main__":
    main()