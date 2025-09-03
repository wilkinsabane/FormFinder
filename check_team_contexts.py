#!/usr/bin/env python3
"""
Check team performance state contexts for debugging Markov feature generator.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check available contexts for test teams
        contexts_query = """
            SELECT DISTINCT home_away_context 
            FROM team_performance_states 
            WHERE team_id IN (2962, 3031) 
            ORDER BY home_away_context
        """
        
        contexts = session.execute(text(contexts_query)).fetchall()
        print("Available contexts for teams 2962 and 3031:")
        for context in contexts:
            print(f"  {context[0]}")
        
        # Check states for test teams
        states_query = """
            SELECT team_id, performance_state, state_date, home_away_context 
            FROM team_performance_states 
            WHERE team_id IN (2962, 3031) 
            ORDER BY team_id, state_date DESC
        """
        
        team_states = session.execute(text(states_query)).fetchall()
        print("\nStates for test teams (most recent first):")
        for state in team_states[:15]:  # Show first 15
            print(f"  Team {state[0]}: {state[1]} on {state[2]} ({state[3]})")
        
        # Check what contexts are used in debug script
        print("\nChecking specific contexts used in debug script:")
        for team_id in [2962, 3031]:
            for context in ['home', 'away', 'overall']:
                context_query = """
                    SELECT COUNT(*) 
                    FROM team_performance_states 
                    WHERE team_id = :team_id 
                      AND league_id = 224 
                      AND home_away_context = :context
                """
                
                count = session.execute(
                    text(context_query),
                    {'team_id': team_id, 'context': context}
                ).fetchone()[0]
                
                print(f"  Team {team_id}, context '{context}': {count} states")

if __name__ == "__main__":
    main()