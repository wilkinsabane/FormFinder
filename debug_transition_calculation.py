#!/usr/bin/env python3
"""
Debug script to test transition matrix calculation for a specific team.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.markov_transition_calculator import MarkovTransitionCalculator
from sqlalchemy import text

def main():
    """Test transition matrix calculation for a specific team."""
    load_config()
    
    with get_db_session() as session:
        # Get a team with many performance states
        result = session.execute(text("""
            SELECT team_id, COUNT(*) as state_count
            FROM team_performance_states 
            WHERE home_away_context = 'overall'
            GROUP BY team_id 
            ORDER BY state_count DESC 
            LIMIT 5
        """))
        
        teams = result.fetchall()
        print("Teams with most performance states:")
        for team_id, count in teams:
            print(f"  Team {team_id}: {count} states")
        
        if not teams:
            print("No teams found with performance states!")
            return
        
        # Test with the team that has the most states
        test_team_id = teams[0][0]
        print(f"\nTesting transition calculation for team {test_team_id}")
        
        # Get team's performance states
        states_result = session.execute(text("""
            SELECT performance_state, state_date, home_away_context
            FROM team_performance_states
            WHERE team_id = :team_id
            ORDER BY state_date ASC
            LIMIT 20
        """), {'team_id': test_team_id})
        
        states = states_result.fetchall()
        print(f"\nFirst 20 performance states for team {test_team_id}:")
        for state, date, context in states:
            print(f"  {date}: {state} ({context})")
        
        # Test transition calculator
        calculator = MarkovTransitionCalculator()
        
        # Test calculate_transition_matrix method directly
        print(f"\nTesting transition matrix calculation...")
        
        try:
            # Get league_id for this team
            league_result = session.execute(text("""
                SELECT DISTINCT league_id 
                FROM team_performance_states 
                WHERE team_id = :team_id 
                LIMIT 1
            """), {'team_id': test_team_id})
            
            league_id = league_result.fetchone()[0]
            print(f"Team {test_team_id} is in league {league_id}")
            
            # Calculate transition matrix
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            matrix = calculator.calculate_transition_matrix(
                team_id=test_team_id,
                league_id=league_id,
                start_date=start_date,
                end_date=end_date,
                context='overall'
            )
            
            print(f"\nTransition matrix result: {type(matrix)}")
            if matrix:
                print("Matrix contents:")
                for from_state, to_states in matrix.items():
                    print(f"  {from_state}: {to_states}")
            else:
                print("Matrix is None or empty!")
                
            # Test process_team_transitions
            print(f"\nTesting process_team_transitions...")
            results = calculator.process_team_transitions(
                team_id=test_team_id,
                league_id=league_id,
                contexts=['overall']
            )
            
            print(f"Process results: {results}")
            
        except Exception as e:
            print(f"Error during calculation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()