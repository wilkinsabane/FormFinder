#!/usr/bin/env python3
"""
Check the team_performance_states table to see if it has any data.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check total records
        result = session.execute(text('SELECT COUNT(*) FROM team_performance_states')).fetchone()
        print(f'Total records in team_performance_states: {result[0]}')
        
        if result[0] > 0:
            # Show some sample records
            sample_query = text("""
                SELECT team_id, league_id, performance_state, state_score, state_date, home_away_context
                FROM team_performance_states
                ORDER BY state_date DESC
                LIMIT 10
            """)
            
            sample_results = session.execute(sample_query).fetchall()
            print("\nSample records:")
            print("Team ID | League ID | State | Score | Date | Context")
            print("-" * 60)
            for row in sample_results:
                print(f"{row[0]:7} | {row[1]:9} | {row[2]:7} | {row[3]:5.3f} | {row[4]} | {row[5]}")
        else:
            print("\nThe team_performance_states table is empty!")
            print("This explains why all states are showing as 'average'.")
            print("The state calculation logic is only triggered when no existing state is found.")

if __name__ == '__main__':
    main()