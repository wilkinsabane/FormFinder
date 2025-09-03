#!/usr/bin/env python3
"""
Check which teams have performance state data.
"""

import sys
from pathlib import Path
from sqlalchemy import text

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session
from formfinder.config import load_config

def main():
    """Check available teams with performance state data."""
    try:
        # Load configuration first
        load_config()
        
        with get_db_session() as session:
            # Get teams with most performance state records
            result = session.execute(text("""
                SELECT 
                    team_id,
                    COUNT(*) as state_count,
                    MIN(state_date) as earliest_date,
                    MAX(state_date) as latest_date,
                    COUNT(DISTINCT home_away_context) as contexts
                FROM team_performance_states
                GROUP BY team_id
                HAVING COUNT(*) >= 5
                ORDER BY state_count DESC
                LIMIT 10
            """))
            
            print('Teams with most performance state data:')
            teams_with_data = []
            for row in result.fetchall():
                print(f'Team {row.team_id}: {row.state_count} states, {row.contexts} contexts, {row.earliest_date} to {row.latest_date}')
                teams_with_data.append(row.team_id)
            
            if len(teams_with_data) >= 2:
                print(f'\nRecommended test teams: {teams_with_data[0]} and {teams_with_data[1]}')
                
                # Check if these teams have recent states
                for team_id in teams_with_data[:2]:
                    recent_result = session.execute(text("""
                        SELECT state_date, home_away_context, performance_state, state_score
                        FROM team_performance_states
                        WHERE team_id = :team_id
                        ORDER BY state_date DESC
                        LIMIT 3
                    """), {'team_id': team_id})
                    
                    print(f'\nRecent states for team {team_id}:')
                    for state_row in recent_result.fetchall():
                        print(f'  {state_row.state_date} ({state_row.home_away_context}): {state_row.performance_state} (score: {state_row.state_score})')
                        
    except Exception as e:
        print(f'Error checking teams: {e}')
        raise

if __name__ == '__main__':
    main()