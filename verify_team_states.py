#!/usr/bin/env python3
"""
Verify team_performance_states table data.
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
    """Verify team_performance_states table data."""
    try:
        # Load configuration first
        load_config()
        
        with get_db_session() as session:
            # Get basic stats
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    MIN(state_date) as earliest,
                    MAX(state_date) as latest,
                    COUNT(DISTINCT team_id) as unique_teams,
                    COUNT(DISTINCT league_id) as unique_leagues
                FROM team_performance_states
            """))
            
            row = result.fetchone()
            print(f'Total records: {row.total}')
            print(f'Date range: {row.earliest} to {row.latest}')
            print(f'Unique teams: {row.unique_teams}')
            print(f'Unique leagues: {row.unique_leagues}')
            
            # Get sample records
            print('\nSample records:')
            sample_result = session.execute(text("""
                SELECT 
                    team_id,
                    state_date,
                    home_away_context,
                    performance_state,
                    state_score,
                    matches_considered
                FROM team_performance_states
                ORDER BY state_date DESC
                LIMIT 5
            """))
            
            for row in sample_result.fetchall():
                print(f'Team {row.team_id}: {row.state_date} ({row.home_away_context}) - {row.performance_state} (score: {row.state_score}, matches: {row.matches_considered})')
            
            # Check performance state distribution
            print('\nPerformance state distribution:')
            dist_result = session.execute(text("""
                SELECT 
                    performance_state,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM team_performance_states), 2) as percentage
                FROM team_performance_states
                GROUP BY performance_state
                ORDER BY count DESC
            """))
            
            for row in dist_result.fetchall():
                print(f'{row.performance_state}: {row.count} ({row.percentage}%)')
                
    except Exception as e:
        print(f'Error verifying data: {e}')
        raise

if __name__ == '__main__':
    main()