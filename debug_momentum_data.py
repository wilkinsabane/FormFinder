#!/usr/bin/env python3
"""
Debug momentum data availability
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.database import get_db_session
from sqlalchemy import text

def debug_momentum_data():
    """Debug momentum data availability"""
    
    load_config()
    
    with get_db_session() as session:
        # Get our test fixture details
        result = session.execute(text("""
            SELECT id, home_team_id, away_team_id, match_date, league_id 
            FROM fixtures 
            WHERE home_score IS NOT NULL 
            AND away_score IS NOT NULL 
            AND status = 'finished'
            ORDER BY match_date DESC 
            LIMIT 1
        """))
        
        fixture = result.fetchone()
        print(f"Test fixture: {fixture.id}")
        print(f"  Home Team: {fixture.home_team_id}")
        print(f"  Away Team: {fixture.away_team_id}")
        print(f"  Match Date: {fixture.match_date}")
        print(f"  League: {fixture.league_id}")
        
        # Check if these teams have any performance states
        for team_id in [fixture.home_team_id, fixture.away_team_id]:
            result = session.execute(text("""
                SELECT COUNT(*), MIN(state_date), MAX(state_date)
                FROM team_performance_states
                WHERE team_id = :team_id
            """), {'team_id': team_id})
            
            count, min_date, max_date = result.fetchone()
            print(f"\nTeam {team_id} performance states:")
            print(f"  Count: {count}")
            print(f"  Date range: {min_date} to {max_date}")
            
            # Check for this specific league
            result = session.execute(text("""
                SELECT COUNT(*)
                FROM team_performance_states
                WHERE team_id = :team_id AND league_id = :league_id
            """), {'team_id': team_id, 'league_id': fixture.league_id})
            
            league_count = result.scalar()
            print(f"  In league {fixture.league_id}: {league_count}")
        
        # Check what teams DO have performance states
        result = session.execute(text("""
            SELECT DISTINCT team_id, league_id, COUNT(*) as state_count
            FROM team_performance_states
            GROUP BY team_id, league_id
            ORDER BY state_count DESC
            LIMIT 10
        """))
        
        print("\nTeams with most performance states:")
        for row in result.fetchall():
            print(f"  Team {row[0]} in League {row[1]}: {row[2]} states")

if __name__ == "__main__":
    debug_momentum_data()