#!/usr/bin/env python3
"""
Check team coverage in team_performance_states table
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Get total teams
        result = session.execute(text('SELECT COUNT(*) FROM teams'))
        total_teams = result.scalar()
        
        # Get teams with performance states
        result = session.execute(text('SELECT COUNT(DISTINCT team_id) FROM team_performance_states'))
        covered_teams = result.scalar()
        
        # Calculate remaining
        remaining_teams = total_teams - covered_teams
        
        print(f"Total teams in database: {total_teams}")
        print(f"Teams with performance states: {covered_teams}")
        print(f"Remaining teams to populate: {remaining_teams}")
        print(f"Coverage percentage: {(covered_teams / total_teams * 100):.1f}%")
        
        # Show some teams without performance states
        if remaining_teams > 0:
            print("\nSample teams without performance states:")
            result = session.execute(text("""
                SELECT t.id, t.name, l.name as league_name
                FROM teams t
                JOIN leagues l ON t.league_id = l.league_pk
                WHERE t.id NOT IN (SELECT DISTINCT team_id FROM team_performance_states)
                LIMIT 10
            """))
            
            uncovered_teams = result.fetchall()
            for team in uncovered_teams:
                print(f"  Team ID {team[0]}: {team[1]} (League: {team[2]})")

if __name__ == "__main__":
    main()