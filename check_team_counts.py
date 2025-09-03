from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_team_counts():
    load_config()
    
    with get_db_session() as session:
        # Count teams with performance states
        result = session.execute(text("""
            SELECT COUNT(DISTINCT team_id) as total_teams 
            FROM team_performance_states
        """))
        teams_with_states = result.fetchone()[0]
        print(f"Teams with performance states: {teams_with_states}")
        
        # Count total unique teams in fixtures
        result = session.execute(text("""
            SELECT COUNT(DISTINCT team_id) as total_teams FROM (
                SELECT home_team_id as team_id FROM fixtures
                UNION
                SELECT away_team_id as team_id FROM fixtures
            ) as all_teams
        """))
        total_teams = result.fetchone()[0]
        print(f"Total unique teams in fixtures: {total_teams}")
        
        # Check teams without performance states
        result = session.execute(text("""
            SELECT COUNT(*) as missing_teams FROM (
                SELECT home_team_id as team_id FROM fixtures
                UNION
                SELECT away_team_id as team_id FROM fixtures
            ) as all_teams
            WHERE team_id NOT IN (
                SELECT DISTINCT team_id FROM team_performance_states
            )
        """))
        missing_teams = result.fetchone()[0]
        print(f"Teams missing performance states: {missing_teams}")
        
        # Show some teams without states
        if missing_teams > 0:
            result = session.execute(text("""
                SELECT DISTINCT team_id
                FROM (
                    SELECT home_team_id as team_id FROM fixtures
                    UNION
                    SELECT away_team_id as team_id FROM fixtures
                ) as all_teams
                WHERE team_id NOT IN (
                    SELECT DISTINCT team_id FROM team_performance_states
                )
                LIMIT 10
            """))
            
            print("\nSample team IDs without performance states:")
            for row in result:
                print(f"- Team ID: {row[0]}")

if __name__ == "__main__":
    check_team_counts()