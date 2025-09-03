#!/usr/bin/env python3
"""
Check standings data coverage across leagues.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_standings_coverage():
    """Check which leagues have standings data and which don't."""
    load_config()
    
    with get_db_session() as session:
        print("=== Standings Coverage Analysis ===")
        
        # Get total leagues
        total_leagues = session.execute(text("SELECT COUNT(*) FROM leagues")).scalar()
        print(f"Total leagues: {total_leagues}")
        
        # Get leagues with standings data
        leagues_with_standings = session.execute(text("""
            SELECT COUNT(DISTINCT league_id) FROM standings
        """)).scalar()
        print(f"Leagues with standings data: {leagues_with_standings}")
        print(f"Leagues without standings: {total_leagues - leagues_with_standings}")
        
        # Show leagues without standings data
        print("\n=== Leagues WITHOUT standings data ===")
        leagues_without_standings = session.execute(text("""
            SELECT l.id, l.name, l.season, COUNT(t.id) as team_count
            FROM leagues l
            LEFT JOIN teams t ON l.id = t.league_id
            WHERE l.id NOT IN (SELECT DISTINCT league_id FROM standings)
            GROUP BY l.id, l.name, l.season
            ORDER BY team_count DESC
            LIMIT 10
        """)).fetchall()
        
        for row in leagues_without_standings:
            print(f"League {row[0]}: {row[1]} ({row[2]}) - {row[3]} teams")
        
        # Show leagues with standings data
        print("\n=== Leagues WITH standings data ===")
        leagues_with_standings_detail = session.execute(text("""
            SELECT l.id, l.name, l.season, COUNT(DISTINCT s.team_id) as teams_with_standings
            FROM leagues l
            JOIN standings s ON l.id = s.league_id
            GROUP BY l.id, l.name, l.season
            ORDER BY teams_with_standings DESC
            LIMIT 10
        """)).fetchall()
        
        for row in leagues_with_standings_detail:
            print(f"League {row[0]}: {row[1]} ({row[2]}) - {row[3]} teams with standings")
        
        # Check specific league 311
        print("\n=== League 311 Analysis ===")
        league_311_info = session.execute(text("""
            SELECT l.name, l.season, COUNT(t.id) as total_teams
            FROM leagues l
            LEFT JOIN teams t ON l.id = t.league_id
            WHERE l.id = 311
            GROUP BY l.name, l.season
        """)).fetchone()
        
        if league_311_info:
            print(f"League 311: {league_311_info[0]} ({league_311_info[1]}) - {league_311_info[2]} teams")
            
            # Check if any teams in league 311 have standings
            standings_311 = session.execute(text("""
                SELECT COUNT(*) FROM standings WHERE league_id = 311
            """)).scalar()
            print(f"Standings records for League 311: {standings_311}")
            
            # Check fixtures in league 311
            fixtures_311 = session.execute(text("""
                SELECT COUNT(*) FROM fixtures WHERE league_id = 311
            """)).scalar()
            print(f"Fixtures in League 311: {fixtures_311}")
        
        # Check if we need to fetch standings for these leagues
        print("\n=== Recommendation ===")
        if leagues_without_standings:
            print("Some leagues are missing standings data. Consider:")
            print("1. Running standings fetch for these leagues")
            print("2. Updating the _get_team_position method to handle missing data gracefully")
            print("3. Using a more robust fallback strategy")

if __name__ == "__main__":
    check_standings_coverage()