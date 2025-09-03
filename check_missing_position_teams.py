#!/usr/bin/env python3
"""
Check why teams 3374 and 5159 don't have position data available.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_missing_teams():
    """Check teams 3374 and 5159 for position data issues."""
    load_config()
    
    team_ids = [3374, 5159]
    
    with get_db_session() as session:
        for team_id in team_ids:
            print(f"\n=== Checking Team {team_id} ===")
            
            # Check if team exists in teams table
            team_query = text("""
                SELECT id, name, league_id
                FROM teams
                WHERE id = :team_id
            """)
            
            team_result = session.execute(team_query, {'team_id': team_id}).fetchone()
            if team_result:
                print(f"Team found: {team_result[1]} (League: {team_result[2]})")
                league_id = team_result[2]
            else:
                print(f"Team {team_id} not found in teams table")
                continue
            
            # Check standings data for this team
            standings_query = text("""
                SELECT team_id, league_id, position, season, updated_at
                FROM standings
                WHERE team_id = :team_id
                ORDER BY updated_at DESC
                LIMIT 5
            """)
            
            standings_result = session.execute(standings_query, {'team_id': team_id}).fetchall()
            if standings_result:
                print(f"Standings data found ({len(standings_result)} records):")
                for row in standings_result:
                    print(f"  Position: {row[2]}, Season: {row[3]}, Updated: {row[4]}")
            else:
                print("No standings data found for this team")
            
            # Check what season the league is currently in
            league_query = text("""
                SELECT id, season, name
                FROM leagues
                WHERE id = :league_id
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            league_result = session.execute(league_query, {'league_id': league_id}).fetchone()
            if league_result:
                current_season = league_result[1]
                print(f"League {league_id} current season: {current_season}")
                
                # Check if team has standings for current season
                season_standings_query = text("""
                    SELECT position, updated_at
                    FROM standings
                    WHERE team_id = :team_id
                        AND league_id = :league_id
                        AND season = :season
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                
                season_result = session.execute(season_standings_query, {
                    'team_id': team_id,
                    'league_id': league_id,
                    'season': current_season
                }).fetchone()
                
                if season_result:
                    print(f"Current season standings: Position {season_result[0]}, Updated: {season_result[1]}")
                else:
                    print(f"No standings data for current season {current_season}")
            else:
                print(f"League {league_id} not found")
            
            # Check recent fixtures for this team
            fixtures_query = text("""
                SELECT f.id, f.match_date, f.league_id,
                       CASE WHEN f.home_team_id = :team_id THEN 'home' ELSE 'away' END as venue
                FROM fixtures f
                WHERE (f.home_team_id = :team_id OR f.away_team_id = :team_id)
                ORDER BY f.match_date DESC
                LIMIT 3
            """)
            
            fixtures_result = session.execute(fixtures_query, {'team_id': team_id}).fetchall()
            if fixtures_result:
                print(f"Recent fixtures ({len(fixtures_result)} found):")
                for row in fixtures_result:
                    print(f"  Fixture {row[0]}: {row[1]} (League: {row[2]}, Venue: {row[3]})")
            else:
                print("No fixtures found for this team")

if __name__ == "__main__":
    check_missing_teams()