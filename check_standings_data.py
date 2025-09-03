#!/usr/bin/env python3
"""Check standings data availability."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check if standings table exists and has data
        standings_count = session.execute(text(
            "SELECT COUNT(*) FROM standings"
        )).fetchone()[0]
        
        print(f"Total standings records: {standings_count}")
        
        if standings_count > 0:
            # Check sample standings data
            sample_standings = session.execute(text(
                "SELECT team_id, league_id, position, updated_at FROM standings ORDER BY updated_at DESC LIMIT 10"
            )).fetchall()
            
            print("\nSample standings data:")
            for row in sample_standings:
                print(f"Team {row[0]}, League {row[1]}, Position {row[2]}, Updated: {row[3]}")
            
            # Check unique positions
            unique_positions = session.execute(text(
                "SELECT DISTINCT position FROM standings ORDER BY position"
            )).fetchall()
            
            print(f"\nUnique positions in standings: {[row[0] for row in unique_positions]}")
            
            # Check for a specific fixture's teams
            fixture_teams = session.execute(text(
                "SELECT f.id, f.home_team_id, f.away_team_id, f.league_id, f.match_date FROM fixtures f WHERE f.id IN (SELECT fixture_id FROM pre_computed_features LIMIT 1)"
            )).fetchone()
            
            if fixture_teams:
                print(f"\nChecking standings for fixture {fixture_teams[0]}:")
                print(f"Home team {fixture_teams[1]}, Away team {fixture_teams[2]}, League {fixture_teams[3]}, Date {fixture_teams[4]}")
                
                # Check standings for these specific teams
                for team_type, team_id in [('Home', fixture_teams[1]), ('Away', fixture_teams[2])]:
                    team_standings = session.execute(text(
                        "SELECT position, updated_at FROM standings WHERE team_id = :team_id AND league_id = :league_id ORDER BY updated_at DESC LIMIT 5"
                    ), {'team_id': team_id, 'league_id': fixture_teams[3]}).fetchall()
                    
                    print(f"\n{team_type} team {team_id} standings:")
                    if team_standings:
                        for pos, updated in team_standings:
                            print(f"  Position {pos}, Updated: {updated}")
                    else:
                        print(f"  No standings data found for team {team_id} in league {fixture_teams[3]}")
        else:
            print("No standings data found in database!")
            
            # Check if standings table exists
            table_exists = session.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'standings')"
            )).fetchone()[0]
            
            print(f"Standings table exists: {table_exists}")

if __name__ == '__main__':
    main()