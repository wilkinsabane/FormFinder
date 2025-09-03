#!/usr/bin/env python3

import sys
sys.path.append('.')

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    # Load configuration first
    load_config()
    with get_db_session() as session:
        # Get fixtures with complete data
        query = """
            SELECT f.id, f.home_team_id, f.away_team_id, f.match_date, f.league_id,
                   ht.name as home_team, at.name as away_team
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.match_date < NOW() 
            ORDER BY f.match_date DESC 
            LIMIT 10
        """
        
        result = session.execute(text(query))
        fixtures = result.fetchall()
        
        print("Available fixtures with complete data:")
        for fixture in fixtures:
            print(f"ID: {fixture[0]}, {fixture[5]} vs {fixture[6]}, Date: {fixture[3]}, League: {fixture[4]}")
        
        # Also check if we have any pre-computed features
        count_query = "SELECT COUNT(*) FROM pre_computed_features"
        count_result = session.execute(text(count_query))
        count = count_result.fetchone()[0]
        print(f"\nCurrent pre-computed features count: {count}")
        
        # Check a specific fixture's pre-computed features
        if fixtures:
            fixture_id = fixtures[0][0]
            feature_query = "SELECT home_team_position, away_team_position FROM pre_computed_features WHERE fixture_id = :fixture_id"
            feature_result = session.execute(text(feature_query), {"fixture_id": fixture_id})
            feature_row = feature_result.fetchone()
            if feature_row:
                print(f"Fixture {fixture_id} positions: home={feature_row[0]}, away={feature_row[1]}")
            else:
                print(f"No pre-computed features found for fixture {fixture_id}")
                print(f"Let's try to process fixture {fixture_id} to test our fix...")
                return fixture_id

if __name__ == "__main__":
    main()