#!/usr/bin/env python3
"""
Test the fixture query for a specific team
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    # Load configuration
    load_config()
    
    # Test with Chesterfield (ID: 5977) which should have 17 fixtures
    team_id = 5977
    
    with get_db_session() as session:
        # Test the exact query from populate_remaining_teams.py
        result = session.execute(text("""
            SELECT f.id, f.home_team_id, f.away_team_id, f.home_score, f.away_score, f.match_date
            FROM fixtures f
            WHERE (f.home_team_id = :team_id OR f.away_team_id = :team_id)
                 AND f.home_score IS NOT NULL
                 AND f.away_score IS NOT NULL
                 AND f.match_date >= CURRENT_DATE - INTERVAL '180 days'
            ORDER BY f.match_date DESC
            LIMIT 10
        """), {"team_id": team_id})
        
        fixtures = result.fetchall()
        print(f"Found {len(fixtures)} fixtures for team {team_id}")
        
        for fixture in fixtures:
            print(f"Fixture {fixture[0]}: {fixture[1]} vs {fixture[2]} ({fixture[3]}-{fixture[4]}) on {fixture[5]}")
        
        # Also test without the date restriction
        result = session.execute(text("""
            SELECT f.id, f.home_team_id, f.away_team_id, f.home_score, f.away_score, f.match_date
            FROM fixtures f
            WHERE (f.home_team_id = :team_id OR f.away_team_id = :team_id)
                 AND f.home_score IS NOT NULL
                 AND f.away_score IS NOT NULL
            ORDER BY f.match_date DESC
            LIMIT 10
        """), {"team_id": team_id})
        
        all_fixtures = result.fetchall()
        print(f"\nFound {len(all_fixtures)} fixtures for team {team_id} (no date restriction)")
        
        if all_fixtures:
            latest = all_fixtures[0]
            print(f"Latest fixture: {latest[5]}")

if __name__ == "__main__":
    main()