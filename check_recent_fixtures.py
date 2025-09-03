#!/usr/bin/env python3
"""
Check for recent fixtures that need processing to find ones with sufficient historical data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
from datetime import datetime

def main():
    # Load configuration first
    load_config()
    
    with get_db_session() as session:
        # Check for fixtures after 2021-01-01 that need processing
        recent_fixtures_query = text("""
            SELECT f.id, f.league_id, f.home_team_id, f.away_team_id, f.match_date,
                   f.home_score, f.away_score
            FROM fixtures f
            WHERE f.status = 'finished'
              AND f.home_score IS NOT NULL
              AND f.away_score IS NOT NULL
              AND f.match_date >= '2021-01-01'
              AND f.id NOT IN (SELECT fixture_id FROM pre_computed_features)
            ORDER BY f.match_date DESC
            LIMIT 10
        """)
        
        recent_fixtures = session.execute(recent_fixtures_query).fetchall()
        
        print(f"Recent fixtures (after 2021-01-01) needing processing: {len(recent_fixtures)}")
        print("\nSample recent fixtures:")
        for fixture in recent_fixtures:
            print(f"  Fixture {fixture.id}: League {fixture.league_id}, "
                  f"Teams {fixture.home_team_id} vs {fixture.away_team_id}, "
                  f"Date {fixture.match_date}, Score {fixture.home_score}-{fixture.away_score}")
        
        # For the first recent fixture, check historical data availability
        if recent_fixtures:
            first_fixture = recent_fixtures[0]
            print(f"\nChecking historical data for fixture {first_fixture.id}:")
            
            # Check historical fixtures for home team
            home_history_query = text("""
                SELECT COUNT(*) as count
                FROM fixtures
                WHERE (home_team_id = :team_id OR away_team_id = :team_id)
                  AND status = 'finished'
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                  AND match_date < :match_date
            """)
            
            home_history = session.execute(home_history_query, {
                'team_id': first_fixture.home_team_id,
                'match_date': first_fixture.match_date
            }).scalar()
            
            away_history = session.execute(home_history_query, {
                'team_id': first_fixture.away_team_id,
                'match_date': first_fixture.match_date
            }).scalar()
            
            print(f"  Home team {first_fixture.home_team_id}: {home_history} historical fixtures")
            print(f"  Away team {first_fixture.away_team_id}: {away_history} historical fixtures")
            
            # Check league historical data
            league_history_query = text("""
                SELECT COUNT(*) as count
                FROM fixtures
                WHERE league_id = :league_id
                  AND status = 'finished'
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                  AND match_date < :match_date
            """)
            
            league_history = session.execute(league_history_query, {
                'league_id': first_fixture.league_id,
                'match_date': first_fixture.match_date
            }).scalar()
            
            print(f"  League {first_fixture.league_id}: {league_history} historical fixtures")

if __name__ == "__main__":
    main()