#!/usr/bin/env python3
"""
Check which fixtures have successful feature extraction (non-null values).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    # Load configuration first
    load_config()
    
    with get_db_session() as session:
        # Check fixtures with successful feature extraction (non-null values)
        successful_query = text("""
            SELECT pcf.fixture_id, f.match_date, f.league_id, f.home_team_id, f.away_team_id,
                   pcf.home_xg, pcf.away_xg, pcf.home_team_strength, pcf.away_team_strength,
                   pcf.home_team_momentum, pcf.away_team_momentum
            FROM pre_computed_features pcf
            JOIN fixtures f ON pcf.fixture_id = f.id
            WHERE pcf.home_xg IS NOT NULL
              AND pcf.home_team_strength IS NOT NULL
              AND pcf.home_team_momentum IS NOT NULL
            ORDER BY f.match_date DESC
            LIMIT 10
        """)
        
        successful_fixtures = session.execute(successful_query).fetchall()
        
        print(f"Fixtures with successful feature extraction: {len(successful_fixtures)}")
        print("\nSample successful fixtures:")
        for fixture in successful_fixtures:
            print(f"  Fixture {fixture.fixture_id}: Date {fixture.match_date}, "
                  f"League {fixture.league_id}, Teams {fixture.home_team_id} vs {fixture.away_team_id}")
            print(f"    xG: {fixture.home_xg:.3f} vs {fixture.away_xg:.3f}, "
                  f"Strength: {fixture.home_team_strength:.3f} vs {fixture.away_team_strength:.3f}, "
                  f"Momentum: {fixture.home_team_momentum:.3f} vs {fixture.away_team_momentum:.3f}")
        
        # Check the date range of successful fixtures
        date_range_query = text("""
            SELECT MIN(f.match_date) as earliest, MAX(f.match_date) as latest, COUNT(*) as count
            FROM pre_computed_features pcf
            JOIN fixtures f ON pcf.fixture_id = f.id
            WHERE pcf.home_xg IS NOT NULL
              AND pcf.home_team_strength IS NOT NULL
              AND pcf.home_team_momentum IS NOT NULL
        """)
        
        date_range = session.execute(date_range_query).fetchone()
        
        print(f"\nSuccessful fixtures date range:")
        print(f"  Earliest: {date_range.earliest}")
        print(f"  Latest: {date_range.latest}")
        print(f"  Total count: {date_range.count}")
        
        # Check if there are any fixtures with partial features (some null, some not)
        partial_query = text("""
            SELECT COUNT(*) as count
            FROM pre_computed_features
            WHERE (home_xg IS NULL AND home_team_strength IS NOT NULL)
               OR (home_xg IS NOT NULL AND home_team_strength IS NULL)
               OR (home_team_strength IS NULL AND home_team_momentum IS NOT NULL)
               OR (home_team_strength IS NOT NULL AND home_team_momentum IS NULL)
        """)
        
        partial_count = session.execute(partial_query).scalar()
        print(f"\nFixtures with partial features: {partial_count}")
        
        # Check the most recent fixtures that failed
        failed_recent_query = text("""
            SELECT pcf.fixture_id, f.match_date, f.league_id, f.home_team_id, f.away_team_id
            FROM pre_computed_features pcf
            JOIN fixtures f ON pcf.fixture_id = f.id
            WHERE pcf.home_xg IS NULL
            ORDER BY f.match_date DESC
            LIMIT 5
        """)
        
        failed_fixtures = session.execute(failed_recent_query).fetchall()
        
        print(f"\nMost recent fixtures with null features:")
        for fixture in failed_fixtures:
            print(f"  Fixture {fixture.fixture_id}: Date {fixture.match_date}, "
                  f"League {fixture.league_id}, Teams {fixture.home_team_id} vs {fixture.away_team_id}")

if __name__ == "__main__":
    main()