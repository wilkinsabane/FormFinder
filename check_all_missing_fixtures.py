#!/usr/bin/env python3
"""
Check all fixtures that need processing regardless of date.
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
        # Check for all fixtures that need processing
        missing_fixtures_query = text("""
            SELECT f.id, f.league_id, f.home_team_id, f.away_team_id, f.match_date,
                   f.home_score, f.away_score
            FROM fixtures f
            WHERE f.status = 'finished'
              AND f.home_score IS NOT NULL
              AND f.away_score IS NOT NULL
              AND f.id NOT IN (SELECT fixture_id FROM pre_computed_features)
            ORDER BY f.match_date DESC
            LIMIT 20
        """)
        
        missing_fixtures = session.execute(missing_fixtures_query).fetchall()
        
        print(f"Total fixtures needing processing: {len(missing_fixtures)}")
        print("\nAll missing fixtures:")
        for fixture in missing_fixtures:
            print(f"  Fixture {fixture.id}: League {fixture.league_id}, "
                  f"Teams {fixture.home_team_id} vs {fixture.away_team_id}, "
                  f"Date {fixture.match_date}, Score {fixture.home_score}-{fixture.away_score}")
        
        # Check what's in pre_computed_features table
        pcf_count_query = text("SELECT COUNT(*) FROM pre_computed_features")
        pcf_count = session.execute(pcf_count_query).scalar()
        
        # Check total finished fixtures with scores
        total_fixtures_query = text("""
            SELECT COUNT(*) FROM fixtures
            WHERE status = 'finished'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
        """)
        total_fixtures = session.execute(total_fixtures_query).scalar()
        
        print(f"\nDatabase status:")
        print(f"  Total finished fixtures with scores: {total_fixtures}")
        print(f"  Pre-computed features records: {pcf_count}")
        print(f"  Missing features: {total_fixtures - pcf_count}")
        
        # Check a sample of existing pre_computed_features
        sample_pcf_query = text("""
            SELECT fixture_id, home_xg, away_xg, home_team_strength, away_team_strength,
                   home_momentum, away_momentum, home_sentiment, away_sentiment
            FROM pre_computed_features
            ORDER BY fixture_id DESC
            LIMIT 5
        """)
        
        sample_pcf = session.execute(sample_pcf_query).fetchall()
        
        print(f"\nSample pre-computed features:")
        for pcf in sample_pcf:
            print(f"  Fixture {pcf.fixture_id}: xG({pcf.home_xg:.2f}, {pcf.away_xg:.2f}), "
                  f"Strength({pcf.home_team_strength:.2f}, {pcf.away_team_strength:.2f}), "
                  f"Momentum({pcf.home_momentum:.2f}, {pcf.away_momentum:.2f}), "
                  f"Sentiment({pcf.home_sentiment:.2f}, {pcf.away_sentiment:.2f})")

if __name__ == "__main__":
    main()