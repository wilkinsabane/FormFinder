#!/usr/bin/env python3
"""
Check fixtures with non-zero enhanced features.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_nonzero_features():
    """Check fixtures with non-zero enhanced features."""
    load_config()
    
    with get_db_session() as session:
        # Get fixtures with non-zero enhanced features
        result = session.execute(text("""
            SELECT fixture_id, home_team_momentum, away_team_momentum, 
                   home_team_sentiment, away_team_sentiment
            FROM pre_computed_features 
            WHERE home_team_momentum > 0 OR away_team_momentum > 0 
               OR home_team_sentiment != 0 OR away_team_sentiment != 0
            LIMIT 20
        """))
        
        print("Fixtures with non-zero enhanced features:")
        count = 0
        for row in result:
            count += 1
            print(f"Fixture {row[0]}: home_momentum={row[1]}, away_momentum={row[2]}, home_sentiment={row[3]}, away_sentiment={row[4]}")
        
        if count == 0:
            print("No fixtures found with non-zero enhanced features")
        else:
            print(f"\nFound {count} fixtures with non-zero enhanced features")

if __name__ == "__main__":
    check_nonzero_features()