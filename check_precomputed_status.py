#!/usr/bin/env python3
"""
Check the current status of pre_computed_features table.
"""

import sys
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Check total records in pre_computed_features
        result = session.execute(text(
            "SELECT COUNT(*) FROM pre_computed_features"
        ))
        total_records = result.scalar()
        print(f"Total records in pre_computed_features: {total_records}")
        
        # Check records with non-null values in key columns
        result = session.execute(text(
            "SELECT COUNT(*) FROM pre_computed_features WHERE home_xg IS NOT NULL"
        ))
        home_xg_records = result.scalar()
        print(f"Records with home_xg: {home_xg_records}")
        
        result = session.execute(text(
            "SELECT COUNT(*) FROM pre_computed_features WHERE home_team_strength IS NOT NULL"
        ))
        team_strength_records = result.scalar()
        print(f"Records with home_team_strength: {team_strength_records}")
        
        result = session.execute(text(
            "SELECT COUNT(*) FROM pre_computed_features WHERE home_team_momentum IS NOT NULL"
        ))
        momentum_records = result.scalar()
        print(f"Records with home_team_momentum: {momentum_records}")
        
        result = session.execute(text(
            "SELECT COUNT(*) FROM pre_computed_features WHERE home_team_sentiment IS NOT NULL"
        ))
        sentiment_records = result.scalar()
        print(f"Records with home_team_sentiment: {sentiment_records}")
        
        # Check for Markov features
        result = session.execute(text(
            "SELECT COUNT(*) FROM pre_computed_features WHERE home_team_markov_momentum IS NOT NULL"
        ))
        markov_records = result.scalar()
        print(f"Records with home_team_markov_momentum: {markov_records}")
        
        # Check recent records
        result = session.execute(text(
            "SELECT fixture_id, home_xg, home_team_strength, home_team_momentum, home_team_sentiment, home_team_markov_momentum FROM pre_computed_features ORDER BY created_at DESC LIMIT 5"
        ))
        recent_records = result.fetchall()
        print("\nRecent records:")
        for record in recent_records:
            print(f"  Fixture {record[0]}: xG={record[1]}, strength={record[2]}, momentum={record[3]}, sentiment={record[4]}, markov={record[5]}")
        
        # Check if we have any markov_features records
        result = session.execute(text(
            "SELECT COUNT(*) FROM markov_features"
        ))
        markov_feature_count = result.scalar()
        print(f"\nTotal records in markov_features table: {markov_feature_count}")

if __name__ == "__main__":
    main()