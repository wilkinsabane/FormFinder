#!/usr/bin/env python3
"""
Check records with computed features.
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
        # Check records with computed features
        result = session.execute(text(
            "SELECT fixture_id, home_xg, home_team_strength, home_team_momentum, home_team_sentiment, home_team_markov_momentum FROM pre_computed_features WHERE home_xg IS NOT NULL ORDER BY created_at DESC LIMIT 5"
        ))
        records = result.fetchall()
        
        print("Records with computed features:")
        for record in records:
            print(f"  Fixture {record[0]}: xG={record[1]}, strength={record[2]}, momentum={record[3]}, sentiment={record[4]}, markov={record[5]}")
        
        # Check if there are any null values in the computed records
        result = session.execute(text(
            "SELECT COUNT(*) FROM pre_computed_features WHERE home_xg IS NOT NULL AND (home_team_strength IS NULL OR home_team_momentum IS NULL OR home_team_sentiment IS NULL)"
        ))
        null_count = result.scalar()
        print(f"\nRecords with xG but missing other features: {null_count}")
        
        # Check the distribution of feature values
        result = session.execute(text(
            "SELECT AVG(home_xg), AVG(home_team_strength), AVG(home_team_momentum), AVG(home_team_sentiment) FROM pre_computed_features WHERE home_xg IS NOT NULL"
        ))
        averages = result.fetchone()
        print(f"\nAverage feature values:")
        print(f"  Average home_xg: {averages[0]:.3f}")
        print(f"  Average home_team_strength: {averages[1]:.3f}")
        print(f"  Average home_team_momentum: {averages[2]:.3f}")
        print(f"  Average home_team_sentiment: {averages[3]:.3f}")

if __name__ == "__main__":
    main()