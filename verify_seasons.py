#!/usr/bin/env python3
"""Verify seasons in the database."""

from formfinder.database import get_db_session, League
from formfinder.config import load_config

def verify_seasons():
    """Check available seasons and count entries."""
    load_config()
    
    with get_db_session() as session:
        # Get all unique seasons
        seasons = [s[0] for s in session.query(League.season).distinct().order_by(League.season).all()]
        print('Available seasons:', seasons)
        
        # Count entries for 2022-2023
        count_2022 = session.query(League).filter(League.season == '2022-2023').count()
        print('2022-2023 entries:', count_2022)
        
        # Show sample entries for 2022-2023
        sample = session.query(League).filter(League.season == '2022-2023').limit(5).all()
        print('Sample 2022-2023 leagues:')
        for league in sample:
            print(f"  - {league.name} ({league.country}) - ID: {league.id}")

if __name__ == "__main__":
    verify_seasons()