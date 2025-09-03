#!/usr/bin/env python3
"""
Debug script to check fixtures in the database for Markov feature generation.
"""

import sys
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime

def main():
    # Load configuration
    load_config()
    with get_db_session() as session:
        # Check total fixtures in league 228
        result = session.execute(text(
            "SELECT COUNT(*) FROM fixtures WHERE league_id = 228"
        ))
        total_fixtures = result.scalar()
        print(f"Total fixtures in league 228: {total_fixtures}")
        
        # Check fixtures with scores
        result = session.execute(text(
            "SELECT COUNT(*) FROM fixtures WHERE league_id = 228 AND home_score IS NOT NULL AND away_score IS NOT NULL"
        ))
        fixtures_with_scores = result.scalar()
        print(f"Fixtures with scores in league 228: {fixtures_with_scores}")
        
        # Check fixtures in date range 2024-01-01 to 2024-01-31
        result = session.execute(text(
            "SELECT COUNT(*) FROM fixtures WHERE match_date BETWEEN :start_date AND :end_date AND league_id = 228"
        ), {'start_date': '2024-01-01', 'end_date': '2024-01-31'})
        fixtures_in_range = result.scalar()
        print(f"Fixtures in Jan 2024 range: {fixtures_in_range}")
        
        # Check fixtures in date range with scores
        result = session.execute(text(
            "SELECT COUNT(*) FROM fixtures WHERE match_date BETWEEN :start_date AND :end_date AND home_score IS NOT NULL AND away_score IS NOT NULL AND league_id = 228"
        ), {'start_date': '2024-01-01', 'end_date': '2024-01-31'})
        fixtures_in_range_with_scores = result.scalar()
        print(f"Fixtures in Jan 2024 range with scores: {fixtures_in_range_with_scores}")
        
        # Check date range of fixtures in league 228
        result = session.execute(text(
            "SELECT MIN(match_date), MAX(match_date) FROM fixtures WHERE league_id = 228"
        ))
        date_range = result.fetchone()
        print(f"Date range for league 228: {date_range[0]} to {date_range[1]}")
        
        # Check some sample fixtures
        result = session.execute(text(
            "SELECT id, home_team_id, away_team_id, match_date, home_score, away_score FROM fixtures WHERE league_id = 228 AND home_score IS NOT NULL LIMIT 5"
        ))
        sample_fixtures = result.fetchall()
        print("\nSample fixtures with scores:")
        for fixture in sample_fixtures:
            print(f"  ID: {fixture[0]}, Teams: {fixture[1]} vs {fixture[2]}, Date: {fixture[3]}, Score: {fixture[4]}-{fixture[5]}")

if __name__ == "__main__":
    main()