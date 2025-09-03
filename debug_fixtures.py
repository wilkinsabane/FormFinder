#!/usr/bin/env python3
"""
Debug script to check fixture data availability
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime, timedelta

def main():
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Check total fixtures
        result = session.execute(text("SELECT COUNT(*) FROM fixtures"))
        total_fixtures = result.scalar()
        print(f"Total fixtures in database: {total_fixtures}")
        
        # Check fixtures with scores
        result = session.execute(text("""
            SELECT COUNT(*) FROM fixtures 
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """))
        finished_fixtures = result.scalar()
        print(f"Fixtures with scores: {finished_fixtures}")
        
        # Check recent fixtures (last 6 months)
        six_months_ago = datetime.now() - timedelta(days=180)
        result = session.execute(text("""
            SELECT COUNT(*) FROM fixtures 
            WHERE match_date >= :date AND home_score IS NOT NULL AND away_score IS NOT NULL
        """), {"date": six_months_ago})
        recent_fixtures = result.scalar()
        print(f"Recent fixtures (last 6 months) with scores: {recent_fixtures}")
        
        # Check sample of teams without performance states
        result = session.execute(text("""
            SELECT t.id, t.name, COUNT(f.id) as fixture_count
            FROM teams t
            LEFT JOIN team_performance_states tps ON t.id = tps.team_id
            LEFT JOIN fixtures f ON (t.id = f.home_team_id OR t.id = f.away_team_id)
                AND f.match_date >= :date AND f.home_score IS NOT NULL AND f.away_score IS NOT NULL
            WHERE tps.team_id IS NULL
            GROUP BY t.id, t.name
            ORDER BY fixture_count DESC
            LIMIT 10
        """), {"date": six_months_ago})
        
        print("\nSample teams without performance states and their fixture counts:")
        for row in result.fetchall():
            print(f"Team: {row[1]} (ID: {row[0]}) - Fixtures: {row[2]}")
        
        # Check date range of fixtures
        result = session.execute(text("""
            SELECT MIN(match_date) as earliest, MAX(match_date) as latest
            FROM fixtures
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """))
        date_range = result.fetchone()
        print(f"\nFixture date range: {date_range[0]} to {date_range[1]}")

if __name__ == "__main__":
    main()