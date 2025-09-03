#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

# Load configuration
load_config()

def check_team_fixtures(team_id):
    """Check fixtures for a specific team."""
    with get_db_session() as session:
        # Count all fixtures
        result = session.execute(text("""
            SELECT COUNT(*) FROM fixtures 
            WHERE home_team_id = :team_id OR away_team_id = :team_id
        """), {"team_id": team_id})
        total_fixtures = result.scalar()
        print(f"Total fixtures for team {team_id}: {total_fixtures}")
        
        # Count finished fixtures with scores
        result = session.execute(text("""
            SELECT COUNT(*) FROM fixtures 
            WHERE (home_team_id = :team_id OR away_team_id = :team_id)
                AND status = 'FINISHED'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
        """), {"team_id": team_id})
        finished_fixtures = result.scalar()
        print(f"Finished fixtures with scores for team {team_id}: {finished_fixtures}")
        
        # Sample fixtures
        result = session.execute(text("""
            SELECT f.id, f.match_date, f.status, f.home_score, f.away_score,
                   ht.name as home_team, at.name as away_team
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE (f.home_team_id = :team_id OR f.away_team_id = :team_id)
            ORDER BY f.match_date DESC
            LIMIT 5
        """), {"team_id": team_id})
        
        print(f"\nSample fixtures for team {team_id}:")
        for row in result:
            print(f"  ID: {row.id}, Date: {row.match_date}, Status: {row.status}, Score: {row.home_score}-{row.away_score}, {row.home_team} vs {row.away_team}")

def check_fixture_statuses():
    """Check what fixture statuses exist in the database."""
    with get_db_session() as session:
        result = session.execute(text("SELECT DISTINCT status FROM fixtures ORDER BY status"))
        print("Distinct fixture statuses:")
        for row in result:
            print(f"  {row[0]}")
        
        # Count by status
        result = session.execute(text("SELECT status, COUNT(*) FROM fixtures GROUP BY status ORDER BY status"))
        print("\nFixture counts by status:")
        for row in result:
            print(f"  {row[0]}: {row[1]}")

if __name__ == "__main__":
    # Check fixture statuses first
    print("Checking fixture statuses...")
    check_fixture_statuses()
    
    # Check a few sample teams
    sample_teams = [5060, 6965, 12012]
    for team_id in sample_teams:
        print(f"\n{'='*50}")
        check_team_fixtures(team_id)