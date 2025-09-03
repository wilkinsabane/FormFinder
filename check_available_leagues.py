#!/usr/bin/env python3
"""
Script to check available leagues and their fixture counts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    # Load configuration first
    load_config()
    
    with get_db_session() as session:
        print("=== Available Leagues with Fixtures ===")
        
        # Check leagues with fixture counts
        leagues_query = """
        SELECT l.league_pk, l.name, COUNT(f.id) as fixture_count,
               MIN(f.match_date) as earliest_date,
               MAX(f.match_date) as latest_date
        FROM leagues l
        LEFT JOIN fixtures f ON l.league_pk = f.league_id
        WHERE f.id IS NOT NULL
        GROUP BY l.league_pk, l.name
        ORDER BY fixture_count DESC
        LIMIT 20
        """
        
        try:
            results = session.execute(text(leagues_query)).fetchall()
            print(f"Found {len(results)} leagues with fixtures:")
            print()
            
            for row in results:
                league_id, name, count, earliest, latest = row
                print(f"League {league_id}: {name}")
                print(f"  Fixtures: {count}")
                print(f"  Date range: {earliest} to {latest}")
                print()
                
        except Exception as e:
            print(f"Error checking leagues: {e}")
        
        print("\n=== Checking specific leagues ===")
        
        # Check specific popular leagues
        specific_leagues = [39, 140, 78, 135, 61]  # Premier League, La Liga, Bundesliga, Serie A, Ligue 1
        
        for league_id in specific_leagues:
            try:
                count_query = "SELECT COUNT(*) FROM fixtures WHERE league_id = :league_id"
                count = session.execute(text(count_query), {'league_id': league_id}).scalar()
                
                if count > 0:
                    date_query = """
                    SELECT MIN(match_date), MAX(match_date) 
                    FROM fixtures 
                    WHERE league_id = :league_id
                    """
                    dates = session.execute(text(date_query), {'league_id': league_id}).fetchone()
                    print(f"League {league_id}: {count} fixtures ({dates[0]} to {dates[1]})")
                else:
                    print(f"League {league_id}: No fixtures found")
                    
            except Exception as e:
                print(f"Error checking league {league_id}: {e}")

if __name__ == "__main__":
    main()