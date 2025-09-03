#!/usr/bin/env python3
"""
Check completed fixtures to understand historical data availability.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
import pandas as pd

def check_completed_fixtures():
    """Check completed fixtures and their distribution."""
    load_config()
    
    with get_db_session() as session:
        conn = session.get_bind().raw_connection()
        
        # Check completed fixtures by date
        query1 = """
        SELECT 
            DATE(match_date) as match_date,
            COUNT(*) as completed_fixtures
        FROM fixtures 
        WHERE home_score IS NOT NULL
        GROUP BY DATE(match_date)
        ORDER BY match_date DESC
        LIMIT 20
        """
        
        df1 = pd.read_sql(query1, conn)
        print("=== Completed Fixtures by Date (Recent 20 days) ===")
        print(df1.to_string(index=False))
        print()
        
        # Check a specific fixture that failed
        query2 = """
        SELECT 
            f.id,
            f.home_team_id,
            f.away_team_id,
            f.match_date,
            f.home_score,
            f.away_score,
            f.league_id,
            -- Count historical fixtures for home team
            (SELECT COUNT(*) FROM fixtures f2 
             WHERE (f2.home_team_id = f.home_team_id OR f2.away_team_id = f.home_team_id)
             AND f2.home_score IS NOT NULL 
             AND f2.match_date < f.match_date
             AND f2.league_id = f.league_id) as home_team_history,
            -- Count historical fixtures for away team
            (SELECT COUNT(*) FROM fixtures f3 
             WHERE (f3.home_team_id = f.away_team_id OR f3.away_team_id = f.away_team_id)
             AND f3.home_score IS NOT NULL 
             AND f3.match_date < f.match_date
             AND f3.league_id = f.league_id) as away_team_history
        FROM fixtures f
        WHERE f.id IN (19435, 19434, 19433)
        ORDER BY f.id DESC
        """
        
        df2 = pd.read_sql(query2, conn)
        print("=== Sample Failed Fixtures Analysis ===")
        print(df2.to_string(index=False))
        print()
        
        # Check earliest completed fixtures by league
        query3 = """
        SELECT 
            league_id,
            MIN(match_date) as earliest_completed,
            MAX(match_date) as latest_completed,
            COUNT(*) as total_completed
        FROM fixtures 
        WHERE home_score IS NOT NULL
        GROUP BY league_id
        ORDER BY league_id
        """
        
        df3 = pd.read_sql(query3, conn)
        print("=== Completed Fixtures by League ===")
        print(df3.to_string(index=False))
        
        conn.close()

if __name__ == "__main__":
    check_completed_fixtures()