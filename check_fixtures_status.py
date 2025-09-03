#!/usr/bin/env python3
"""
Check fixtures status in the database to understand why feature extraction is failing.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
import pandas as pd

def check_fixtures_status():
    """Check the status of fixtures in the database."""
    load_config()
    with get_db_session() as session:
        conn = session.get_bind().raw_connection()
        
        # Check overall fixture statistics
        query1 = """
        SELECT 
            COUNT(*) as total_fixtures,
            COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as completed_fixtures,
            MIN(match_date) as earliest_date,
            MAX(match_date) as latest_date
        FROM fixtures
        """
        
        df1 = pd.read_sql(query1, conn)
        print("=== Fixture Statistics ===")
        print(df1.to_string(index=False))
        print()
        
        # Check recent fixtures
        query2 = """
        SELECT 
            id,
            home_team_id,
            away_team_id,
            match_date,
            home_score,
            away_score,
            league_id
        FROM fixtures 
        ORDER BY id DESC 
        LIMIT 10
        """
        
        df2 = pd.read_sql(query2, conn)
        print("=== Recent Fixtures ===")
        print(df2.to_string(index=False))
        print()
        
        # Check fixtures by completion status
        query3 = """
        SELECT 
            league_id,
            COUNT(*) as total,
            COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as completed,
            COUNT(CASE WHEN home_score IS NULL THEN 1 END) as pending
        FROM fixtures 
        GROUP BY league_id
        ORDER BY league_id
        """
        
        df3 = pd.read_sql(query3, conn)
        print("=== Fixtures by League ===")
        print(df3.to_string(index=False))
        
        conn.close()

if __name__ == "__main__":
    check_fixtures_status()