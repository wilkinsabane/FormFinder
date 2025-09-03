#!/usr/bin/env python3
"""
Debug specific fixtures that are failing feature extraction.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
import pandas as pd

def debug_specific_fixtures():
    """Debug specific fixtures that failed feature extraction."""
    load_config()
    
    with get_db_session() as session:
        conn = session.get_bind().raw_connection()
        
        # Check the specific fixtures that failed
        query1 = """
        SELECT 
            id,
            home_team_id,
            away_team_id,
            match_date,
            home_score,
            away_score,
            league_id
        FROM fixtures 
        WHERE id IN (19435, 19434, 19433, 19432, 19431)
        ORDER BY id DESC
        """
        
        df1 = pd.read_sql(query1, conn)
        print("=== Failed Fixtures Details ===")
        print(df1.to_string(index=False))
        print()
        
        # Check what fixtures exist in pre_computed_features table
        query2 = """
        SELECT 
            COUNT(*) as total_precomputed,
            MIN(fixture_id) as min_fixture_id,
            MAX(fixture_id) as max_fixture_id
        FROM pre_computed_features
        """
        
        df2 = pd.read_sql(query2, conn)
        print("=== Pre-computed Features Status ===")
        print(df2.to_string(index=False))
        print()
        
        # Check which fixtures need feature computation
        query3 = """
        SELECT 
            f.id,
            f.match_date,
            f.home_score,
            f.away_score,
            f.league_id,
            CASE WHEN pcf.fixture_id IS NOT NULL THEN 'YES' ELSE 'NO' END as has_features
        FROM fixtures f
        LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
        WHERE f.home_score IS NOT NULL
        ORDER BY f.id DESC
        LIMIT 20
        """
        
        df3 = pd.read_sql(query3, conn)
        print("=== Recent Completed Fixtures and Feature Status ===")
        print(df3.to_string(index=False))
        print()
        
        # Check if there are any completed fixtures that could provide historical data
        query4 = """
        SELECT 
            league_id,
            COUNT(*) as completed_fixtures,
            MIN(match_date) as earliest_date,
            MAX(match_date) as latest_date
        FROM fixtures 
        WHERE home_score IS NOT NULL
        AND match_date < '2024-05-12'  -- Before the failed fixtures
        GROUP BY league_id
        HAVING COUNT(*) > 0
        ORDER BY league_id
        """
        
        df4 = pd.read_sql(query4, conn)
        print("=== Historical Data Available (before 2024-05-12) ===")
        print(df4.to_string(index=False))
        
        conn.close()

if __name__ == "__main__":
    debug_specific_fixtures()