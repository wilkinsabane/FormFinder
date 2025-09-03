#!/usr/bin/env python3
"""
Find fixtures that have sufficient historical data for feature calculation.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
import pandas as pd
from datetime import datetime, timedelta

def find_processable_fixtures():
    """Find fixtures that have sufficient historical data."""
    load_config()
    
    with get_db_session() as session:
        conn = session.get_bind().raw_connection()
        
        # Find fixtures with sufficient historical data (at least 10 games per team)
        query = """
        WITH fixture_history AS (
            SELECT 
                f.id,
                f.match_date,
                f.league_id,
                f.home_team_id,
                f.away_team_id,
                f.home_score,
                f.away_score,
                -- Count historical games for home team
                (SELECT COUNT(*) FROM fixtures f2 
                 WHERE (f2.home_team_id = f.home_team_id OR f2.away_team_id = f.home_team_id)
                 AND f2.home_score IS NOT NULL 
                 AND f2.match_date < f.match_date
                 AND f2.league_id = f.league_id) as home_team_history,
                -- Count historical games for away team
                (SELECT COUNT(*) FROM fixtures f3 
                 WHERE (f3.home_team_id = f.away_team_id OR f3.away_team_id = f.away_team_id)
                 AND f3.home_score IS NOT NULL 
                 AND f3.match_date < f.match_date
                 AND f3.league_id = f.league_id) as away_team_history,
                -- Check if already processed
                CASE WHEN pcf.fixture_id IS NOT NULL THEN 1 ELSE 0 END as already_processed
            FROM fixtures f
            LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.home_score IS NOT NULL 
            AND f.away_score IS NOT NULL
            AND f.status = 'finished'
        )
        SELECT 
            league_id,
            COUNT(*) as total_completed,
            COUNT(CASE WHEN home_team_history >= 5 AND away_team_history >= 5 THEN 1 END) as processable,
            COUNT(CASE WHEN already_processed = 1 THEN 1 END) as already_processed,
            COUNT(CASE WHEN home_team_history >= 5 AND away_team_history >= 5 AND already_processed = 0 THEN 1 END) as needs_processing,
            MIN(match_date) as earliest_date,
            MAX(match_date) as latest_date
        FROM fixture_history
        GROUP BY league_id
        ORDER BY needs_processing DESC
        """
        
        df = pd.read_sql(query, conn)
        print("=== Fixtures with Sufficient Historical Data ===")
        print(df.to_string(index=False))
        print()
        
        # Find specific fixtures that can be processed
        query2 = """
        WITH fixture_history AS (
            SELECT 
                f.id,
                f.match_date,
                f.league_id,
                f.home_team_id,
                f.away_team_id,
                -- Count historical games for home team
                (SELECT COUNT(*) FROM fixtures f2 
                 WHERE (f2.home_team_id = f.home_team_id OR f2.away_team_id = f.home_team_id)
                 AND f2.home_score IS NOT NULL 
                 AND f2.match_date < f.match_date
                 AND f2.league_id = f.league_id) as home_team_history,
                -- Count historical games for away team
                (SELECT COUNT(*) FROM fixtures f3 
                 WHERE (f3.home_team_id = f.away_team_id OR f3.away_team_id = f.away_team_id)
                 AND f3.home_score IS NOT NULL 
                 AND f3.match_date < f.match_date
                 AND f3.league_id = f.league_id) as away_team_history
            FROM fixtures f
            LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.home_score IS NOT NULL 
            AND f.away_score IS NOT NULL
            AND f.status = 'finished'
            AND pcf.fixture_id IS NULL
        )
        SELECT 
            id,
            match_date,
            league_id,
            home_team_history,
            away_team_history
        FROM fixture_history
        WHERE home_team_history >= 5 AND away_team_history >= 5
        ORDER BY match_date DESC
        LIMIT 20
        """
        
        df2 = pd.read_sql(query2, conn)
        print("=== Sample Processable Fixtures ===")
        print(df2.to_string(index=False))
        
        conn.close()

if __name__ == "__main__":
    find_processable_fixtures()