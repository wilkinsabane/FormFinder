#!/usr/bin/env python3
"""Check fixture date ranges and counts for enhanced_predictor debugging."""

from sqlalchemy import create_engine, text
import pandas as pd
from formfinder.config import load_config, get_config
from datetime import datetime, timedelta

def main():
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Current cutoff date used by enhanced_predictor (365 days back)
    cutoff_date = datetime.now() - timedelta(days=365)
    print(f"Current cutoff date (365 days back): {cutoff_date.strftime('%Y-%m-%d')}")
    
    # Check fixtures since cutoff date
    query = """
    SELECT l.name, l.id, COUNT(f.id) as fixture_count 
    FROM fixtures f 
    JOIN leagues l ON f.league_id = l.id 
    WHERE f.home_score IS NOT NULL 
        AND f.match_date >= :cutoff_date 
    GROUP BY l.id, l.name 
    ORDER BY fixture_count DESC
    """
    
    result = pd.read_sql_query(text(query), engine, params={'cutoff_date': cutoff_date.isoformat()})
    print("\nFixtures since cutoff date:")
    print(result)
    
    # Check with more lenient cutoff (2 years back)
    cutoff_date_2y = datetime.now() - timedelta(days=730)
    print(f"\nWith 2-year cutoff date: {cutoff_date_2y.strftime('%Y-%m-%d')}")
    
    result_2y = pd.read_sql_query(text(query), engine, params={'cutoff_date': cutoff_date_2y.isoformat()})
    print("\nFixtures since 2-year cutoff:")
    print(result_2y)
    
    # Check specific leagues that failed
    failed_leagues = [40, 209, 229]  # From the error logs
    print("\nChecking specific failed leagues:")
    for league_id in failed_leagues:
        league_query = """
        SELECT l.name, l.id, COUNT(f.id) as fixture_count,
               MIN(f.match_date) as earliest,
               MAX(f.match_date) as latest
        FROM fixtures f 
        JOIN leagues l ON f.league_id = l.id 
        WHERE f.home_score IS NOT NULL 
            AND l.id = :league_id
        GROUP BY l.id, l.name
        """
        
        league_result = pd.read_sql_query(text(league_query), engine, params={'league_id': league_id})
        print(f"\nLeague {league_id}:")
        print(league_result)

if __name__ == "__main__":
    main()