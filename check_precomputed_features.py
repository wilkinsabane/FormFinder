#!/usr/bin/env python3
"""Check pre-computed features table data."""

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
import pandas as pd

def check_precomputed_features():
    """Check what data exists in pre_computed_features table."""
    print("=== Pre-computed Features Analysis ===")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Check total rows
    query = """
    SELECT 
        COUNT(*) as total_rows,
        MIN(match_date) as earliest_date,
        MAX(match_date) as latest_date
    FROM pre_computed_features 
    WHERE total_goals IS NOT NULL
    """
    
    try:
        result = pd.read_sql_query(text(query), engine)
        print(f"Total pre-computed features: {result.iloc[0]['total_rows']}")
        print(f"Date range: {result.iloc[0]['earliest_date']} to {result.iloc[0]['latest_date']}")
    except Exception as e:
        print(f"Error querying pre_computed_features: {e}")
        print("Table might not exist or be empty")
        return
    
    # Check by league
    league_query = """
    SELECT 
        league_id,
        COUNT(*) as feature_count
    FROM pre_computed_features 
    WHERE total_goals IS NOT NULL
    GROUP BY league_id
    ORDER BY COUNT(*) DESC
    LIMIT 10
    """
    
    try:
        league_result = pd.read_sql_query(text(league_query), engine)
        print("\nTop leagues in pre_computed_features:")
        for _, row in league_result.iterrows():
            print(f"  League {row['league_id']}: {row['feature_count']} features")
    except Exception as e:
        print(f"Error querying league data: {e}")
    
    # Check table structure
    structure_query = """
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'pre_computed_features'
    ORDER BY ordinal_position
    """
    
    try:
        structure_result = pd.read_sql_query(text(structure_query), engine)
        print(f"\nTable structure ({len(structure_result)} columns):")
        for _, row in structure_result.head(10).iterrows():
            print(f"  {row['column_name']}: {row['data_type']}")
        if len(structure_result) > 10:
            print(f"  ... and {len(structure_result) - 10} more columns")
    except Exception as e:
        print(f"Error querying table structure: {e}")

if __name__ == "__main__":
    check_precomputed_features()