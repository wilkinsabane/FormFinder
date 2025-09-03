#!/usr/bin/env python3
"""
Check for null values in pre_computed_features table fields.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def check_null_fields():
    """Check for null values in key fields of pre_computed_features table."""
    load_config()
    
    with get_db_session() as session:
        # Check for null values in key fields
        null_check_query = text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN home_form_last_5_games IS NULL THEN 1 END) as null_home_form,
                COUNT(CASE WHEN away_form_last_5_games IS NULL THEN 1 END) as null_away_form,
                COUNT(CASE WHEN home_score IS NULL THEN 1 END) as null_home_score,
                COUNT(CASE WHEN away_score IS NULL THEN 1 END) as null_away_score
            FROM pre_computed_features
        """)
        
        result = session.execute(null_check_query).fetchone()
        
        print(f"Pre-computed features table analysis:")
        print(f"Total records: {result.total_records}")
        print(f"Null home_form_last_5_games: {result.null_home_form}")
        print(f"Null away_form_last_5_games: {result.null_away_form}")
        print(f"Null home_score: {result.null_home_score}")
        print(f"Null away_score: {result.null_away_score}")
        
        # Get sample records with null values
        sample_query = text("""
            SELECT fixture_id, home_form_last_5_games, away_form_last_5_games, 
                   home_score, away_score, computation_source
            FROM pre_computed_features 
            WHERE home_form_last_5_games IS NULL 
               OR away_form_last_5_games IS NULL 
               OR home_score IS NULL 
               OR away_score IS NULL
            LIMIT 10
        """)
        
        sample_results = session.execute(sample_query).fetchall()
        
        print("\nSample records with null values:")
        for row in sample_results:
            print(f"Fixture {row.fixture_id}: home_form={row.home_form_last_5_games}, "
                  f"away_form={row.away_form_last_5_games}, home_score={row.home_score}, "
                  f"away_score={row.away_score}, source={row.computation_source}")

if __name__ == "__main__":
    check_null_fields()