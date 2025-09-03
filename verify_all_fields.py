#!/usr/bin/env python3
"""
Comprehensive verification script to check that all previously null fields
are now being populated correctly across the entire dataset.
"""

import sys
import os
from sqlalchemy import text
from formfinder.config import load_config, get_config
from formfinder.database import get_db_session

def verify_all_fields():
    """Verify that all fields are properly populated."""
    
    # Load configuration
    load_config()
    config = get_config()
    
    print("Verifying field population across all records...")
    
    with get_db_session() as session:
        # Check total records
        total_query = text("SELECT COUNT(*) as total FROM pre_computed_features")
        total_result = session.execute(total_query).fetchone()
        total_records = total_result.total
        
        print(f"\nTotal records in pre_computed_features: {total_records}")
        
        # Check null counts for each field
        fields_to_check = [
            'home_score',
            'away_score', 
            'home_form_last_5_games',
            'away_form_last_5_games'
        ]
        
        for field in fields_to_check:
            null_query = text(f"""
                SELECT COUNT(*) as null_count 
                FROM pre_computed_features 
                WHERE {field} IS NULL
            """)
            null_result = session.execute(null_query).fetchone()
            null_count = null_result.null_count
            
            percentage = (null_count / total_records * 100) if total_records > 0 else 0
            
            if null_count == 0:
                print(f"✅ {field}: 0 null values (100% populated)")
            else:
                print(f"❌ {field}: {null_count} null values ({percentage:.1f}% null)")
        
        # Show sample of populated records
        print("\n=== Sample of Populated Records ===")
        sample_query = text("""
            SELECT 
                fixture_id,
                home_score,
                away_score,
                home_form_last_5_games,
                away_form_last_5_games,
                computation_source
            FROM pre_computed_features 
            WHERE home_score IS NOT NULL 
                AND away_score IS NOT NULL
                AND home_form_last_5_games IS NOT NULL
                AND away_form_last_5_games IS NOT NULL
            ORDER BY features_computed_at DESC
            LIMIT 5
        """)
        
        sample_result = session.execute(sample_query)
        sample_rows = sample_result.fetchall()
        
        for row in sample_rows:
            print(f"\nFixture {row.fixture_id} ({row.computation_source}):")
            print(f"  Scores: {row.home_score}-{row.away_score}")
            print(f"  Home Form: {row.home_form_last_5_games}")
            print(f"  Away Form: {row.away_form_last_5_games}")
        
        # Check for any remaining null records
        print("\n=== Records with Any Null Values ===")
        null_records_query = text("""
            SELECT 
                fixture_id,
                computation_source,
                CASE WHEN home_score IS NULL THEN 'home_score ' ELSE '' END ||
                CASE WHEN away_score IS NULL THEN 'away_score ' ELSE '' END ||
                CASE WHEN home_form_last_5_games IS NULL THEN 'home_form ' ELSE '' END ||
                CASE WHEN away_form_last_5_games IS NULL THEN 'away_form ' ELSE '' END as null_fields
            FROM pre_computed_features 
            WHERE home_score IS NULL 
                OR away_score IS NULL
                OR home_form_last_5_games IS NULL
                OR away_form_last_5_games IS NULL
            LIMIT 10
        """)
        
        null_records_result = session.execute(null_records_query)
        null_records = null_records_result.fetchall()
        
        if null_records:
            print(f"Found {len(null_records)} records with null values:")
            for record in null_records:
                print(f"  Fixture {record.fixture_id} ({record.computation_source}): {record.null_fields.strip()}")
        else:
            print("✅ No records with null values found!")
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify_all_fields()