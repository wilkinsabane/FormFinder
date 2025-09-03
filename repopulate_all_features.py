#!/usr/bin/env python3
"""
Script to repopulate all pre-computed features with the corrected implementation.
This will ensure all existing records get the fixed home_score, away_score, 
home_form_last_5_games, and away_form_last_5_games values.
"""

import sys
import os
from sqlalchemy import text
from formfinder.config import load_config, get_config
from formfinder.database import get_db_session
from populate_precomputed_features_unified import UnifiedFeaturePopulator

def repopulate_all_features():
    """Repopulate all pre-computed features with corrected implementation."""
    
    # Load configuration
    load_config()
    config = get_config()
    
    print("Starting full repopulation of pre-computed features...")
    
    # Get all fixture IDs that need repopulation
    with get_db_session() as session:
        fixtures_query = text("""
            SELECT DISTINCT fixture_id 
            FROM pre_computed_features 
            ORDER BY fixture_id
        """)
        
        result = session.execute(fixtures_query)
        fixture_ids = [row.fixture_id for row in result.fetchall()]
    
    print(f"Found {len(fixture_ids)} fixtures to repopulate")
    
    if len(fixture_ids) == 0:
        print("No fixtures found to repopulate.")
        return
    
    # Create populator instance
    populator = UnifiedFeaturePopulator()
    
    # Process all fixtures
    try:
        populator.populate_features(fixture_ids)
        print(f"✅ Successfully repopulated features for {len(fixture_ids)} fixtures")
    except Exception as e:
        print(f"❌ Error during repopulation: {e}")
        return
    
    # Verify the repopulation
    print("\nVerifying repopulation results...")
    
    with get_db_session() as session:
        # Check for any remaining null values
        null_check_query = text("""
            SELECT COUNT(*) as null_count
            FROM pre_computed_features 
            WHERE home_score IS NULL 
                OR away_score IS NULL
                OR home_form_last_5_games IS NULL
                OR away_form_last_5_games IS NULL
        """)
        
        null_result = session.execute(null_check_query).fetchone()
        null_count = null_result.null_count
        
        if null_count == 0:
            print("✅ All fields successfully populated - no null values remaining!")
        else:
            print(f"⚠️ Warning: {null_count} records still have null values")
        
        # Show final statistics
        total_query = text("SELECT COUNT(*) as total FROM pre_computed_features")
        total_result = session.execute(total_query).fetchone()
        total_records = total_result.total
        
        print(f"\nFinal Statistics:")
        print(f"  Total records: {total_records}")
        print(f"  Records with null values: {null_count}")
        print(f"  Successfully populated: {total_records - null_count} ({((total_records - null_count) / total_records * 100):.1f}%)")
    
    print("\n=== Repopulation Complete ===")

if __name__ == "__main__":
    repopulate_all_features()