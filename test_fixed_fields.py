#!/usr/bin/env python3
"""
Test script to verify that the fixed populate_precomputed_features_unified.py
correctly populates home_score, away_score, home_form_last_5_games, and away_form_last_5_games fields.
"""

import asyncio
import sys
import os
from sqlalchemy import text
from formfinder.config import load_config, get_config
from formfinder.database import get_db_session

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from populate_precomputed_features_unified import UnifiedFeaturePopulator

async def test_fixed_fields():
    """Test that the fixed script correctly populates the previously null fields."""
    
    # Load configuration
    load_config()
    config = get_config()
    
    # Create populator instance
    populator = UnifiedFeaturePopulator()
    
    print("Testing fixed fields population...")
    
    # Process just 5 fixtures to test
    await populator.populate_features(limit=5)
    
    # Check the results
    with get_db_session() as session:
        # Query the last 5 processed fixtures
        query = text("""
            SELECT 
                fixture_id,
                home_score,
                away_score,
                home_form_last_5_games,
                away_form_last_5_games,
                computation_source
            FROM pre_computed_features 
            WHERE computation_source = 'unified'
            ORDER BY features_computed_at DESC
            LIMIT 5
        """)
        
        result = session.execute(query)
        rows = result.fetchall()
        
        print(f"\nFound {len(rows)} recently processed fixtures:")
        
        for row in rows:
            print(f"\nFixture {row.fixture_id}:")
            print(f"  Home Score: {row.home_score}")
            print(f"  Away Score: {row.away_score}")
            print(f"  Home Form: {row.home_form_last_5_games}")
            print(f"  Away Form: {row.away_form_last_5_games}")
            print(f"  Source: {row.computation_source}")
            
            # Check for null values
            null_fields = []
            if row.home_score is None:
                null_fields.append('home_score')
            if row.away_score is None:
                null_fields.append('away_score')
            if row.home_form_last_5_games is None:
                null_fields.append('home_form_last_5_games')
            if row.away_form_last_5_games is None:
                null_fields.append('away_form_last_5_games')
                
            if null_fields:
                print(f"  ❌ NULL fields found: {', '.join(null_fields)}")
            else:
                print(f"  ✅ All fields populated correctly")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_fixed_fields())