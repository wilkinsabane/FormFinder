#!/usr/bin/env python3
"""
Test script for enhanced feature computation
"""

from formfinder.feature_precomputer import FeaturePrecomputer
from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def test_enhanced_features():
    """Test the enhanced feature computation system"""
    print("Loading configuration...")
    load_config()
    
    print("Getting database session...")
    with get_db_session() as session:
        print("Initializing FeaturePrecomputer...")
        precomputer = FeaturePrecomputer(session)
        
        print("Finding test fixture...")
        result = session.execute(text("SELECT id FROM fixtures WHERE status = 'Not Started' LIMIT 1")).fetchall()
        
        if result:
            fixture_id = result[0][0]
            print(f"Computing features for fixture {fixture_id}...")
            
            try:
                precomputer._compute_fixture_features(fixture_id, force_refresh=True)
                print("✅ Feature computation completed successfully!")
                
                # Check how many features were stored
                feature_check = session.execute(text(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'pre_computed_features'"
                )).fetchone()
                
                print(f"Total columns in pre_computed_features table: {feature_check[0]}")
                
                # Check if the fixture has computed features
                computed_check = session.execute(text(
                    "SELECT COUNT(*) FROM pre_computed_features WHERE fixture_id = :fixture_id"
                ), {'fixture_id': fixture_id}).fetchone()
                
                print(f"Features computed for fixture {fixture_id}: {computed_check[0] > 0}")
                
            except Exception as e:
                print(f"❌ Error during feature computation: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No upcoming fixtures found for testing")
            
            # Try with any fixture
            result = session.execute(text("SELECT id FROM fixtures LIMIT 1")).fetchall()
            if result:
                fixture_id = result[0][0]
                print(f"Testing with any fixture {fixture_id}...")
                
                try:
                    precomputer._compute_fixture_features(fixture_id, force_refresh=True)
                    print("✅ Feature computation completed successfully!")
                except Exception as e:
                    print(f"❌ Error during feature computation: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("No fixtures found in database")

if __name__ == "__main__":
    test_enhanced_features()