#!/usr/bin/env python3
"""
Test script to verify that all 87 features are computed and stored correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.feature_precomputer import FeaturePrecomputer
from sqlalchemy import text
import json

def test_feature_storage():
    """Test that all 87 features are computed and stored correctly."""
    print("Testing feature computation and storage...")
    
    # Load configuration
    print("Loading configuration...")
    load_config()
    
    # Get database session
    with get_db_session() as db_session:
        # Initialize FeaturePrecomputer
        precomputer = FeaturePrecomputer(db_session)
        
        # Find a test fixture
        query = text("""
            SELECT id FROM fixtures 
            WHERE home_team_id IS NOT NULL 
            AND away_team_id IS NOT NULL 
            AND league_id IS NOT NULL
            LIMIT 1
        """)
        
        result = db_session.execute(query).fetchone()
        if not result:
            print("❌ No fixtures found for testing")
            return False
        
        fixture_id = result[0]
        print(f"Testing with fixture {fixture_id}...")
        
        # Compute features for the fixture
        try:
            stats = precomputer.compute_all_features([fixture_id], force_refresh=True)
            print(f"✅ Feature computation stats: {stats}")
            
            if stats['successful_computations'] == 0:
                print("❌ No features were computed successfully")
                return False
                
        except Exception as e:
            print(f"❌ Feature computation failed: {e}")
            return False
        
        # Verify features are stored in database
        query = text("""
            SELECT * FROM pre_computed_features 
            WHERE fixture_id = :fixture_id
        """)
        
        result = db_session.execute(query, {'fixture_id': fixture_id}).fetchone()
        if not result:
            print("❌ No features found in database")
            return False
        
        # Convert result to dictionary
        feature_data = dict(result._mapping)
        
        # Count non-null features (excluding metadata columns)
        metadata_columns = {
            'id', 'fixture_id', 'home_team_id', 'away_team_id', 'match_date', 
            'league_id', 'total_goals', 'over_2_5', 'home_score', 'away_score', 
            'match_result', 'features_computed_at', 'data_quality_score', 
            'computation_source', 'features_json'
        }
        
        feature_columns = [col for col in feature_data.keys() if col not in metadata_columns]
        non_null_features = [col for col in feature_columns if feature_data[col] is not None]
        
        print(f"✅ Found {len(non_null_features)} non-null features out of {len(feature_columns)} total feature columns")
        print(f"✅ Data quality score: {feature_data.get('data_quality_score', 'N/A')}")
        print(f"✅ Computation source: {feature_data.get('computation_source', 'N/A')}")
        
        # Print some sample features
        print("\nSample features:")
        sample_features = [
            'home_goals_scored_last_5', 'away_goals_scored_last_5',
            'markov_home_current_state', 'markov_away_current_state',
            'home_position', 'away_position',
            'home_xg_avg', 'away_xg_avg'
        ]
        
        for feature in sample_features:
            if feature in feature_data:
                value = feature_data[feature]
                print(f"  {feature}: {value} (type: {type(value).__name__})")
        
        # Check if we have at least 80 features (allowing for some optional ones)
        if len(non_null_features) >= 80:
            print(f"\n✅ SUCCESS: {len(non_null_features)} features computed and stored successfully!")
            return True
        else:
            print(f"\n⚠️  WARNING: Only {len(non_null_features)} features found, expected at least 80")
            print("Missing or null features:")
            null_features = [col for col in feature_columns if feature_data[col] is None]
            for feature in null_features[:10]:  # Show first 10 missing
                print(f"  - {feature}")
            if len(null_features) > 10:
                print(f"  ... and {len(null_features) - 10} more")
            return False

if __name__ == "__main__":
    success = test_feature_storage()
    sys.exit(0 if success else 1)