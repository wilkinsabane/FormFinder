#!/usr/bin/env python3
"""
Test script for improved team position logic.
Tests various scenarios including recent data, older data, median fallback, and default fallback.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from formfinder.database import get_db_session
from formfinder.config import load_config, get_config
from enhanced_predictor import EnhancedGoalPredictor
from formfinder.enhanced_feature_computer import EnhancedFeatureComputer

def test_position_logic():
    """Test the improved position logic with various scenarios."""
    print("Testing improved team position logic...")
    
    # Load configuration
    load_config()
    config = get_config()
    
    with get_db_session() as session:
        # Initialize predictors
        enhanced_predictor = EnhancedGoalPredictor(config=config)
        feature_computer = EnhancedFeatureComputer(session)
        
        # Test scenarios
        test_cases = [
            {
                'name': 'Recent data test',
                'team_id': 1,
                'league_id': 1,
                'match_date': datetime.now() - timedelta(days=5)
            },
            {
                'name': 'Older data test',
                'team_id': 2,
                'league_id': 1,
                'match_date': datetime.now() - timedelta(days=60)
            },
            {
                'name': 'No data test (should use median)',
                'team_id': 9999,  # Non-existent team
                'league_id': 1,
                'match_date': datetime.now()
            },
            {
                'name': 'Non-existent league test (should use default)',
                'team_id': 1,
                'league_id': 9999,  # Non-existent league
                'match_date': datetime.now()
            }
        ]
        
        print("\n=== Enhanced Predictor Tests ===")
        for test_case in test_cases:
            print(f"\nTest: {test_case['name']}")
            print(f"Team ID: {test_case['team_id']}, League ID: {test_case['league_id']}")
            
            try:
                # Test full method (returns tuple)
                position, confidence = enhanced_predictor._get_team_position(
                    test_case['team_id'],
                    test_case['match_date'],
                    test_case['league_id']
                )
                print(f"Position: {position}, Confidence: {confidence}")
                
                # Test simple method (returns just position)
                simple_position = enhanced_predictor._get_team_position_simple(
                    test_case['team_id'],
                    test_case['match_date'],
                    test_case['league_id']
                )
                print(f"Simple position: {simple_position}")
                
                # Validate consistency
                assert position == simple_position, f"Position mismatch: {position} != {simple_position}"
                
                # Validate confidence score
                assert 0.0 <= confidence <= 1.0, f"Invalid confidence score: {confidence}"
                
                print("✓ Test passed")
                
            except Exception as e:
                print(f"✗ Test failed: {e}")
        
        print("\n=== Enhanced Feature Computer Tests ===")
        for test_case in test_cases:
            print(f"\nTest: {test_case['name']}")
            print(f"Team ID: {test_case['team_id']}, League ID: {test_case['league_id']}")
            
            try:
                # Test full method (returns tuple)
                position, confidence = feature_computer._get_team_position(
                    test_case['team_id'],
                    test_case['league_id'],
                    test_case['match_date']
                )
                print(f"Position: {position}, Confidence: {confidence}")
                
                # Test simple method (returns just position)
                simple_position = feature_computer._get_team_position_simple(
                    test_case['team_id'],
                    test_case['league_id'],
                    test_case['match_date']
                )
                print(f"Simple position: {simple_position}")
                
                # Validate consistency
                assert position == simple_position, f"Position mismatch: {position} != {simple_position}"
                
                # Validate confidence score
                assert 0.0 <= confidence <= 1.0, f"Invalid confidence score: {confidence}"
                
                print("✓ Test passed")
                
            except Exception as e:
                print(f"✗ Test failed: {e}")
        
        print("\n=== Feature Extraction Test ===")
        try:
            # Test feature extraction with position confidence
            test_fixture_id = 20819  # Use existing fixture
            features = enhanced_predictor.extract_enhanced_features(test_fixture_id)
            
            print(f"\nFeature extraction for fixture {test_fixture_id}:")
            print(f"Home team position: {features.get('home_team_position', 'N/A')}")
            print(f"Away team position: {features.get('away_team_position', 'N/A')}")
            print(f"Home position confidence: {features.get('home_position_confidence', 'N/A')}")
            print(f"Away position confidence: {features.get('away_position_confidence', 'N/A')}")
            
            # Validate that confidence features exist
            assert 'home_position_confidence' in features, "Missing home_position_confidence feature"
            assert 'away_position_confidence' in features, "Missing away_position_confidence feature"
            
            # Validate confidence values
            home_conf = features['home_position_confidence']
            away_conf = features['away_position_confidence']
            assert 0.0 <= home_conf <= 1.0, f"Invalid home confidence: {home_conf}"
            assert 0.0 <= away_conf <= 1.0, f"Invalid away confidence: {away_conf}"
            
            print("✓ Feature extraction test passed")
            
        except Exception as e:
            print(f"✗ Feature extraction test failed: {e}")
        
        print("\n=== Summary ===")
        print("Improved position logic testing completed.")
        print("Key improvements:")
        print("- Uses most recent position data regardless of match date")
        print("- Falls back to league-specific median when no data available")
        print("- Provides confidence scores for data quality assessment")
        print("- Maintains backward compatibility with simple position method")

if __name__ == "__main__":
    test_position_logic()