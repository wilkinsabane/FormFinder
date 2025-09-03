#!/usr/bin/env python3
"""
Debug script to test enhanced predictor feature extraction.
"""

import sys
sys.path.append('.')

from enhanced_predictor import EnhancedGoalPredictor
import json

def test_enhanced_predictor():
    """Test enhanced predictor feature extraction."""
    
    # Initialize predictor
    predictor = EnhancedGoalPredictor()
    
    # Test fixture ID
    fixture_id = 20819
    
    print(f"Testing enhanced predictor for fixture {fixture_id}")
    print("=" * 50)
    
    try:
        # Extract features
        features = predictor.extract_enhanced_features(fixture_id)
        
        if features:
            print(f"Total features extracted: {len(features)}")
            print("\nMarkov features found:")
            
            markov_features = {k: v for k, v in features.items() if 'markov' in k.lower()}
            
            if markov_features:
                for key, value in markov_features.items():
                    print(f"  {key}: {value} (type: {type(value)})")
            else:
                print("  No Markov features found!")
                
            print("\nAll features:")
            for key, value in features.items():
                print(f"  {key}: {value}")
        else:
            print("No features extracted!")
            
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_predictor()