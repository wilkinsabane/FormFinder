#!/usr/bin/env python3
"""Test momentum calculation with proper configuration loading."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from formfinder.database import get_db_session
from datetime import datetime

def test_momentum():
    """Test momentum calculation."""
    try:
        # Load configuration
        print("Loading configuration...")
        load_config()
        print("Configuration loaded successfully")
        
        # Initialize components
        print("Initializing MarkovFeatureGenerator...")
        db = get_db_session()
        generator = MarkovFeatureGenerator(db)
        print("MarkovFeatureGenerator initialized")
        
        # Test momentum calculation for team 3245
        print("Calculating momentum for team 3245...")
        momentum = generator.calculate_momentum_score(
            team_id=3245, 
            league_id=207, 
            reference_date=datetime(2024, 9, 24), 
            context='overall'
        )
        
        print(f"Momentum for team 3245: {momentum}")
        
        # Test full feature generation
        print("\nGenerating full Markov features...")
        features = generator.generate_team_features(
            team_id=3245,
            league_id=207,
            reference_date=datetime(2024, 9, 24),
            context='overall'
        )
        
        print("Generated features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_momentum()