#!/usr/bin/env python3
"""Debug script to test EnhancedGoalPredictor feature extraction."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from enhanced_predictor import EnhancedGoalPredictor
from sqlalchemy import text
import json

def test_enhanced_predictor():
    """Test the enhanced predictor with a real fixture."""
    print("=== Enhanced Predictor Debug Test ===")
    
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Get a sample fixture
        result = session.execute(text("""
            SELECT f.id, f.home_team_id, f.away_team_id, f.match_date, f.league_id,
                   f.home_score, f.away_score
            FROM fixtures f
            WHERE f.home_score IS NOT NULL 
            AND f.away_score IS NOT NULL
            AND f.match_date >= '2020-09-01'
            ORDER BY f.match_date DESC
            LIMIT 1
        """))
        
        fixture = result.fetchone()
        if not fixture:
            print("No fixtures found")
            return
            
        print(f"Testing with fixture {fixture.id}: {fixture.home_team_id} vs {fixture.away_team_id}")
        print(f"Date: {fixture.match_date}, Score: {fixture.home_score}-{fixture.away_score}")
        
        # Initialize enhanced predictor
        try:
            enhanced_predictor = EnhancedGoalPredictor()
            print("✅ Enhanced predictor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize enhanced predictor: {e}")
            return
            
        # Test feature extraction
        try:
            features = enhanced_predictor.extract_enhanced_features(fixture.id)
            
            print(f"\n✅ Features extracted successfully")
            print(f"Number of features: {len(features)}")
            
            # Print key features we're interested in
            key_features = [
                'home_team_strength', 'away_team_strength',
                'home_team_momentum', 'away_team_momentum', 
                'home_team_sentiment', 'away_team_sentiment',
                'home_xg', 'away_xg'
            ]
            
            print("\n=== Key Enhanced Features ===")
            for key in key_features:
                value = features.get(key, 'NOT_FOUND')
                print(f"{key}: {value}")
                
            # Print all features for debugging
            print("\n=== All Features (first 20) ===")
            for i, (key, value) in enumerate(features.items()):
                if i >= 20:
                    print(f"... and {len(features) - 20} more features")
                    break
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"❌ Failed to extract features: {e}")
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    test_enhanced_predictor()