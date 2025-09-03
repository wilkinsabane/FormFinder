#!/usr/bin/env python3
"""Test script to verify enhanced predictor sentiment feature extraction."""

import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and initialize config first
from formfinder.config import load_config, get_config
load_config()

# Now import enhanced predictor
from enhanced_predictor import EnhancedGoalPredictor
from formfinder.database import DatabaseManager, Fixture

def test_enhanced_sentiment():
    """Test enhanced predictor sentiment feature extraction."""
    print("Testing Enhanced Predictor Sentiment Features")
    print("=" * 50)
    
    try:
        # Initialize enhanced predictor
        predictor = EnhancedGoalPredictor()
        print(f"✅ Enhanced predictor initialized")
        print(f"Sentiment analyzer available: {predictor.sentiment_analyzer is not None}")
        
        if predictor.sentiment_analyzer:
            print(f"Sentiment analyzer API key: {predictor.sentiment_analyzer.api_key[:10]}...")
        
        # Get a test fixture
        db_manager = DatabaseManager()
        with db_manager.get_session() as session:
            fixture = session.query(Fixture).filter(
                Fixture.status == 'finished'
            ).first()
            
            if not fixture:
                print("❌ No finished fixtures found for testing")
                return
                
            print(f"\n📋 Testing with fixture {fixture.id}:")
            print(f"   {fixture.home_team.name} vs {fixture.away_team.name}")
            print(f"   Date: {fixture.match_date}")
            
            # Extract features
            features = predictor.extract_enhanced_features(fixture.id)
            
            if features:
                print(f"\n✅ Features extracted successfully")
                print(f"Total features: {len(features)}")
                
                # Check sentiment features specifically
                home_sentiment = features.get('home_team_sentiment', 'NOT_FOUND')
                away_sentiment = features.get('away_team_sentiment', 'NOT_FOUND')
                
                print(f"\n🎭 Sentiment Features:")
                print(f"   Home team sentiment: {home_sentiment}")
                print(f"   Away team sentiment: {away_sentiment}")
                
                if home_sentiment != 'NOT_FOUND' and away_sentiment != 'NOT_FOUND':
                    if home_sentiment != 0.0 or away_sentiment != 0.0:
                        print("✅ Sentiment features are working!")
                    else:
                        print("⚠️ Sentiment features are 0.0 - may indicate no articles found")
                else:
                    print("❌ Sentiment features not found in extracted features")
                    
            else:
                print("❌ Failed to extract features")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_sentiment()