#!/usr/bin/env python3
"""
Simple test to verify enhanced predictor feature extraction works correctly.
"""

import logging
from formfinder.config import load_config
from formfinder.database import DatabaseManager
from enhanced_predictor import EnhancedGoalPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_features():
    """Test enhanced predictor feature extraction."""
    # Load configuration
    load_config()
    
    # Initialize components
    db_manager = DatabaseManager()
    predictor = EnhancedGoalPredictor()
    
    # Get a test fixture
    with db_manager.get_session() as session:
        from formfinder.database import Fixture
        fixture = session.query(Fixture).filter(
            Fixture.status == 'finished'
        ).first()
        
        if not fixture:
            logger.error("No finished fixtures found for testing")
            return False
            
        logger.info(f"Testing with fixture {fixture.id}: {fixture.home_team.name} vs {fixture.away_team.name}")
        
        # Extract features
        try:
            features = predictor.extract_enhanced_features(fixture.id)
            logger.info(f"Successfully extracted {len(features)} features")
            
            # Print feature keys
            feature_keys = list(features.keys())
            logger.info(f"Feature keys: {feature_keys[:10]}...")  # Show first 10
            
            # Check for expected feature types
            expected_features = [
                'home_team_form_score', 'away_team_form_score',
                'home_team_position', 'away_team_position',
                'home_xg', 'away_xg',
                'home_team_strength', 'away_team_strength'
            ]
            
            missing_features = [f for f in expected_features if f not in features]
            if missing_features:
                logger.warning(f"Missing expected features: {missing_features}")
            else:
                logger.info("All expected features present")
                
            return True
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return False

if __name__ == "__main__":
    success = test_enhanced_features()
    if success:
        print("✅ Enhanced feature extraction test PASSED")
    else:
        print("❌ Enhanced feature extraction test FAILED")
        exit(1)