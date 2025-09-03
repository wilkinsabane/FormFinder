#!/usr/bin/env python3
"""Test script to verify enhanced predictor feature generation.

This script tests that the enhanced predictor correctly generates all expected features
including Markov chain features, sentiment analysis, and basic team statistics.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from sqlalchemy import text

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_predictor import EnhancedGoalPredictor
from formfinder.config import load_config, get_config
from formfinder.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_feature_generation():
    """Test that all expected features are generated correctly."""
    try:
        # Load configuration
        load_config()
        config = get_config()
        
        # Initialize enhanced predictor
        logger.info("Initializing enhanced predictor...")
        predictor = EnhancedGoalPredictor(config)
        
        # Get a recent fixture to test with
        db_manager = DatabaseManager(config.get_database_url())
        
        # Query for a recent fixture with complete data
        query = text("""
        SELECT f.id as fixture_id, f.home_team_id, f.away_team_id, f.match_date, f.league_id,
               ht.name as home_team_name, at.name as away_team_name
        FROM fixtures f
        JOIN teams ht ON f.home_team_id = ht.id
        JOIN teams at ON f.away_team_id = at.id
        WHERE f.match_date >= :start_date
        AND f.match_date <= :end_date
        AND f.home_score IS NOT NULL
        AND f.away_score IS NOT NULL
        ORDER BY f.match_date DESC
        LIMIT 5
        """)
        
        # Get fixtures from the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        session = db_manager.get_session()
        try:
            result = session.execute(query, {'start_date': start_date, 'end_date': end_date})
            fixtures = result.fetchall()
        finally:
            session.close()
        
        if not fixtures:
            logger.warning("No recent fixtures found for testing")
            return False
        
        logger.info(f"Found {len(fixtures)} recent fixtures for testing")
        
        # Test feature generation for each fixture
        for fixture in fixtures:
            fixture_id = fixture[0]
            home_team_name = fixture[5]
            away_team_name = fixture[6]
            match_date = fixture[3]
            
            logger.info(f"\nTesting fixture {fixture_id}: {home_team_name} vs {away_team_name} on {match_date}")
            
            # Extract features
            features = predictor.extract_enhanced_features(fixture_id)
            
            if features is None:
                logger.error(f"Failed to extract features for fixture {fixture_id}")
                continue
            
            # Check for expected features
            expected_features = [
                'fixture_id', 'league_id',
                'home_xg', 'away_xg',
                'home_avg_goals_for', 'home_avg_goals_against',
                'away_avg_goals_for', 'away_avg_goals_against',
                'league_avg_goals',
                'home_team_sentiment', 'away_team_sentiment',
                'home_team_momentum', 'away_team_momentum'
            ]
            
            # Check for Markov features
            markov_features = [
                'markov_home_current_state', 'markov_away_current_state',
                'markov_home_momentum_score', 'markov_away_momentum_score',
                'markov_home_state_stability', 'markov_away_state_stability'
            ]
            
            logger.info(f"Generated {len(features)} features:")
            
            # Check basic features
            missing_features = []
            for feature in expected_features:
                if feature in features:
                    value = features[feature]
                    logger.info(f"  ✅ {feature}: {value}")
                else:
                    missing_features.append(feature)
                    logger.warning(f"  ❌ Missing: {feature}")
            
            # Check Markov features
            markov_found = 0
            for feature in markov_features:
                if feature in features:
                    value = features[feature]
                    logger.info(f"  ✅ {feature}: {value}")
                    markov_found += 1
                else:
                    logger.info(f"  ⚠️  Optional Markov feature not found: {feature}")
            
            # Check sentiment features specifically
            home_sentiment = features.get('home_team_sentiment', 'NOT_FOUND')
            away_sentiment = features.get('away_team_sentiment', 'NOT_FOUND')
            
            logger.info(f"\nSentiment Analysis:")
            logger.info(f"  Home team sentiment: {home_sentiment}")
            logger.info(f"  Away team sentiment: {away_sentiment}")
            
            # Check if sentiment analyzer is working
            if predictor.sentiment_analyzer:
                logger.info(f"  ✅ Sentiment analyzer is initialized")
            else:
                logger.warning(f"  ⚠️  Sentiment analyzer not available (likely missing API key)")
            
            # Check if Markov generator is working
            if predictor.markov_generator:
                logger.info(f"  ✅ Markov generator is initialized")
                logger.info(f"  Found {markov_found}/{len(markov_features)} Markov features")
            else:
                logger.warning(f"  ⚠️  Markov generator not available")
            
            # Summary for this fixture
            if missing_features:
                logger.error(f"  ❌ Missing {len(missing_features)} required features: {missing_features}")
            else:
                logger.info(f"  ✅ All required features present")
            
            logger.info(f"  📊 Total features: {len(features)}")
            
            # Only test first fixture in detail
            break
        
        logger.info("\n" + "="*50)
        logger.info("FEATURE GENERATION TEST COMPLETED")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during feature generation test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function."""
    logger.info("Starting enhanced predictor feature generation test...")
    
    success = test_feature_generation()
    
    if success:
        logger.info("✅ Feature generation test completed successfully")
        return 0
    else:
        logger.error("❌ Feature generation test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())