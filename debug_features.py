#!/usr/bin/env python3
"""
Debug script to check what features are actually returned by enhanced predictor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_predictor import EnhancedGoalPredictor
from formfinder.database import DatabaseManager
from formfinder.config import load_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_enhanced_features():
    """Debug what features are actually returned by enhanced predictor."""
    try:
        # Load config and initialize
        load_config()
        db_manager = DatabaseManager()
        predictor = EnhancedGoalPredictor()
        
        # Get a test fixture
        with db_manager.get_session() as session:
            from formfinder.database import Fixture
            fixture = session.query(Fixture).filter(
                Fixture.status == 'finished'
            ).first()
            
            if not fixture:
                logger.error("No finished fixtures found")
                return
                
            logger.info(f"Testing with fixture {fixture.id}")
            
            # Extract features
            features = predictor.extract_enhanced_features(fixture.id)
            
            if features:
                logger.info(f"\n=== ENHANCED PREDICTOR FEATURES ({len(features)} total) ===")
                for key, value in sorted(features.items()):
                    logger.info(f"{key}: {value}")
                    
                logger.info("\n=== MISSING FEATURES THAT MAPPING EXPECTS ===")
                expected_features = [
                    'home_team_momentum', 'away_team_momentum',
                    'home_team_sentiment', 'away_team_sentiment',
                    'markov_home_momentum_score', 'markov_away_momentum_score',
                    'markov_home_state_stability', 'markov_away_state_stability',
                    'markov_home_transition_entropy', 'markov_away_transition_entropy',
                    'markov_home_performance_volatility', 'markov_away_performance_volatility',
                    'markov_home_current_state', 'markov_away_current_state',
                    'markov_home_state_duration', 'markov_away_state_duration',
                    'markov_home_expected_next_state', 'markov_away_expected_next_state',
                    'markov_home_state_confidence', 'markov_away_state_confidence',
                    'markov_match_prediction_confidence', 'markov_outcome_probabilities'
                ]
                
                missing_features = []
                for expected in expected_features:
                    if expected not in features:
                        missing_features.append(expected)
                        
                for missing in missing_features:
                    logger.info(f"MISSING: {missing}")
                    
                logger.info(f"\nSUMMARY: {len(missing_features)} features missing out of {len(expected_features)} expected")
                
            else:
                logger.error("No features returned")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_enhanced_features()