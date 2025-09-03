#!/usr/bin/env python3
"""
Script to populate the pre_computed_features table with enhanced predictor features.

This script extracts features using the EnhancedGoalPredictor for all finished fixtures
and stores them in the pre_computed_features table for consistent training/prediction.
"""

import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from formfinder.database import DatabaseManager, PreComputedFeatures, Fixture
from enhanced_predictor import EnhancedGoalPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('populate_precomputed_features.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PrecomputedFeaturePopulator:
    """Populates pre_computed_features table with enhanced predictor features."""
    
    def __init__(self):
        """Initialize the populator with database connection and enhanced predictor."""
        load_config()  # Load configuration first
        self.config = get_config()
        self.db_manager = DatabaseManager()
        self.engine = self.db_manager.engine
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize enhanced predictor
        try:
            self.enhanced_predictor = EnhancedGoalPredictor()
            logger.info("Enhanced predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced predictor: {e}")
            raise
    
    def get_fixtures_to_process(self, session: Session, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get finished fixtures that need feature computation.
        
        Args:
            session: Database session
            limit: Optional limit on number of fixtures to process
            
        Returns:
            List of fixture dictionaries with required information
        """
        query = """
        SELECT f.id, f.league_id, f.home_team_id, f.away_team_id, f.match_date
        FROM fixtures f
        LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
        WHERE f.status = 'finished' 
        AND f.home_score IS NOT NULL 
        AND f.away_score IS NOT NULL
        AND pcf.fixture_id IS NULL
        ORDER BY f.match_date DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = session.execute(text(query))
        fixtures = [{
            'id': row.id,
            'league_id': row.league_id,
            'home_team_id': row.home_team_id,
            'away_team_id': row.away_team_id,
            'match_date': row.match_date
        } for row in result]
        
        logger.info(f"Found {len(fixtures)} fixtures to process")
        return fixtures
    
    def extract_features_for_fixture(self, fixture_id: int) -> Optional[np.ndarray]:
        """Extract features for a specific fixture using enhanced predictor.
        
        Args:
            fixture_id: ID of the fixture to extract features for
            
        Returns:
            Feature array or None if extraction fails
        """
        try:
            features = self.enhanced_predictor.extract_enhanced_features(fixture_id)
            if features is not None and len(features) > 0:
                logger.debug(f"Extracted {len(features)} features for fixture {fixture_id}")
                return features
            else:
                logger.warning(f"No features extracted for fixture {fixture_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract features for fixture {fixture_id}: {e}")
            return None
    
    def map_features_to_columns(self, features: Dict[str, Any], fixture_info: Dict[str, Any]) -> Dict[str, Any]:
        """Map feature array to PreComputedFeatures table columns.
        
        Args:
            features: Feature array from enhanced predictor
            fixture_info: Fixture information dictionary
            
        Returns:
            Dictionary mapping column names to feature values
        """
        # Initialize with fixture metadata
        feature_dict = {
            'fixture_id': fixture_info['id'],
            'league_id': fixture_info['league_id'],
            'home_team_id': fixture_info['home_team_id'],
            'away_team_id': fixture_info['away_team_id'],
            'match_date': fixture_info['match_date']
        }
        
        # Map features from dictionary to database columns
        # The enhanced predictor returns a dictionary with feature names as keys
        
        def safe_float(value):
            """Safely convert value to float, handling NaN and None."""
            if value is None:
                return None
            try:
                val = float(value)
                return None if np.isnan(val) or np.isinf(val) else val
            except (ValueError, TypeError):
                return None
        
        def safe_int(value):
            """Safely convert value to int, handling NaN and None."""
            if value is None:
                return None
            try:
                val = float(value)
                return None if np.isnan(val) or np.isinf(val) else int(val)
            except (ValueError, TypeError):
                return None
        
        # Map enhanced predictor features to database columns (matching actual PreComputedFeatures model)
        feature_dict.update({
            # Team form features - map to actual database columns
            'home_avg_goals_scored': safe_float(features.get('home_avg_goals_scored', 0.0)),
            'home_avg_goals_conceded': safe_float(features.get('home_avg_goals_conceded', 0.0)),
            'home_avg_goals_scored_home': safe_float(features.get('home_avg_goals_scored_home', 0.0)),
            'home_avg_goals_conceded_home': safe_float(features.get('home_avg_goals_conceded_home', 0.0)),
            'home_form_last_5_games': features.get('home_form_last_5_games', '[]'),
            'home_wins_last_5': safe_int(features.get('home_wins_last_5', 0)),
            'home_draws_last_5': safe_int(features.get('home_draws_last_5', 0)),
            'home_losses_last_5': safe_int(features.get('home_losses_last_5', 0)),
            'home_goals_for_last_5': safe_int(features.get('home_goals_for_last_5', 0)),
            'home_goals_against_last_5': safe_int(features.get('home_goals_against_last_5', 0)),
            
            'away_avg_goals_scored': safe_float(features.get('away_avg_goals_scored', 0.0)),
            'away_avg_goals_conceded': safe_float(features.get('away_avg_goals_conceded', 0.0)),
            'away_avg_goals_scored_away': safe_float(features.get('away_avg_goals_scored_away', 0.0)),
            'away_avg_goals_conceded_away': safe_float(features.get('away_avg_goals_conceded_away', 0.0)),
            'away_form_last_5_games': features.get('away_form_last_5_games', '[]'),
            'away_wins_last_5': safe_int(features.get('away_wins_last_5', 0)),
            'away_draws_last_5': safe_int(features.get('away_draws_last_5', 0)),
            'away_losses_last_5': safe_int(features.get('away_losses_last_5', 0)),
            'away_goals_for_last_5': safe_int(features.get('away_goals_for_last_5', 0)),
            'away_goals_against_last_5': safe_int(features.get('away_goals_against_last_5', 0)),
            
            # Head-to-head features - map to actual database columns
            'h2h_total_matches': safe_int(features.get('h2h_total_matches', 0)),
            'h2h_avg_goals': safe_float(features.get('h2h_avg_goals', 0.0)),
            'h2h_home_wins': safe_int(features.get('h2h_home_wins', 0)),
            'h2h_away_wins': safe_int(features.get('h2h_away_wins', 0)),
            
            # xG features
            'home_xg': safe_float(features.get('home_xg', 0.0)),
            'away_xg': safe_float(features.get('away_xg', 0.0)),
            
            # Team strength and momentum features
            'home_team_strength': safe_float(features.get('home_team_strength', 0.0)),
            'away_team_strength': safe_float(features.get('away_team_strength', 0.0)),
            'home_team_momentum': safe_float(features.get('home_team_momentum', 0.0)),
            'away_team_momentum': safe_float(features.get('away_team_momentum', 0.0)),
            
            # Sentiment features
            'home_team_sentiment': safe_float(features.get('home_team_sentiment', 0.0)),
            'away_team_sentiment': safe_float(features.get('away_team_sentiment', 0.0)),
            
            # Markov features - map from markov_ prefixed features
            'home_team_markov_momentum': safe_float(features.get('markov_home_momentum_score', 0.0)),
            'away_team_markov_momentum': safe_float(features.get('markov_away_momentum_score', 0.0)),
            'home_team_state_stability': safe_float(features.get('markov_home_state_stability', 0.0)),
            'away_team_state_stability': safe_float(features.get('markov_away_state_stability', 0.0)),
            'home_team_transition_entropy': safe_float(features.get('markov_home_transition_entropy', 0.0)),
            'away_team_transition_entropy': safe_float(features.get('markov_away_transition_entropy', 0.0)),
            'home_team_performance_volatility': safe_float(features.get('markov_home_performance_volatility', 0.0)),
            'away_team_performance_volatility': safe_float(features.get('markov_away_performance_volatility', 0.0)),
            'home_team_current_state': features.get('markov_home_current_state', 'unknown'),
            'away_team_current_state': features.get('markov_away_current_state', 'unknown'),
            'home_team_state_duration': safe_int(features.get('markov_home_state_duration', 0)),
            'away_team_state_duration': safe_int(features.get('markov_away_state_duration', 0)),
            'home_team_expected_next_state': features.get('markov_home_expected_next_state', 'unknown'),
            'away_team_expected_next_state': features.get('markov_away_expected_next_state', 'unknown'),
            'home_team_state_confidence': safe_float(features.get('markov_home_state_confidence', 0.0)),
            'away_team_state_confidence': safe_float(features.get('markov_away_state_confidence', 0.0)),
            'markov_match_prediction_confidence': safe_float(features.get('markov_match_prediction_confidence', 0.0)),
            'markov_outcome_probabilities': str(features.get('markov_outcome_probabilities', '{}')),
            
            # Weather and excitement features (use defaults)
            'excitement_rating': safe_float(features.get('excitement_rating', 5.0)),
            'weather_temp_c': safe_float(features.get('weather_temp_c', 21.0)),
            'weather_temp_f': safe_float(features.get('weather_temp_f', 69.8)),
            'weather_humidity': safe_float(features.get('weather_humidity', 50.0)),
            'weather_wind_speed': safe_float(features.get('weather_wind_speed', 5.0)),
            'weather_precipitation': safe_float(features.get('weather_precipitation', 0.0)),
            'weather_condition': features.get('weather_condition', 'Clear'),
            
            # Metadata
            'features_computed_at': datetime.now(),
            'data_quality_score': safe_float(features.get('data_quality_score', 0.8)),
            'computation_source': 'enhanced_predictor'
        })
        
        return feature_dict
    
    def save_precomputed_features(self, session: Session, feature_dict: Dict[str, Any]) -> bool:
        """Save precomputed features to database.
        
        Args:
            session: Database session
            feature_dict: Dictionary of feature values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            precomputed_feature = PreComputedFeatures(**feature_dict)
            session.add(precomputed_feature)
            session.commit()
            logger.debug(f"Saved features for fixture {feature_dict['fixture_id']}")
            return True
        except IntegrityError as e:
            session.rollback()
            logger.warning(f"Features already exist for fixture {feature_dict['fixture_id']}: {e}")
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save features for fixture {feature_dict['fixture_id']}: {e}")
            return False
    
    def process_fixtures(self, limit: Optional[int] = None, batch_size: int = 100) -> Dict[str, int]:
        """Process fixtures and populate precomputed features.
        
        Args:
            limit: Optional limit on number of fixtures to process
            batch_size: Number of fixtures to process in each batch
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_fixtures': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        with self.Session() as session:
            fixtures = self.get_fixtures_to_process(session, limit)
            stats['total_fixtures'] = len(fixtures)
            
            if not fixtures:
                logger.info("No fixtures to process")
                return stats
            
            logger.info(f"Processing {len(fixtures)} fixtures in batches of {batch_size}")
            
            for i in range(0, len(fixtures), batch_size):
                batch = fixtures[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(fixtures)-1)//batch_size + 1} ({len(batch)} fixtures)")
                
                for fixture_info in batch:
                    fixture_id = fixture_info['id']
                    
                    try:
                        # Extract features
                        features = self.extract_features_for_fixture(fixture_id)
                        
                        if features is None:
                            logger.warning(f"Skipping fixture {fixture_id} - no features extracted")
                            stats['skipped'] += 1
                            continue
                        
                        # Map features to database columns
                        feature_dict = self.map_features_to_columns(features, fixture_info)
                        
                        # Save to database
                        if self.save_precomputed_features(session, feature_dict):
                            stats['successful'] += 1
                        else:
                            stats['failed'] += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing fixture {fixture_id}: {e}")
                        logger.error(f"Exception type: {type(e).__name__}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        stats['failed'] += 1
                
                # Log progress
                processed = min(i + batch_size, len(fixtures))
                logger.info(f"Progress: {processed}/{len(fixtures)} fixtures processed")
        
        return stats
    
    def run(self, limit: Optional[int] = None, batch_size: int = 100):
        """Run the feature population process.
        
        Args:
            limit: Optional limit on number of fixtures to process
            batch_size: Number of fixtures to process in each batch
        """
        logger.info("Starting precomputed feature population")
        start_time = datetime.now()
        
        try:
            stats = self.process_fixtures(limit, batch_size)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("Feature population completed")
            logger.info(f"Duration: {duration}")
            logger.info(f"Total fixtures: {stats['total_fixtures']}")
            logger.info(f"Successful: {stats['successful']}")
            logger.info(f"Failed: {stats['failed']}")
            logger.info(f"Skipped: {stats['skipped']}")
            
            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} fixtures failed to process")
            
        except Exception as e:
            logger.error(f"Feature population failed: {e}")
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate pre_computed_features table')
    parser.add_argument('--limit', type=int, help='Limit number of fixtures to process')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        populator = PrecomputedFeaturePopulator()
        populator.run(limit=args.limit, batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()