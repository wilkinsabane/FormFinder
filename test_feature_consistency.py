#!/usr/bin/env python3
"""
Test script to check feature consistency between training and prediction pipelines.
This script identifies discrepancies in feature ordering and naming between
the enhanced predictor and training engine.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.training_engine import TrainingEngine
from enhanced_predictor import EnhancedGoalPredictor
from scripts.database_feature_engine import DatabaseFeatureEngine
from formfinder.config import FormFinderConfig, load_config
from formfinder.database import get_db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sample_fixture_id() -> int:
    """Get a sample fixture ID for testing."""
    engine = get_db_manager().engine
    query = """
    SELECT id FROM fixtures 
    WHERE home_score IS NOT NULL 
        AND away_score IS NOT NULL 
        AND match_date >= '2024-01-01'
    ORDER BY match_date DESC 
    LIMIT 1
    """
    
    result = pd.read_sql_query(query, engine)
    if len(result) == 0:
        raise ValueError("No completed fixtures found for testing")
    
    return int(result.iloc[0]['id'])

def test_enhanced_predictor_features(fixture_id: int) -> Dict:
    """Test enhanced predictor feature extraction."""
    logger.info(f"Testing enhanced predictor features for fixture {fixture_id}")
    
    try:
        predictor = EnhancedGoalPredictor()
        features = predictor.extract_enhanced_features(fixture_id)
        
        if not features:
            logger.error("Enhanced predictor returned no features")
            return {}
        
        # Remove metadata columns
        feature_keys = [key for key in sorted(features.keys()) 
                       if key not in ['fixture_id', 'league_id']]
        
        logger.info(f"Enhanced predictor extracted {len(feature_keys)} features")
        logger.info(f"Feature keys: {feature_keys[:10]}...")  # Show first 10
        
        return {
            'features': features,
            'feature_keys': feature_keys,
            'feature_count': len(feature_keys)
        }
        
    except Exception as e:
        logger.error(f"Error testing enhanced predictor: {e}")
        return {}

def test_training_engine_features(league_id: int = 39) -> Dict:
    """Test training engine feature extraction."""
    logger.info(f"Testing training engine features for league {league_id}")
    
    try:
        config = FormFinderConfig()
        engine = TrainingEngine(config)
        
        # Load enhanced training data
        X_df, y_total, y_over25 = engine.load_enhanced_training_data(league_id)
        
        if X_df.empty:
            logger.error("Training engine returned no features")
            return {}
        
        # Get feature columns from prepared data
        X_prepared, y_prepared, feature_cols = engine.prepare_enhanced_features(X_df, y_total)
        
        logger.info(f"Training engine prepared {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols[:10]}...")  # Show first 10
        
        return {
            'feature_columns': feature_cols,
            'feature_count': len(feature_cols),
            'sample_data': X_prepared.head(1) if not X_prepared.empty else None
        }
        
    except Exception as e:
        logger.error(f"Error testing training engine: {e}")
        return {}

def compare_feature_sets(enhanced_features: List[str], training_features: List[str]) -> Dict:
    """Compare feature sets between enhanced predictor and training engine."""
    logger.info("Comparing feature sets...")
    
    enhanced_set = set(enhanced_features)
    training_set = set(training_features)
    
    # Find differences
    only_in_enhanced = enhanced_set - training_set
    only_in_training = training_set - enhanced_set
    common_features = enhanced_set & training_set
    
    logger.info(f"Common features: {len(common_features)}")
    logger.info(f"Only in enhanced predictor: {len(only_in_enhanced)}")
    logger.info(f"Only in training engine: {len(only_in_training)}")
    
    if only_in_enhanced:
        logger.warning(f"Features only in enhanced predictor: {list(only_in_enhanced)[:10]}")
    
    if only_in_training:
        logger.warning(f"Features only in training engine: {list(only_in_training)[:10]}")
    
    return {
        'common_features': list(common_features),
        'only_in_enhanced': list(only_in_enhanced),
        'only_in_training': list(only_in_training),
        'feature_alignment_score': len(common_features) / max(len(enhanced_set), len(training_set))
    }

def test_feature_ordering(enhanced_features: List[str], training_features: List[str]) -> Dict:
    """Test if feature ordering matches between systems."""
    logger.info("Testing feature ordering...")
    
    # Check if both lists have the same features in the same order
    if enhanced_features == training_features:
        logger.info("✓ Feature ordering is identical")
        return {'ordering_match': True, 'differences': []}
    
    # Find ordering differences
    differences = []
    min_len = min(len(enhanced_features), len(training_features))
    
    for i in range(min_len):
        if enhanced_features[i] != training_features[i]:
            differences.append({
                'position': i,
                'enhanced': enhanced_features[i],
                'training': training_features[i]
            })
    
    logger.warning(f"✗ Feature ordering differs at {len(differences)} positions")
    if differences:
        logger.warning(f"First few differences: {differences[:5]}")
    
    return {
        'ordering_match': False,
        'differences': differences,
        'difference_count': len(differences)
    }

def main():
    """Main test function."""
    logger.info("Starting feature consistency test...")
    
    try:
        # Load configuration first
        load_config()
        
        # Get sample fixture for testing
        fixture_id = get_sample_fixture_id()
        logger.info(f"Using fixture {fixture_id} for testing")
        
        # Test enhanced predictor
        enhanced_result = test_enhanced_predictor_features(fixture_id)
        if not enhanced_result:
            logger.error("Failed to test enhanced predictor")
            return
        
        # Test training engine
        training_result = test_training_engine_features()
        if not training_result:
            logger.error("Failed to test training engine")
            return
        
        # Compare feature sets
        comparison = compare_feature_sets(
            enhanced_result['feature_keys'],
            training_result['feature_columns']
        )
        
        # Test feature ordering
        ordering_test = test_feature_ordering(
            enhanced_result['feature_keys'],
            training_result['feature_columns']
        )
        
        # Generate summary report
        logger.info("\n" + "="*50)
        logger.info("FEATURE CONSISTENCY TEST SUMMARY")
        logger.info("="*50)
        logger.info(f"Enhanced predictor features: {enhanced_result['feature_count']}")
        logger.info(f"Training engine features: {training_result['feature_count']}")
        logger.info(f"Feature alignment score: {comparison['feature_alignment_score']:.2%}")
        logger.info(f"Feature ordering match: {ordering_test['ordering_match']}")
        
        if comparison['feature_alignment_score'] < 0.9:
            logger.error("⚠️  LOW FEATURE ALIGNMENT - This may cause prediction errors!")
        
        if not ordering_test['ordering_match']:
            logger.error("⚠️  FEATURE ORDERING MISMATCH - This will cause prediction errors!")
        
        if comparison['feature_alignment_score'] >= 0.9 and ordering_test['ordering_match']:
            logger.info("✅ Feature consistency test PASSED")
        else:
            logger.error("❌ Feature consistency test FAILED")
        
        return {
            'enhanced_result': enhanced_result,
            'training_result': training_result,
            'comparison': comparison,
            'ordering_test': ordering_test
        }
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()