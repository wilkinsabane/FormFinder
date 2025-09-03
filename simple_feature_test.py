#!/usr/bin/env python3
"""Simple feature consistency test without complex configuration."""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from enhanced_predictor import EnhancedGoalPredictor
from scripts.database_feature_engine import DatabaseFeatureEngine
from formfinder.config import load_config, get_config

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sample_fixture_id():
    """Get a sample fixture ID from the database"""
    try:
        # Load configuration and get database connection
        load_config()
        config = get_config()
        
        # Create PostgreSQL connection
        db_url = f"postgresql://{config.database.postgresql.username}:{config.database.postgresql.password}@{config.database.postgresql.host}:{config.database.postgresql.port}/{config.database.postgresql.database}"
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get a sample fixture with scores
        query = text("""
            SELECT id FROM fixtures 
            WHERE home_score IS NOT NULL 
            AND away_score IS NOT NULL 
            AND status = 'finished'
            LIMIT 1
        """)
        
        result = session.execute(query).fetchone()
        session.close()
        
        if result:
            return result[0]
        else:
            print("No finished fixtures found in database")
            return None
            
    except Exception as e:
        print(f"Error getting sample fixture: {e}")
        return None

def test_enhanced_predictor_features():
    """Test enhanced predictor feature extraction."""
    try:
        from enhanced_predictor import EnhancedGoalPredictor
        
        # Create predictor instance
        predictor = EnhancedGoalPredictor()
        
        # Get a sample fixture ID from database
        fixture_id = get_sample_fixture_id()
        
        if not fixture_id:
            logger.error("No completed fixtures found in database")
            return None
            
        logger.info(f"Testing with fixture ID: {fixture_id}")
        
        # Extract features using enhanced predictor
        features = predictor.extract_enhanced_features(fixture_id)
        
        if features:
            logger.info(f"Enhanced predictor extracted {len(features)} features")
            logger.info(f"Feature names: {list(features.keys())[:10]}...")  # Show first 10
            return features
        else:
            logger.error("Enhanced predictor returned no features")
            return None
                
    except Exception as e:
        logger.error(f"Enhanced predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_database_features():
    """Test database feature extraction."""
    try:
        # Load configuration and create PostgreSQL connection
        load_config()
        config = get_config()
        
        db_url = f"postgresql://{config.database.postgresql.username}:{config.database.postgresql.password}@{config.database.postgresql.host}:{config.database.postgresql.port}/{config.database.postgresql.database}"
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        feature_engine = DatabaseFeatureEngine(session)
        
        # Get sample fixture
        fixture_id = get_sample_fixture_id()
        
        if not fixture_id:
            logger.error("No completed fixtures found in database")
            return None
        
        # Extract features using database feature engine - use load_prediction_features for single fixture
        features_df = feature_engine.load_prediction_features([fixture_id])
        
        if features_df is not None and not features_df.empty:
            # Convert to numpy array or list for comparison
            features = features_df.iloc[0].drop(['fixture_id'], errors='ignore').values
            session.close()
            logger.info(f"Database feature engine extracted {len(features)} features")
            logger.info(f"Feature shape: {features.shape if hasattr(features, 'shape') else 'N/A'}")
            logger.info(f"Feature columns: {list(features_df.columns)}")
            return features
        else:
            session.close()
            logger.error("Database feature engine returned no features")
            return None
                
    except Exception as e:
        logger.error(f"Database feature engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_features(enhanced_features, db_features):
    """Compare the two feature sets."""
    logger.info("\n=== Feature Comparison ===")
    
    if enhanced_features is None or db_features is None:
        logger.error("Cannot compare - one or both feature sets are None")
        return
    
    # Enhanced features are dict, db features might be array
    if isinstance(enhanced_features, dict):
        enhanced_count = len(enhanced_features)
        enhanced_names = list(enhanced_features.keys())
    else:
        enhanced_count = len(enhanced_features) if hasattr(enhanced_features, '__len__') else 0
        enhanced_names = []
    
    if hasattr(db_features, 'shape'):
        db_count = db_features.shape[0] if len(db_features.shape) > 0 else 0
    elif hasattr(db_features, '__len__'):
        db_count = len(db_features)
    else:
        db_count = 0
    
    logger.info(f"Enhanced predictor features: {enhanced_count}")
    logger.info(f"Database features: {db_count}")
    
    if enhanced_names:
        logger.info(f"Sample enhanced feature names: {enhanced_names[:10]}")
    
    # Check for feature alignment issues
    if enhanced_count != db_count:
        logger.warning(f"Feature count mismatch: {enhanced_count} vs {db_count}")
        logger.warning("This indicates a feature alignment issue between training and prediction")
    else:
        logger.info("Feature counts match!")

def main():
    """Main test function."""
    logger.info("Starting simple feature consistency test...")
    
    try:
        # Test enhanced predictor features
        logger.info("\n=== Testing Enhanced Predictor ===")
        enhanced_features = test_enhanced_predictor_features()
        
        # Test database features
        logger.info("\n=== Testing Database Feature Engine ===")
        db_features = test_database_features()
        
        # Compare features
        compare_features(enhanced_features, db_features)
        
        logger.info("\n=== Test Complete ===")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()