#!/usr/bin/env python3
"""
Debug script to examine H2H features in detail
"""

import logging
from formfinder.config import load_config
from formfinder.database import DatabaseManager
from enhanced_predictor import EnhancedGoalPredictor
from sqlalchemy import text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_h2h_features():
    """Debug H2H features in detail."""
    load_config()
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # Find a fixture with known H2H history (teams 3057 vs 3528)
        test_fixture_query = text("""
            SELECT id, home_team_id, away_team_id, match_date, league_id
            FROM fixtures
            WHERE ((home_team_id = 3057 AND away_team_id = 3528)
                   OR (home_team_id = 3528 AND away_team_id = 3057))
                AND status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 1
        """)
        
        test_fixture = session.execute(test_fixture_query).fetchone()
        
        if not test_fixture:
            logger.error("No test fixture found")
            return
        
        fixture_id, home_team_id, away_team_id, match_date, league_id = test_fixture
        logger.info(f"=== Testing Fixture {fixture_id} ===")
        logger.info(f"Teams: {home_team_id} vs {away_team_id}")
        logger.info(f"Date: {match_date}")
        
        # Initialize enhanced predictor
        predictor = EnhancedGoalPredictor()
        
        # Extract features
        logger.info("\n=== Extracting Features ===")
        features = predictor.extract_enhanced_features(fixture_id)
        
        if not features:
            logger.error("No features extracted")
            return
        
        logger.info(f"Total features extracted: {len(features)}")
        
        # Look for H2H-related features
        logger.info("\n=== H2H-Related Features ===")
        h2h_features = {k: v for k, v in features.items() if 'h2h' in k.lower()}
        
        if h2h_features:
            for key, value in h2h_features.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.warning("No H2H features found in extracted features")
        
        # Look for other potentially relevant features
        logger.info("\n=== Other Potentially Relevant Features ===")
        relevant_keywords = ['match', 'history', 'head', 'previous', 'past']
        other_features = {k: v for k, v in features.items() 
                         if any(keyword in k.lower() for keyword in relevant_keywords)}
        
        if other_features:
            for key, value in other_features.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("No other relevant features found")
        
        # Show all feature keys for reference
        logger.info("\n=== All Feature Keys ===")
        all_keys = sorted(features.keys())
        for i, key in enumerate(all_keys):
            logger.info(f"  {i+1:2d}. {key}")
        
        # Test direct H2H query to verify data exists
        logger.info("\n=== Direct H2H Query Test ===")
        h2h_query = text("""
            SELECT COUNT(*) as total_matches
            FROM fixtures
            WHERE ((home_team_id = :home_team_id AND away_team_id = :away_team_id)
                   OR (home_team_id = :away_team_id AND away_team_id = :home_team_id))
                AND status = 'finished'
                AND match_date < :match_date
        """)
        
        h2h_result = session.execute(h2h_query, {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'match_date': match_date
        }).fetchone()
        
        logger.info(f"Direct H2H query result: {h2h_result[0]} matches found")

if __name__ == "__main__":
    debug_h2h_features()