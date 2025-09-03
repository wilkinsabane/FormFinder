#!/usr/bin/env python3
"""
Test script to verify store_team_features method.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from formfinder.logger import get_logger
from sqlalchemy import text

logger = get_logger(__name__)

def main():
    """Test store_team_features method."""
    try:
        # Load configuration
        load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize feature generator
        feature_generator = MarkovFeatureGenerator(lookback_window=10)
        logger.info("MarkovFeatureGenerator initialized")
        
        # Get a sample fixture from league 228
        with get_db_session() as session:
            fixture_query = """
                SELECT id, home_team_id, away_team_id, match_date, league_id
                FROM fixtures 
                WHERE league_id = 228 
                  AND home_score IS NOT NULL 
                  AND away_score IS NOT NULL
                ORDER BY match_date
                LIMIT 1
            """
            
            result = session.execute(text(fixture_query))
            fixture = result.fetchone()
            
            if not fixture:
                logger.error("No fixture found for testing")
                return
            
            fixture_id, home_team_id, away_team_id, match_date, league_id = fixture
            logger.info(f"Testing with fixture: {fixture_id}, teams: {home_team_id} vs {away_team_id}, date: {match_date}")
            
            # Generate features for home team
            logger.info("Generating features for home team...")
            home_features = feature_generator.generate_team_features(
                team_id=home_team_id,
                league_id=league_id,
                reference_date=match_date,
                context='overall'
            )
            
            if home_features:
                logger.info(f"Home team features generated: {len(home_features)} features")
                
                # Try to store features
                logger.info("Attempting to store home team features...")
                try:
                    stored_record = feature_generator.store_team_features(
                        team_id=home_team_id,
                        league_id=league_id,
                        features=home_features,
                        fixture_id=fixture_id
                    )
                    
                    if stored_record:
                        logger.info(f"SUCCESS: Home team features stored successfully")
                        
                        # Verify the record was actually stored by querying the database
                        verify_query = """
                            SELECT COUNT(*) FROM markov_features 
                            WHERE team_id = :team_id 
                              AND league_id = :league_id 
                              AND fixture_id = :fixture_id
                        """
                        
                        count_result = session.execute(text(verify_query), {
                            'team_id': home_team_id,
                            'league_id': league_id,
                            'fixture_id': fixture_id
                        })
                        
                        count = count_result.scalar()
                        logger.info(f"Verification: {count} records found in database")
                        
                    else:
                        logger.error("FAILED: store_team_features returned None")
                        
                except Exception as e:
                    logger.error(f"FAILED: Error storing home team features: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.error("No home team features generated")
                
    except Exception as e:
        logger.error(f"Test script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()