#!/usr/bin/env python3
"""
Debug script to test MarkovFeatureGenerator methods directly.
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
    """Debug MarkovFeatureGenerator methods."""
    try:
        # Load configuration
        load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize feature generator
        feature_generator = MarkovFeatureGenerator(lookback_window=10)
        logger.info("MarkovFeatureGenerator initialized")
        
        # Get a sample fixture from league 228
        with get_db_session() as session:
            # First check what fixtures are available
            count_query = """
                SELECT COUNT(*) as total,
                       MIN(match_date) as earliest,
                       MAX(match_date) as latest
                FROM fixtures 
                WHERE league_id = 228 
                  AND home_score IS NOT NULL 
                  AND away_score IS NOT NULL
            """
            
            count_result = session.execute(text(count_query))
            count_row = count_result.fetchone()
            logger.info(f"League 228 fixtures with scores: {count_row[0]}, date range: {count_row[1]} to {count_row[2]}")
            
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
            
            # Test generate_team_features for home team
            logger.info("Testing generate_team_features for home team...")
            try:
                home_features = feature_generator.generate_team_features(
                    team_id=home_team_id,
                    league_id=league_id,
                    reference_date=match_date,
                    context='overall'
                )
                
                if home_features:
                    logger.info(f"✅ Home team features generated: {len(home_features)} features")
                    logger.info(f"Sample features: {dict(list(home_features.items())[:5])}")
                else:
                    logger.warning("❌ Home team features returned empty")
                    
            except Exception as e:
                logger.error(f"❌ Error generating home team features: {e}")
                import traceback
                traceback.print_exc()
            
            # Test generate_team_features for away team
            logger.info("Testing generate_team_features for away team...")
            try:
                away_features = feature_generator.generate_team_features(
                    team_id=away_team_id,
                    league_id=league_id,
                    reference_date=match_date,
                    context='overall'
                )
                
                if away_features:
                    logger.info(f"✅ Away team features generated: {len(away_features)} features")
                    logger.info(f"Sample features: {dict(list(away_features.items())[:5])}")
                else:
                    logger.warning("❌ Away team features returned empty")
                    
            except Exception as e:
                logger.error(f"❌ Error generating away team features: {e}")
                import traceback
                traceback.print_exc()
            
            # Test combined generate_features method
            logger.info("Testing combined generate_features method...")
            try:
                combined_features = feature_generator.generate_features(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    match_date=match_date,
                    league_id=league_id
                )
                
                if combined_features:
                    logger.info(f"✅ Combined features generated: {len(combined_features)} features")
                    logger.info(f"Sample features: {dict(list(combined_features.items())[:5])}")
                else:
                    logger.warning("❌ Combined features returned empty")
                    
            except Exception as e:
                logger.error(f"❌ Error generating combined features: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        logger.error(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()