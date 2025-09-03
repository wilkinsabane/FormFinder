#!/usr/bin/env python3
"""
Test H2H feature computation with real fixtures to verify the fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import DatabaseManager
from formfinder.enhanced_feature_computer import EnhancedFeatureComputer
from sqlalchemy import text
import logging
from datetime import datetime

# Load configuration
load_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_h2h_computation():
    """Test H2H feature computation with real fixtures."""
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
        logger.info(f"League: {league_id}")
        
        # Initialize enhanced feature computer
        feature_computer = EnhancedFeatureComputer(session)
        
        # Test H2H computation
        logger.info("\n=== Computing Enhanced Features ===")
        try:
            # Use the enhanced predictor to extract features
            from enhanced_predictor import EnhancedGoalPredictor
            predictor = EnhancedGoalPredictor()
            features = predictor.extract_enhanced_features(fixture_id)
            
            # Extract H2H features
            h2h_features = {k: v for k, v in features.items() if k.startswith('h2h_')}
            
            logger.info(f"\n=== H2H Features ===")
            for key, value in h2h_features.items():
                logger.info(f"{key}: {value}")
            
            # Check if H2H features show real data (not defaults)
            if h2h_features.get('h2h_total_matches', 0) > 0:
                logger.info("\n✅ SUCCESS: H2H computation is working! Found historical matches.")
                logger.info(f"Total H2H matches: {h2h_features.get('h2h_total_matches')}")
                logger.info(f"Average total goals: {h2h_features.get('h2h_avg_total_goals')}")
                logger.info(f"Home wins: {h2h_features.get('h2h_home_wins')}")
                logger.info(f"Away wins: {h2h_features.get('h2h_away_wins')}")
            else:
                logger.warning("⚠️  H2H computation returned default values (0 matches)")
            
            # Test a few more features to ensure overall computation works
            logger.info(f"\n=== Other Key Features ===")
            logger.info(f"Weather temperature: {features.get('weather_temperature')}")
            logger.info(f"Home recent form: {features.get('home_recent_form')}")
            logger.info(f"Away recent form: {features.get('away_recent_form')}")
            
        except Exception as e:
            logger.error(f"❌ ERROR: Feature computation failed: {str(e)}")
            import traceback
            traceback.print_exc()

def test_multiple_fixtures():
    """Test H2H computation on multiple fixtures to verify consistency."""
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # Get fixtures from teams with known H2H history
        test_fixtures_query = text("""
            SELECT id, home_team_id, away_team_id, match_date, league_id
            FROM fixtures
            WHERE ((home_team_id IN (3057, 3528, 3640, 3806) AND away_team_id IN (3057, 3528, 3640, 3806))
                   AND home_team_id != away_team_id)
                AND status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 5
        """)
        
        test_fixtures = session.execute(test_fixtures_query).fetchall()
        
        logger.info(f"\n=== Testing {len(test_fixtures)} Fixtures ===")
        
        feature_computer = EnhancedFeatureComputer(session)
        success_count = 0
        
        for fixture_id, home_team_id, away_team_id, match_date, league_id in test_fixtures:
            try:
                # Use the enhanced predictor to extract features
                from enhanced_predictor import EnhancedGoalPredictor
                predictor = EnhancedGoalPredictor()
                features = predictor.extract_enhanced_features(fixture_id)
                
                h2h_matches = features.get('h2h_total_matches', 0)
                logger.info(f"Fixture {fixture_id} ({home_team_id} vs {away_team_id}): {h2h_matches} H2H matches")
                
                if h2h_matches > 0:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to compute features for fixture {fixture_id}: {str(e)}")
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Successful H2H computations: {success_count}/{len(test_fixtures)}")
        if success_count > 0:
            logger.info("✅ H2H computation is working correctly!")
        else:
            logger.warning("⚠️  No successful H2H computations found")

if __name__ == "__main__":
    test_h2h_computation()
    test_multiple_fixtures()