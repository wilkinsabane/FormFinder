#!/usr/bin/env python3
"""
Validation script to test that both weather and H2H data show proper variation after fixes.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_predictor import EnhancedGoalPredictor
from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test data variation across multiple fixtures."""
    # Load configuration
    load_config()
    
    predictor = EnhancedGoalPredictor()
    
    with get_db_session() as db_session:
        # Get a sample of recent fixtures (both scheduled and completed)
        query = text("""
            SELECT id, home_team_id, away_team_id, match_date, league_id
            FROM fixtures 
            WHERE match_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY match_date DESC
            LIMIT 10
        """)
        
        fixtures = db_session.execute(query).fetchall()
        logger.info(f"Testing data variation across {len(fixtures)} fixtures")
        
        # Track unique values
        weather_values = defaultdict(set)
        h2h_values = defaultdict(set)
        
        successful_extractions = 0
        
        for fixture in fixtures:
            fixture_id, home_team_id, away_team_id, match_date, league_id = fixture
            
            try:
                logger.info(f"\n=== Testing Fixture {fixture_id} ({home_team_id} vs {away_team_id}) ===")
                
                # Extract features
                features = predictor.extract_enhanced_features(fixture_id)
                
                if features:
                    successful_extractions += 1
                    
                    # Collect weather data
                    weather_features = {
                        'temp_c': features.get('weather_temp_c'),
                        'humidity': features.get('weather_humidity'),
                        'wind_speed': features.get('weather_wind_speed'),
                        'condition': features.get('weather_condition')
                    }
                    
                    # Collect H2H data
                    h2h_features = {
                        'total_matches': features.get('h2h_total_matches'),
                        'home_wins': features.get('h2h_home_wins'),
                        'away_wins': features.get('h2h_away_wins'),
                        'draws': features.get('h2h_draws'),
                        'avg_goals': features.get('h2h_avg_goals')
                    }
                    
                    # Track unique values
                    for key, value in weather_features.items():
                        if value is not None:
                            weather_values[key].add(value)
                    
                    for key, value in h2h_features.items():
                        if value is not None:
                            h2h_values[key].add(value)
                    
                    # Log current fixture data
                    logger.info(f"Weather: {weather_features}")
                    logger.info(f"H2H: {h2h_features}")
                    
            except Exception as e:
                logger.error(f"Error processing fixture {fixture_id}: {e}")
        
        # Analyze variation
        logger.info(f"\n=== DATA VARIATION ANALYSIS ===")
        logger.info(f"Successfully processed: {successful_extractions}/{len(fixtures)} fixtures")
        
        logger.info(f"\n--- Weather Data Variation ---")
        for key, values in weather_values.items():
            logger.info(f"{key}: {len(values)} unique values - {sorted(list(values))[:5]}{'...' if len(values) > 5 else ''}")
        
        logger.info(f"\n--- H2H Data Variation ---")
        for key, values in h2h_values.items():
            logger.info(f"{key}: {len(values)} unique values - {sorted(list(values))[:5]}{'...' if len(values) > 5 else ''}")
        
        # Validation results
        weather_variation_good = all(len(values) > 1 for values in weather_values.values() if len(values) > 0)
        h2h_variation_good = any(len(values) > 1 for values in h2h_values.values() if len(values) > 0)
        
        logger.info(f"\n=== VALIDATION RESULTS ===")
        logger.info(f"Weather data shows variation: {'✅ YES' if weather_variation_good else '❌ NO'}")
        logger.info(f"H2H data shows variation: {'✅ YES' if h2h_variation_good else '❌ NO'}")
        
        if weather_variation_good and h2h_variation_good:
            logger.info(f"🎉 SUCCESS: Both weather and H2H data show proper variation!")
        else:
            logger.warning(f"⚠️  Some data still shows limited variation")

if __name__ == "__main__":
    main()