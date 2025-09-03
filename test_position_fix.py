#!/usr/bin/env python3
"""
Test script to verify that position values are correctly computed
"""

import sys
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import get_db_session
from enhanced_predictor import EnhancedGoalPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_position_extraction():
    """Test position extraction for a specific fixture"""
    load_config()
    
    with get_db_session() as session:
        # Test with fixture 5014 (Atletico Choloma vs Policia NFC)
        fixture_id = 5014
        home_team_id = None
        away_team_id = None
        match_date = None
        league_id = None
        
        # Get fixture details
        fixture_query = """
            SELECT home_team_id, away_team_id, match_date, league_id
            FROM fixtures 
            WHERE id = :fixture_id
        """
        from sqlalchemy import text
        result = session.execute(text(fixture_query), {"fixture_id": fixture_id})
        fixture_row = result.fetchone()
        
        if not fixture_row:
            print(f"Fixture {fixture_id} not found")
            return
            
        home_team_id, away_team_id, match_date, league_id = fixture_row
        print(f"Testing fixture {fixture_id}: home_team={home_team_id}, away_team={away_team_id}, date={match_date}, league={league_id}")
        
        # Initialize predictor
        predictor = EnhancedGoalPredictor()
        
        # Test the _get_team_position method directly
        home_position = predictor._get_team_position(home_team_id, league_id, match_date)
        away_position = predictor._get_team_position(away_team_id, league_id, match_date)
        
        print(f"Direct position lookup:")
        print(f"  Home team position: {home_position}")
        print(f"  Away team position: {away_position}")
        
        # Test feature extraction with position values
        try:
            features = predictor.extract_enhanced_features(fixture_id)
            
            if features:
                print(f"\nFull feature extraction:")
                print(f"  Home team position: {features.get('home_team_position', 'Not found')}")
                print(f"  Away team position: {features.get('away_team_position', 'Not found')}")
                print(f"  Total features extracted: {len(features)}")
            else:
                print("No features extracted")
                
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_position_extraction()