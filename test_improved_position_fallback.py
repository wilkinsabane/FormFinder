#!/usr/bin/env python3
"""
Test the improved position fallback method to verify it reduces warning noise.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.enhanced_feature_computer import EnhancedFeatureComputer
from datetime import datetime
import logging

def test_position_fallback():
    """Test the improved position fallback method."""
    load_config()
    
    # Set up logging to capture warnings
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    
    with get_db_session() as session:
        computer = EnhancedFeatureComputer(session)
        
        print("=== Testing Position Fallback for Teams 3374 and 5159 ===")
        
        # Test the specific teams that were causing warnings
        test_cases = [
            (3374, 311, "Al Duhail in AFC Champions League"),
            (5159, 311, "Sepahan in AFC Champions League"),
            (9999, 311, "Non-existent team in AFC Champions League"),
            (3374, 9999, "Al Duhail in non-existent league")
        ]
        
        match_date = datetime(2025, 1, 15)
        
        for team_id, league_id, description in test_cases:
            print(f"\nTesting: {description}")
            print(f"Team ID: {team_id}, League ID: {league_id}")
            
            try:
                position, confidence = computer._get_team_position(team_id, league_id, match_date)
                print(f"Result: Position {position}, Confidence {confidence}")
                
                # Test calling it again to verify warning is only logged once
                position2, confidence2 = computer._get_team_position(team_id, league_id, match_date)
                print(f"Second call: Position {position2}, Confidence {confidence2}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n=== Testing League Classification ===")
        
        # Test different league types
        league_test_cases = [
            (311, "AFC Champions League"),  # Should be detected as cup
            (39, "Premier League"),  # Should be regular league
            (310, "UEFA Champions League"),  # Should be detected as cup
        ]
        
        for league_id, expected_name in league_test_cases:
            print(f"\nTesting league classification: {expected_name} (ID: {league_id})")
            
            # Use a dummy team ID that likely doesn't exist
            position, confidence = computer._get_team_position(99999, league_id, match_date)
            print(f"Position: {position}, Confidence: {confidence}")
            
            if confidence == 0.3:
                print("  -> Detected as cup competition")
            elif confidence == 0.0:
                print("  -> Treated as regular league without standings")
            else:
                print("  -> Has standings data")

if __name__ == "__main__":
    test_position_fallback()