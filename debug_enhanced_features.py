#!/usr/bin/env python3
"""
Debug script to check what enhanced features are being computed for a specific fixture.
"""

import sys
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.enhanced_feature_computer import EnhancedFeatureComputer
from sqlalchemy import text
import json

def main():
    config = load_config()
    
    with get_db_session() as session:
        # Get fixture 20819 details
        fixture_query = text("""
        SELECT id, home_team_id, away_team_id, match_date, league_id
        FROM fixtures 
        WHERE id = 20819
        """)
        
        fixture = session.execute(fixture_query).fetchone()
        if not fixture:
            print("Fixture 20819 not found")
            return
            
        print(f"Fixture Details:")
        print(f"  ID: {fixture.id}")
        print(f"  Home Team: {fixture.home_team_id}")
        print(f"  Away Team: {fixture.away_team_id}")
        print(f"  Match Date: {fixture.match_date}")
        print(f"  League: {fixture.league_id}")
        print()
        
        # Initialize enhanced feature computer
        feature_computer = EnhancedFeatureComputer(session)
        
        # Compute enhanced features
        print("Computing enhanced features...")
        enhanced_features = feature_computer.compute_all_features(fixture.id)
        
        print(f"\nEnhanced features computed: {len(enhanced_features)} features")
        
        # Check position-related features specifically
        position_features = {
            k: v for k, v in enhanced_features.items() 
            if 'position' in k.lower()
        }
        
        print(f"\nPosition-related features:")
        for key, value in position_features.items():
            print(f"  {key}: {value}")
            
        # Also check if the _compute_position_features method is being called
        print(f"\nAll enhanced features:")
        for key, value in sorted(enhanced_features.items()):
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()