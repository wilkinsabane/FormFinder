#!/usr/bin/env python3
"""
Test script to debug current state calculation in MarkovFeatureGenerator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

# Load configuration
load_config()

def test_current_state_calculation():
    """Test current state calculation for specific teams."""
    
    # Initialize MarkovFeatureGenerator
    markov_generator = MarkovFeatureGenerator()
    
    # Get some team IDs from pre_computed_features
    with get_db_session() as session:
        query = """
            SELECT DISTINCT home_team_id, away_team_id, league_id, match_date
            FROM pre_computed_features
            LIMIT 3
        """
        
        results = session.execute(text(query)).fetchall()
        
        for result in results:
            home_team_id = result[0]
            away_team_id = result[1]
            league_id = result[2]
            match_date = result[3]
            
            print(f"\n=== Testing teams {home_team_id} vs {away_team_id} in league {league_id} ===")
            print(f"Match date: {match_date}")
            
            # Test home team state calculation
            print(f"\n--- Home team {home_team_id} ---")
            try:
                home_state_info = markov_generator.get_current_state_info(
                    home_team_id, league_id, match_date, 'home'
                )
                print(f"Home state info: {home_state_info}")
                
                # Also test the state classifier directly
                metrics = markov_generator.state_classifier.calculate_performance_score(
                    home_team_id, league_id, match_date, 'home'
                )
                print(f"Home team metrics: {metrics}")
                
                if metrics['matches_analyzed'] >= markov_generator.state_classifier.min_matches:
                    calculated_state = markov_generator.state_classifier.classify_state(metrics['performance_score'])
                    print(f"Calculated home state: {calculated_state}")
                else:
                    print(f"Not enough matches for home team: {metrics['matches_analyzed']}")
                    
            except Exception as e:
                print(f"Error calculating home state: {e}")
            
            # Test away team state calculation
            print(f"\n--- Away team {away_team_id} ---")
            try:
                away_state_info = markov_generator.get_current_state_info(
                    away_team_id, league_id, match_date, 'away'
                )
                print(f"Away state info: {away_state_info}")
                
                # Also test the state classifier directly
                metrics = markov_generator.state_classifier.calculate_performance_score(
                    away_team_id, league_id, match_date, 'away'
                )
                print(f"Away team metrics: {metrics}")
                
                if metrics['matches_analyzed'] >= markov_generator.state_classifier.min_matches:
                    calculated_state = markov_generator.state_classifier.classify_state(metrics['performance_score'])
                    print(f"Calculated away state: {calculated_state}")
                else:
                    print(f"Not enough matches for away team: {metrics['matches_analyzed']}")
                    
            except Exception as e:
                print(f"Error calculating away state: {e}")

if __name__ == "__main__":
    test_current_state_calculation()