#!/usr/bin/env python3
"""
Debug script to test MarkovFeatureGenerator directly.
"""

import sys
sys.path.append('.')

# Load configuration first
from formfinder.config import load_config
load_config()

from formfinder.markov_feature_generator import MarkovFeatureGenerator
from formfinder.database import get_db_session
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_markov_generator():
    """Test MarkovFeatureGenerator directly."""
    
    print("Testing MarkovFeatureGenerator directly")
    print("=" * 50)
    
    try:
        # Initialize generator
        with get_db_session() as session:
            generator = MarkovFeatureGenerator(session, lookback_window=10)
            
            # Test teams and reference date (using teams with good performance state data)
            home_team_id = 3104  # Team with 28 states, excellent recent performance
            away_team_id = 3068  # Team with 26 states, mixed recent performance
            league_id = 224  # Czech Liga (correct league for these teams)
            reference_date = datetime(2025, 6, 1, 18, 0)  # More recent date to capture their states
            
            print(f"Testing teams: {home_team_id} (home) vs {away_team_id} (away)")
            print(f"League: {league_id}, Date: {reference_date}")
            print()
            
            # Test home team state info
            print("=== HOME TEAM STATE INFO ===")
            print(f"Calling get_current_state_info({home_team_id}, {league_id}, {reference_date}, 'home')")
            home_state_info = generator.get_current_state_info(
                home_team_id, league_id, reference_date, 'home'
            )
            print(f"Home team state info: {home_state_info}")
            
            # Test away team state info
            print("\n=== AWAY TEAM STATE INFO ===")
            print(f"Calling get_current_state_info({away_team_id}, {league_id}, {reference_date}, 'away')")
            away_state_info = generator.get_current_state_info(
                away_team_id, league_id, reference_date, 'away'
            )
            print(f"Away team state info: {away_state_info}")
            
            # Test overall context (using 'home' and 'away' contexts instead of 'overall')
            print("\n=== OVERALL CONTEXT ===")
            home_overall = generator.get_current_state_info(
                home_team_id, league_id, reference_date, 'home'
            )
            away_overall = generator.get_current_state_info(
                away_team_id, league_id, reference_date, 'away'
            )
            print(f"Home team overall: {home_overall}")
            print(f"Away team overall: {away_overall}")
            
            # Test team features generation
            print("\n=== TEAM FEATURES GENERATION ===")
            home_features = generator.generate_team_features(
                home_team_id, league_id, reference_date, 'home'
            )
            print(f"Home team features: {home_features}")
            
            away_features = generator.generate_team_features(
                away_team_id, league_id, reference_date, 'away'
            )
            print(f"Away team features: {away_features}")
            
            # Test full feature generation
            print("\n=== FULL FEATURE GENERATION ===")
            all_features = generator.generate_features(
                home_team_id, away_team_id, league_id, reference_date
            )
            print(f"All features: {all_features}")
            
            # Check if state classifier is working
            print("\n=== STATE CLASSIFIER TEST ===")
            if hasattr(generator, 'state_classifier'):
                print(f"State classifier available: {generator.state_classifier}")
                try:
                    metrics = generator.state_classifier._calculate_performance_score_with_session(
                        session, home_team_id, league_id, reference_date, 'home'
                    )
                    print(f"Home team performance metrics: {metrics}")
                    
                    if metrics['matches_analyzed'] >= generator.state_classifier.min_matches:
                        calculated_state = generator.state_classifier.classify_state(metrics['performance_score'])
                        print(f"Calculated home team state: {calculated_state}")
                    else:
                        print(f"Not enough matches for home team: {metrics['matches_analyzed']} < {generator.state_classifier.min_matches}")
                        
                except Exception as e:
                    print(f"Error calculating state: {e}")
            else:
                print("No state classifier available")
            
    except Exception as e:
        print(f"Error testing MarkovFeatureGenerator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_markov_generator()