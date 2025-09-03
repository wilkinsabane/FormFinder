#!/usr/bin/env python3
"""
Debug script to test MarkovFeatureGenerator with reduced minimum matches requirement.
"""

import sys
sys.path.append('.')

from datetime import datetime
from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from formfinder.markov_state_classifier import MarkovStateClassifier

def test_markov_generator_with_min_matches():
    """Test MarkovFeatureGenerator with reduced minimum matches."""
    try:
        # Load configuration
        load_config()
        
        with get_db_session() as session:
            # Create MarkovFeatureGenerator and modify its state classifier
            generator = MarkovFeatureGenerator(session)
            # Reduce minimum matches requirement for testing
            generator.state_classifier.min_matches = 1
            
            # Test parameters
            home_team_id = 2962  # Sparta Prague
            away_team_id = 3031  # Dukla Praha
            league_id = 224  # Czech Liga (correct league for these teams)
            reference_date = datetime(2025, 8, 1, 18, 0)  # After some matches have been played
            
            print(f"Testing teams: {home_team_id} (home) vs {away_team_id} (away)")
            print(f"League: {league_id}, Reference date: {reference_date}")
            print(f"Using minimum matches: {generator.state_classifier.min_matches}")
            print("=" * 50)
            
            # Test get_current_state_info
            print("\n=== CURRENT STATE INFO ===")
            home_state_info = generator.get_current_state_info(home_team_id, league_id, reference_date, 'home')
            away_state_info = generator.get_current_state_info(away_team_id, league_id, reference_date, 'away')
            
            print(f"Home team state info: {home_state_info}")
            print(f"Away team state info: {away_state_info}")
            
            # Test generate_team_features
            print("\n=== TEAM FEATURES ===")
            home_features = generator.generate_team_features(home_team_id, league_id, reference_date, 'home')
            away_features = generator.generate_team_features(away_team_id, league_id, reference_date, 'away')
            
            print(f"Home team features: {home_features}")
            print(f"Away team features: {away_features}")
            
            # Test generate_features
            print("\n=== ALL FEATURES ===")
            all_features = generator.generate_features(home_team_id, away_team_id, league_id, reference_date)
            print(f"All features: {all_features}")
            
            # Test state classifier directly
            print("\n=== STATE CLASSIFIER TEST ===")
            print(f"State classifier available: {generator.state_classifier}")
            
            # Test performance calculation
            home_performance = generator.state_classifier.calculate_performance_score(
                home_team_id, league_id, reference_date, 'overall'
            )
            print(f"Home team performance metrics: {home_performance}")
            
            if home_performance['total_matches'] >= generator.state_classifier.min_matches:
                home_state = generator.state_classifier.classify_state(home_performance['performance_score'])
                print(f"Home team classified state: {home_state}")
            else:
                print(f"Not enough matches for home team: {home_performance['total_matches']} < {generator.state_classifier.min_matches}")
                
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_markov_generator_with_min_matches()