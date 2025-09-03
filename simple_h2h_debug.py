#!/usr/bin/env python3
"""
Simple H2H debug script
"""

import logging
from formfinder.config import load_config
from formfinder.database import DatabaseManager
from enhanced_predictor import EnhancedGoalPredictor
from sqlalchemy import text

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def simple_h2h_debug():
    """Simple H2H debug."""
    load_config()
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # Find a fixture with known H2H history (teams 3057 vs 3528)
        test_fixture_query = text("""
            SELECT id, home_team_id, away_team_id, match_date
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
            print("No test fixture found")
            return
        
        fixture_id, home_team_id, away_team_id, match_date = test_fixture
        print(f"Testing Fixture {fixture_id}: {home_team_id} vs {away_team_id}")
        print(f"Date: {match_date}")
        
        # Test direct H2H query first
        print("\n=== Direct H2H Query ===")
        h2h_query = text("""
            SELECT COUNT(*) as total_matches
            FROM fixtures
            WHERE ((home_team_id = :home_team_id AND away_team_id = :away_team_id)
                   OR (home_team_id = :away_team_id AND away_team_id = :home_team_id))
                AND status = 'finished'
                AND match_date < :match_date
        """)
        
        h2h_result = session.execute(h2h_query, {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'match_date': match_date
        }).fetchone()
        
        print(f"Direct H2H query result: {h2h_result[0]} matches found")
        
        # Extract features
        print("\n=== Extracting Features ===")
        predictor = EnhancedGoalPredictor()
        features = predictor.extract_enhanced_features(fixture_id)
        
        if not features:
            print("No features extracted")
            return
        
        print(f"Total features extracted: {len(features)}")
        
        # Look for H2H-related features
        print("\n=== H2H-Related Features ===")
        h2h_features = {k: v for k, v in features.items() if 'h2h' in k.lower()}
        
        if h2h_features:
            for key, value in h2h_features.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No H2H features found in extracted features")
            
            # Check if there are any features with 'head' or 'match' in the name
            related_features = {k: v for k, v in features.items() 
                              if any(word in k.lower() for word in ['head', 'match', 'history'])}
            if related_features:
                print("\nPotentially related features:")
                for key, value in related_features.items():
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    simple_h2h_debug()