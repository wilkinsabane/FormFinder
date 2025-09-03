#!/usr/bin/env python3
"""
Test script to debug Markov momentum feature generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.database import get_db_session
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from sqlalchemy import text
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_markov_momentum():
    """Test Markov momentum feature generation"""
    
    # Load configuration and database session
    load_config()
    config = get_config()
    
    try:
        with get_db_session() as session:
            # Initialize Markov generator
            print("🔗 Initializing Markov feature generator...")
            markov_generator = MarkovFeatureGenerator(db_session=session, lookback_window=10)
            print("✅ Markov generator initialized successfully")
            
            # Get a test fixture
            result = session.execute(text("""
                SELECT id, home_team_id, away_team_id, match_date, league_id 
                FROM fixtures 
                WHERE home_score IS NOT NULL 
                AND away_score IS NOT NULL 
                AND status = 'finished'
                ORDER BY match_date DESC 
                LIMIT 1
            """))
            
            fixture = result.fetchone()
            if not fixture:
                print("❌ No finished fixtures found")
                return
                
            print(f"\n🎯 Testing with fixture {fixture.id}:")
            print(f"   Home Team: {fixture.home_team_id}")
            print(f"   Away Team: {fixture.away_team_id}")
            print(f"   Match Date: {fixture.match_date}")
            print(f"   League: {fixture.league_id}")
            
            # Test Markov features generation
            print("\n🔍 Generating Markov features...")
            markov_features = markov_generator.generate_features(
                home_team_id=fixture.home_team_id,
                away_team_id=fixture.away_team_id,
                match_date=fixture.match_date,
                league_id=fixture.league_id
            )
            
            if markov_features:
                print(f"✅ Generated {len(markov_features)} Markov features")
                
                # Check for momentum features specifically
                momentum_features = {
                    k: v for k, v in markov_features.items() 
                    if 'momentum' in k.lower()
                }
                
                print(f"\n📊 Momentum features found: {len(momentum_features)}")
                for key, value in momentum_features.items():
                    print(f"   {key}: {value}")
                    
                # Check for all key features
                key_features = [
                    'home_momentum_score', 'away_momentum_score',
                    'home_current_state', 'away_current_state',
                    'home_expected_next_state', 'away_expected_next_state'
                ]
                
                print(f"\n🔑 Key Markov features:")
                for key in key_features:
                    value = markov_features.get(key, 'NOT FOUND')
                    print(f"   {key}: {value}")
                    
            else:
                print("❌ No Markov features generated")
                
    except Exception as e:
        print(f"❌ Error testing Markov momentum: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_markov_momentum()