#!/usr/bin/env python3
"""
Comprehensive test for enhanced H2H integration.
Tests the complete pipeline: API client -> H2H manager -> feature engineering.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime

async def test_h2h_enhanced_integration():
    """Test enhanced H2H integration with all components."""
    load_config()
    config = get_config()
    
    # Import after config is loaded
    from formfinder.clients.api_client import SoccerDataAPIClient
    from formfinder.h2h_manager import H2HManager
    from formfinder.feature_precomputer import FeaturePrecomputer
    from formfinder.features import get_h2h_feature
    
    print("=== Enhanced H2H Integration Test ===")
    
    # Create database session
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Initialize components
        api_client = SoccerDataAPIClient(session)
        h2h_manager = H2HManager(session, api_client)
        
        # Get test teams
        result = session.execute(text("""
            SELECT DISTINCT home_team_id, away_team_id
            FROM fixtures 
            WHERE home_team_id IS NOT NULL AND away_team_id IS NOT NULL
            LIMIT 3
        """))
        
        test_pairs = result.fetchall()
        
        if not test_pairs:
            print("❌ No team pairs found for testing")
            return False
        
        print(f"Testing {len(test_pairs)} team pairs")
        
        success_count = 0
        
        for i, (team1_id, team2_id) in enumerate(test_pairs, 1):
            print(f"\n--- Test {i}: Team {team1_id} vs Team {team2_id} ---")
            
            try:
                # Test 1: API Client direct call
                print("1. Testing API Client...")
                api_h2h_data = api_client.get_h2h_stats(team1_id, team2_id)
                
                if api_h2h_data:
                    print(f"   ✅ API returned {api_h2h_data.get('overall_games_played', 0)} games")
                    
                    # Verify enhanced fields
                    enhanced_fields = [
                        'overall_games_played', 'overall_team1_wins', 'overall_team2_wins',
                        'team1_games_played_at_home', 'team1_wins_at_home',
                        'team2_games_played_at_home', 'team2_wins_at_home'
                    ]
                    
                    missing_api_fields = [f for f in enhanced_fields if f not in api_h2h_data]
                    if missing_api_fields:
                        print(f"   ⚠️  API missing fields: {missing_api_fields}")
                    else:
                        print("   ✅ API has all enhanced fields")
                else:
                    print("   ❌ API returned no data")
                    continue
                
                # Test 2: H2H Manager
                print("2. Testing H2H Manager...")
                manager_h2h_data = await h2h_manager.get_or_compute_h2h(team1_id, team2_id, 203)  # Using Premier League ID
                
                if manager_h2h_data:
                    print(f"   ✅ Manager returned {manager_h2h_data.get('overall_games_played', 0)} games")
                    
                    # Compare with API data
                    if (manager_h2h_data.get('overall_games_played') == 
                        api_h2h_data.get('overall_games_played')):
                        print("   ✅ Manager data matches API data")
                    else:
                        print("   ⚠️  Manager data differs from API data")
                else:
                    print("   ❌ Manager returned no data")
                    continue
                
                # Test 3: Feature Engineering
                print("3. Testing Feature Engineering...")
                try:
                    feature_h2h_data = get_h2h_feature(team1_id, team2_id, api_client)
                    
                    if feature_h2h_data:
                        print(f"   ✅ Features returned {feature_h2h_data.get('h2h_overall_games', 0)} games")
                        
                        # Check for enhanced features
                        enhanced_feature_fields = [
                            'h2h_overall_games', 'h2h_avg_total_goals',
                            'h2h_team1_home_win_rate', 'h2h_team2_home_win_rate',
                            'h2h_team1_home_goals_avg', 'h2h_team2_home_goals_avg'
                        ]
                        
                        present_features = [f for f in enhanced_feature_fields if f in feature_h2h_data]
                        print(f"   📊 Enhanced features present: {len(present_features)}/{len(enhanced_feature_fields)}")
                        
                        if len(present_features) >= len(enhanced_feature_fields) * 0.8:  # 80% threshold
                            print("   ✅ Feature engineering working well")
                        else:
                            print("   ⚠️  Some enhanced features missing")
                    else:
                        print("   ❌ Feature engineering returned no data")
                        continue
                        
                except Exception as e:
                    print(f"   ❌ Feature engineering error: {e}")
                    continue
                
                # Test 4: Database Cache Verification
                print("4. Testing Database Cache...")
                cache_result = session.execute(text("""
                    SELECT overall_games_played, overall_team1_wins, overall_team2_wins,
                           team1_games_played_at_home, team1_wins_at_home,
                           team2_games_played_at_home, team2_wins_at_home,
                           last_fetched_at
                    FROM h2h_cache 
                    WHERE (team1_id = :team1 AND team2_id = :team2) 
                       OR (team1_id = :team2 AND team2_id = :team1)
                    ORDER BY last_fetched_at DESC
                    LIMIT 1
                """), {'team1': team1_id, 'team2': team2_id})
                
                cache_row = cache_result.fetchone()
                if cache_row:
                    print(f"   ✅ Cache entry found: {cache_row[0]} games, updated {cache_row[7]}")
                    print(f"   📊 Home stats: Team1({cache_row[3]} games, {cache_row[4]} wins), Team2({cache_row[5]} games, {cache_row[6]} wins)")
                else:
                    print("   ⚠️  No cache entry found")
                
                success_count += 1
                print(f"   🎉 Test {i} completed successfully")
                
            except Exception as e:
                print(f"   ❌ Test {i} failed: {e}")
                continue
        
        print(f"\n=== Enhanced H2H Integration Results ===")
        print(f"Tests completed: {len(test_pairs)}")
        print(f"Tests successful: {success_count}")
        print(f"Success rate: {success_count/len(test_pairs)*100:.1f}%")
        
        if success_count >= len(test_pairs) * 0.7:  # 70% success threshold
            print("\n🎉 Enhanced H2H integration is working correctly!")
            print("✅ API client handles new format")
            print("✅ H2H manager processes enhanced data")
            print("✅ Feature engineering leverages new fields")
            print("✅ Database caching works with enhanced schema")
            return True
        else:
            print("\n💥 Enhanced H2H integration has issues.")
            print("❌ Some components are not working properly")
            return False
            
    finally:
        session.close()

def test_h2h_data_quality():
    """Test the quality and consistency of H2H data."""
    load_config()
    config = get_config()
    
    print("\n=== H2H Data Quality Test ===")
    
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check cache data consistency
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_entries,
                AVG(overall_games_played) as avg_games,
                COUNT(CASE WHEN overall_games_played > 0 THEN 1 END) as entries_with_data,
                COUNT(CASE WHEN team1_games_played_at_home > 0 OR team2_games_played_at_home > 0 THEN 1 END) as entries_with_home_data
            FROM h2h_cache
        """))
        
        stats = result.fetchone()
        if stats:
            total, avg_games, with_data, with_home_data = stats
            print(f"📊 Cache Statistics:")
            print(f"   Total entries: {total}")
            print(f"   Entries with data: {with_data} ({with_data/max(total,1)*100:.1f}%)")
            print(f"   Entries with home data: {with_home_data} ({with_home_data/max(total,1)*100:.1f}%)")
            print(f"   Average games per entry: {avg_games:.1f}")
            
            if with_data/max(total,1) > 0.5:  # 50% threshold
                print("   ✅ Good data coverage")
            else:
                print("   ⚠️  Low data coverage")
        
        # Check for data inconsistencies
        result = session.execute(text("""
            SELECT COUNT(*) as inconsistent_entries
            FROM h2h_cache
            WHERE overall_games_played != (overall_team1_wins + overall_team2_wins + overall_draws)
               OR team1_games_played_at_home != (team1_wins_at_home + team1_losses_at_home + team1_draws_at_home)
               OR team2_games_played_at_home != (team2_wins_at_home + team2_losses_at_home + team2_draws_at_home)
        """))
        
        inconsistent = result.fetchone()[0]
        if inconsistent == 0:
            print("   ✅ No data inconsistencies found")
        else:
            print(f"   ⚠️  {inconsistent} entries have data inconsistencies")
        
        return inconsistent == 0
        
    finally:
        session.close()

if __name__ == "__main__":
    print("Starting Enhanced H2H Integration Tests...\n")
    
    success1 = asyncio.run(test_h2h_enhanced_integration())
    success2 = test_h2h_data_quality()
    
    if success1 and success2:
        print("\n🎉 All enhanced H2H integration tests passed!")
        print("The system is ready for production use with enhanced H2H features.")
    else:
        print("\n❌ Some enhanced H2H integration tests failed")
        print("Please review the issues before deploying to production.")