#!/usr/bin/env python3
"""
Test H2H integration with the prediction system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def test_h2h_integration():
    """Test H2H integration with upcoming fixtures."""
    load_config()
    config = get_config()
    
    # Import after config is loaded
    from formfinder.clients.api_client import SoccerDataAPIClient
    
    print("=== H2H Integration Test ===")
    
    # Create database session
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Initialize API client with session
        api_client = SoccerDataAPIClient(session)
        
        # Get some upcoming fixtures to test with
        result = session.execute(text("""
            SELECT id, home_team_id, away_team_id, match_date
            FROM fixtures 
            WHERE home_team_id IS NOT NULL AND away_team_id IS NOT NULL
            AND status = 'upcoming'
            LIMIT 5
        """))
        
        upcoming_fixtures = result.fetchall()
        
        if not upcoming_fixtures:
            print("❌ No upcoming fixtures found for testing")
            # Try with any fixtures
            result = session.execute(text("""
                SELECT id, home_team_id, away_team_id, match_date
                FROM fixtures 
                WHERE home_team_id IS NOT NULL AND away_team_id IS NOT NULL
                LIMIT 5
            """))
            upcoming_fixtures = result.fetchall()
        
        if not upcoming_fixtures:
            print("❌ No fixtures found at all")
            return False
        
        print(f"Found {len(upcoming_fixtures)} fixtures to test")
        
        h2h_success_count = 0
        h2h_cache_hits = 0
        h2h_cache_misses = 0
        
        for fixture in upcoming_fixtures:
            fixture_id, home_team_id, away_team_id, match_date = fixture
            print(f"\nTesting fixture {fixture_id}: Team {home_team_id} vs Team {away_team_id}")
            
            try:
                # Test H2H data fetching
                h2h_data = api_client.get_h2h_stats(home_team_id, away_team_id)
                
                if h2h_data:
                    h2h_success_count += 1
                    
                    # Test enhanced H2H data structure
                    games_played = h2h_data.get('overall_games_played', 0)
                    home_wins = h2h_data.get('overall_team1_wins', 0)
                    away_wins = h2h_data.get('overall_team2_wins', 0)
                    draws = h2h_data.get('overall_draws', 0)
                    avg_goals = h2h_data.get('avg_total_goals', 0.0)
                    
                    # Test home-specific statistics
                    team1_home_games = h2h_data.get('team1_games_played_at_home', 0)
                    team1_home_wins = h2h_data.get('team1_wins_at_home', 0)
                    team2_home_games = h2h_data.get('team2_games_played_at_home', 0)
                    team2_home_wins = h2h_data.get('team2_wins_at_home', 0)
                    
                    print(f"  ✅ H2H Overall: {games_played} games, {home_wins}-{draws}-{away_wins}, avg {avg_goals:.1f} goals")
                    print(f"  📊 Team1 at home: {team1_home_games} games, {team1_home_wins} wins")
                    print(f"  📊 Team2 at home: {team2_home_games} games, {team2_home_wins} wins")
                    
                    # Verify data structure completeness
                    required_fields = [
                        'overall_games_played', 'overall_team1_wins', 'overall_team2_wins', 'overall_draws',
                        'overall_team1_scored', 'overall_team2_scored', 'avg_total_goals',
                        'team1_games_played_at_home', 'team1_wins_at_home', 'team1_losses_at_home', 'team1_draws_at_home',
                        'team1_scored_at_home', 'team1_conceded_at_home',
                        'team2_games_played_at_home', 'team2_wins_at_home', 'team2_losses_at_home', 'team2_draws_at_home',
                        'team2_scored_at_home', 'team2_conceded_at_home'
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in h2h_data]
                    if missing_fields:
                        print(f"  ⚠️  Missing fields: {missing_fields}")
                    else:
                        print(f"  ✅ All enhanced H2H fields present")
                    
                    # Check if this was a cache hit or miss
                    if games_played > 0:
                        h2h_cache_hits += 1
                    else:
                        h2h_cache_misses += 1
                else:
                    print(f"  ❌ No H2H data returned")
                    
            except Exception as e:
                print(f"  ❌ H2H error: {e}")
        
        print(f"\n=== H2H Integration Test Results ===")
        print(f"Fixtures tested: {len(upcoming_fixtures)}")
        print(f"H2H requests successful: {h2h_success_count}")
        print(f"Success rate: {h2h_success_count/len(upcoming_fixtures)*100:.1f}%")
        
        if h2h_success_count > 0:
            print("\n🎉 H2H integration is working correctly!")
            print("The prediction system should now be able to fetch H2H data for upcoming matches.")
            return True
        else:
            print("\n💥 H2H integration has issues.")
            return False
            
    finally:
        session.close()

if __name__ == "__main__":
    test_h2h_integration()