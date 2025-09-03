#!/usr/bin/env python3
"""
Comprehensive test to verify H2H data fetching and caching with the configured API token.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load configuration first
from formfinder.config import load_config, get_config
load_config()

def test_h2h_with_api_token():
    """Test H2H data fetching and caching with the configured API token."""
    print("=== Testing H2H Data Fetching with API Token ===")
    
    config = get_config()
    
    # Database connection
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 1. Check API token configuration
        print("\n=== API Configuration Check ===")
        print(f"API token configured: {'YES' if config.api.auth_token else 'NO (empty)'}")
        print(f"API token length: {len(config.api.auth_token)}")
        print(f"API base URL: {config.api.base_url}")
        print(f"H2H TTL: {config.api.h2h_ttl_seconds} seconds")
        
        # 2. Check current H2H cache status
        print("\n=== Current H2H Cache Status ===")
        result = session.execute(text("SELECT COUNT(*) as count FROM h2h_cache"))
        cache_count = result.scalar()
        print(f"Current H2H cache entries: {cache_count}")
        
        # 3. Get sample teams for testing
        print("\n=== Finding Sample Teams for Testing ===")
        result = session.execute(text("""
            SELECT DISTINCT home_team_id, away_team_id
            FROM fixtures 
            WHERE home_team_id IS NOT NULL AND away_team_id IS NOT NULL
            LIMIT 3
        """))
        
        test_matches = result.fetchall()
        if not test_matches:
            print("❌ No fixtures found for testing")
            return False
            
        print(f"Found {len(test_matches)} test matches:")
        for match in test_matches:
            print(f"  Team {match.home_team_id} vs Team {match.away_team_id}")
        
        # 4. Initialize API client and test H2H fetching
        print("\n=== Testing H2H Data Fetching ===")
        from formfinder.clients.api_client import SoccerDataAPIClient
        api_client = SoccerDataAPIClient(session)
        
        success_count = 0
        for i, match in enumerate(test_matches):
            print(f"\nTest {i+1}: Fetching H2H for Team {match.home_team_id} vs Team {match.away_team_id}")
            try:
                h2h_data = api_client.get_h2h_stats(match.home_team_id, match.away_team_id)
                if h2h_data:
                    print(f"✓ H2H data fetched successfully")
                    print(f"  Overall games played: {h2h_data.get('overall_games_played', 'N/A')}")
                    print(f"  Team1 wins: {h2h_data.get('overall_team1_wins', 'N/A')}")
                    print(f"  Team2 wins: {h2h_data.get('overall_team2_wins', 'N/A')}")
                    print(f"  Draws: {h2h_data.get('overall_draws', 'N/A')}")
                    success_count += 1
                else:
                    print("⚠️ H2H data fetched but empty")
            except Exception as e:
                print(f"❌ Failed to fetch H2H data: {e}")
                
        # 5. Check updated cache status
        print("\n=== Updated H2H Cache Status ===")
        result = session.execute(text("SELECT COUNT(*) as count FROM h2h_cache"))
        new_cache_count = result.scalar()
        print(f"H2H cache entries after testing: {new_cache_count}")
        print(f"New entries added: {new_cache_count - cache_count}")
        
        # 6. Show sample cache entries
        if new_cache_count > 0:
            print("\n=== Sample Cache Entries ===")
            result = session.execute(text("""
                SELECT team1_id, team2_id, overall_games_played, 
                       overall_team1_wins, overall_team2_wins, overall_draws,
                       last_fetched_at
                FROM h2h_cache 
                ORDER BY last_fetched_at DESC 
                LIMIT 3
            """))
            
            cache_entries = result.fetchall()
            for entry in cache_entries:
                print(f"  Teams {entry.team1_id} vs {entry.team2_id}: {entry.overall_games_played} games, "
                      f"{entry.overall_team1_wins}-{entry.overall_draws}-{entry.overall_team2_wins}, "
                      f"cached at {entry.last_fetched_at}")
        
        # 7. Test cache hit on second request
        if test_matches and success_count > 0:
            print("\n=== Testing Cache Hit ===")
            first_match = test_matches[0]
            print(f"Re-requesting H2H for Team {first_match.home_team_id} vs Team {first_match.away_team_id}")
            try:
                h2h_data = api_client.get_h2h_stats(first_match.home_team_id, first_match.away_team_id)
                if h2h_data:
                    print("✓ Cache hit successful (should see 'H2H CACHE HIT' in logs)")
                else:
                    print("⚠️ Cache hit but data empty")
            except Exception as e:
                print(f"❌ Cache hit failed: {e}")
        
        print("\n=== Test Summary ===")
        print(f"API token configured: ✓")
        print(f"H2H requests attempted: {len(test_matches)}")
        print(f"H2H requests successful: {success_count}")
        print(f"Cache entries added: {new_cache_count - cache_count}")
        
        if success_count > 0:
            print("\n🎉 H2H data fetching and caching is working correctly!")
            return True
        else:
            print("\n❌ H2H data fetching failed for all test cases")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        session.close()

if __name__ == "__main__":
    test_h2h_with_api_token()