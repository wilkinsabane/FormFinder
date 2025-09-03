#!/usr/bin/env python3
"""
Simple test for a single H2H API request.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def test_single_h2h():
    """Test a single H2H API request."""
    load_config()
    config = get_config()
    
    # Import after config is loaded
    from formfinder.clients.api_client import SoccerDataAPIClient
    
    print("=== Testing Single H2H Request ===")
    print(f"API token configured: {'YES' if config.api.auth_token else 'NO'}")
    print(f"API token length: {len(config.api.auth_token)}")
    
    # Create database session
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Initialize API client with session
    api_client = SoccerDataAPIClient(session)
    
    # Test with specific team IDs
    team1_id = 2689
    team2_id = 2693
    
    print(f"\nTesting H2H for Team {team1_id} vs Team {team2_id}")
    
    try:
        h2h_data = api_client.get_h2h_stats(team1_id, team2_id)
        
        if h2h_data:
            print("✅ H2H data retrieved successfully!")
            print(f"   Games played: {h2h_data.get('overall_games_played', 'N/A')}")
            print(f"   Team {team1_id} wins: {h2h_data.get('overall_team1_wins', 'N/A')}")
            print(f"   Team {team2_id} wins: {h2h_data.get('overall_team2_wins', 'N/A')}")
            print(f"   Draws: {h2h_data.get('overall_draws', 'N/A')}")
            print(f"   Average goals: {h2h_data.get('avg_total_goals', 'N/A')}")
            return True
        else:
            print("❌ No H2H data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error fetching H2H data: {e}")
        return False

if __name__ == "__main__":
    success = test_single_h2h()
    if success:
        print("\n🎉 H2H API functionality is working!")
    else:
        print("\n💥 H2H API functionality has issues.")