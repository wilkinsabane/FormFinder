#!/usr/bin/env python3
"""
Debug test for H2H database constraint issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import traceback

def test_h2h_debug():
    """Debug H2H database constraint issues."""
    load_config()
    config = get_config()
    
    # Import after config is loaded
    from formfinder.clients.api_client import SoccerDataAPIClient
    
    print("=== H2H Debug Test ===")
    
    # Create database session
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Initialize API client with session
    api_client = SoccerDataAPIClient(session)
    
    # Test with specific team IDs
    team1_id = 2689
    team2_id = 2693
    
    print(f"Testing H2H for Team {team1_id} vs Team {team2_id}")
    
    try:
        h2h_data = api_client.get_h2h_stats(team1_id, team2_id)
        print("✅ H2H data retrieved and cached successfully!")
        print(f"   Games played: {h2h_data.get('overall_games_played', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to get more specific error info
        if "constraint" in str(e).lower() or "violates" in str(e).lower():
            print("\n🔍 This appears to be a database constraint violation.")
            
        return False
    finally:
        session.close()

if __name__ == "__main__":
    test_h2h_debug()