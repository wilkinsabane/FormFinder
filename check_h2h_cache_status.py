#!/usr/bin/env python3
"""
Simple script to check H2H cache status.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text

def check_h2h_cache():
    """Check the current status of the H2H cache."""
    load_config()
    config = get_config()
    
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        # Count total entries
        result = conn.execute(text('SELECT COUNT(*) FROM h2h_cache'))
        count = result.scalar()
        print(f"H2H cache entries: {count}")
        
        if count > 0:
            # Show sample entries
            result = conn.execute(text("""
                SELECT team1_id, team2_id, overall_games_played, 
                       overall_team1_wins, overall_team2_wins, overall_draws,
                       last_fetched_at
                FROM h2h_cache 
                ORDER BY last_fetched_at DESC 
                LIMIT 5
            """))
            
            entries = result.fetchall()
            print("\nSample H2H cache entries:")
            for entry in entries:
                print(f"  Teams {entry[0]} vs {entry[1]}: {entry[2]} games, "
                      f"{entry[3]}-{entry[5]}-{entry[4]} (W-D-L), "
                      f"cached at {entry[6]}")
        else:
            print("\nNo H2H cache entries found.")

if __name__ == "__main__":
    check_h2h_cache()