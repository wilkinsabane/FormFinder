#!/usr/bin/env python3
"""
Check team performance states table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_team_states():
    """Check if team performance states exist"""
    
    with get_db_session() as session:
        # Check total count
        result = session.execute(text('SELECT COUNT(*) FROM team_performance_states'))
        total_count = result.scalar()
        print(f"Total team performance states: {total_count}")
        
        if total_count > 0:
            # Check sample data
            result = session.execute(text("""
                SELECT team_id, league_id, performance_state, state_date, home_away_context
                FROM team_performance_states 
                ORDER BY state_date DESC 
                LIMIT 5
            """))
            
            print("\nSample team performance states:")
            for row in result.fetchall():
                print(f"  Team {row[0]}, League {row[1]}: {row[2]} on {row[3]} ({row[4]})")
        else:
            print("❌ No team performance states found - this explains why momentum is 0.0")

if __name__ == "__main__":
    load_config()
    check_team_states()