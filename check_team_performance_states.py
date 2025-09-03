#!/usr/bin/env python3
"""Check team_performance_states table structure and data."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_team_performance_states():
    """Check team_performance_states table."""
    try:
        # Load configuration
        load_config()
        
        with get_db_session() as session:
            # Check if table exists (PostgreSQL syntax)
            print("Checking if team_performance_states table exists...")
            table_exists = session.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'team_performance_states'
            """)).fetchone()
            
            if not table_exists:
                print("❌ team_performance_states table does not exist!")
                
                # Check what tables do exist
                print("\nExisting tables:")
                tables = session.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)).fetchall()
                for table in tables:
                    print(f"  - {table[0]}")
                return
            
            print("✅ team_performance_states table exists")
            
            # Check table structure (PostgreSQL syntax)
            print("\nTable structure:")
            columns = session.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'team_performance_states'
                ORDER BY ordinal_position
            """)).fetchall()
            
            for col in columns:
                print(f"  {col[0]} ({col[1]}) - Nullable: {col[2]}")
            
            # Check record count
            count = session.execute(text("""
                SELECT COUNT(*) FROM team_performance_states
            """)).scalar()
            
            print(f"\nTotal records: {count}")
            
            if count > 0:
                # Show sample data
                print("\nSample records:")
                samples = session.execute(text("""
                    SELECT team_id, league_id, performance_state, state_date, 
                           state_score, home_away_context
                    FROM team_performance_states
                    ORDER BY state_date DESC
                    LIMIT 5
                """)).fetchall()
                
                for sample in samples:
                    print(f"  Team {sample[0]}, League {sample[1]}: {sample[2]} ({sample[3]}) - Score: {sample[4]}, Context: {sample[5]}")
                
                # Check unique states
                print("\nUnique performance states:")
                states = session.execute(text("""
                    SELECT DISTINCT performance_state, COUNT(*) as count
                    FROM team_performance_states
                    GROUP BY performance_state
                    ORDER BY count DESC
                """)).fetchall()
                
                for state in states:
                    print(f"  {state[0]}: {state[1]} records")
                
                # Check date range
                date_range = session.execute(text("""
                    SELECT MIN(state_date) as min_date, MAX(state_date) as max_date
                    FROM team_performance_states
                """)).fetchone()
                
                print(f"\nDate range: {date_range[0]} to {date_range[1]}")
                
                # Check for specific team (3245 from our test)
                team_states = session.execute(text("""
                    SELECT COUNT(*) FROM team_performance_states
                    WHERE team_id = 3245
                """)).scalar()
                
                print(f"\nRecords for team 3245: {team_states}")
            else:
                print("❌ No data in team_performance_states table!")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    check_team_performance_states()