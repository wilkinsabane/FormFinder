#!/usr/bin/env python3
"""
Script to check Markov-related data in the database.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    # Load configuration first
    load_config()
    
    with get_db_session() as session:
        print("=== Checking team_performance_states table ===")
        
        # Check if table exists and has data
        count_query = "SELECT COUNT(*) FROM team_performance_states"
        try:
            count = session.execute(text(count_query)).scalar()
            print(f"Total records in team_performance_states: {count}")
            
            if count > 0:
                # Show sample data
                sample_query = """
                SELECT team_id, league_id, performance_state, state_score, 
                       state_date, home_away_context
                FROM team_performance_states 
                ORDER BY state_date DESC 
                LIMIT 10
                """
                
                results = session.execute(text(sample_query)).fetchall()
                print("\nSample team_performance_states data:")
                for row in results:
                    print(f"Team {row[0]}, League {row[1]}: {row[2]} (score: {row[3]}) on {row[4]} [{row[5]}]")
            else:
                print("No data found in team_performance_states table!")
                
        except Exception as e:
            print(f"Error checking team_performance_states: {e}")
        
        print("\n=== Checking markov_features table ===")
        
        # Check markov_features table
        try:
            count_query = "SELECT COUNT(*) FROM markov_features"
            count = session.execute(text(count_query)).scalar()
            print(f"Total records in markov_features: {count}")
            
            if count > 0:
                # Show sample data
                sample_query = """
                SELECT team_id, league_id, current_state, state_stability, 
                       transition_entropy, performance_volatility, feature_date
                FROM markov_features 
                ORDER BY feature_date DESC 
                LIMIT 10
                """
                
                results = session.execute(text(sample_query)).fetchall()
                print("\nSample markov_features data:")
                for row in results:
                    print(f"Team {row[0]}: {row[2]} (stability: {row[3]}, entropy: {row[4]}, volatility: {row[5]}) on {row[6]}")
            else:
                print("No data found in markov_features table!")
                
        except Exception as e:
            print(f"Error checking markov_features: {e}")
        
        print("\n=== Checking if Markov tables exist ===")
        
        # Check if tables exist
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('team_performance_states', 'markov_features')
        """
        
        try:
            tables = session.execute(text(tables_query)).fetchall()
            existing_tables = [table[0] for table in tables]
            print(f"Existing Markov tables: {existing_tables}")
            
            if 'team_performance_states' not in existing_tables:
                print("WARNING: team_performance_states table does not exist!")
            if 'markov_features' not in existing_tables:
                print("WARNING: markov_features table does not exist!")
                
        except Exception as e:
            print(f"Error checking table existence: {e}")

if __name__ == "__main__":
    main()