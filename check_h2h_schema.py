#!/usr/bin/env python3
"""
Script to check H2H cache table schema.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text

def check_h2h_schema():
    """Check the H2H cache table schema."""
    load_config()
    config = get_config()
    
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        # Check if h2h_cache_enhanced table exists
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'h2h_cache_enhanced'
        """))
        
        table_exists = result.fetchone()
        if not table_exists:
            print("Table 'h2h_cache_enhanced' does not exist")
            return
        
        print("Table 'h2h_cache_enhanced' exists")
        
        # Get table schema
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable, column_default 
            FROM information_schema.columns 
            WHERE table_name = 'h2h_cache_enhanced' 
            ORDER BY ordinal_position
        """))
        
        columns = result.fetchall()
        print(f"\nH2H Cache Enhanced Table Schema ({len(columns)} columns):")
        for col in columns:
            print(f"  {col[0]}: {col[1]} (nullable: {col[2]}, default: {col[3]})")
        
        # Check for specific columns mentioned in the error
        column_names = [col[0] for col in columns]
        
        print("\nChecking for specific columns:")
        for col_name in ['overall_games_played', 'h2h_overall_games', 'overall_team1_wins', 'overall_team2_wins']:
            exists = col_name in column_names
            print(f"  {col_name}: {'EXISTS' if exists else 'MISSING'}")
        
        # Check constraints
        result = conn.execute(text("""
            SELECT constraint_name, constraint_type 
            FROM information_schema.table_constraints 
            WHERE table_name = 'h2h_cache_enhanced'
        """))
        
        constraints = result.fetchall()
        print("\nTable Constraints:")
        for constraint in constraints:
            print(f"  {constraint[0]}: {constraint[1]}")

if __name__ == "__main__":
    check_h2h_schema()