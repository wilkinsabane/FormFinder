#!/usr/bin/env python3
"""
Script to check H2H cache table constraints.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text

def check_h2h_constraints():
    """Check the H2H cache table constraints."""
    load_config()
    config = get_config()
    
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        # Get primary key constraints
        result = conn.execute(text("""
            SELECT constraint_name, constraint_type 
            FROM information_schema.table_constraints 
            WHERE table_name = 'h2h_cache' AND constraint_type = 'PRIMARY KEY'
        """))
        
        pk_constraints = result.fetchall()
        print("Primary key constraints:")
        for constraint in pk_constraints:
            print(f"  {constraint[0]}: {constraint[1]}")
        
        # Get primary key columns
        result = conn.execute(text("""
            SELECT kcu.column_name 
            FROM information_schema.key_column_usage kcu 
            JOIN information_schema.table_constraints tc 
                ON kcu.constraint_name = tc.constraint_name 
            WHERE tc.table_name = 'h2h_cache' 
                AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position
        """))
        
        pk_columns = result.fetchall()
        print("\nPrimary key columns:")
        for col in pk_columns:
            print(f"  {col[0]}")
        
        # Check if there are any existing records that might conflict
        result = conn.execute(text("""
            SELECT team1_id, team2_id, competition_id, COUNT(*) as count
            FROM h2h_cache 
            GROUP BY team1_id, team2_id, competition_id
            HAVING COUNT(*) > 1
        """))
        
        duplicates = result.fetchall()
        if duplicates:
            print("\nDuplicate records found:")
            for dup in duplicates:
                print(f"  Teams {dup[0]} vs {dup[1]}, competition {dup[2]}: {dup[3]} records")
        else:
            print("\nNo duplicate records found.")
        
        # Check for any records with the test teams
        result = conn.execute(text("""
            SELECT team1_id, team2_id, competition_id, last_fetched_at
            FROM h2h_cache 
            WHERE (team1_id = 2689 AND team2_id = 2693) 
               OR (team1_id = 2693 AND team2_id = 2689)
        """))
        
        existing = result.fetchall()
        if existing:
            print("\nExisting records for test teams (2689, 2693):")
            for record in existing:
                print(f"  Teams {record[0]} vs {record[1]}, competition {record[2]}, fetched: {record[3]}")
        else:
            print("\nNo existing records for test teams (2689, 2693).")

if __name__ == "__main__":
    check_h2h_constraints()