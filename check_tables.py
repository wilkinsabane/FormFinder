#!/usr/bin/env python3
"""
Script to check what tables exist in the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

# Load configuration
load_config()

def check_tables():
    """Check what tables exist in the database"""
    with get_db_session() as session:
        # Check for Markov-related tables
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND (table_name LIKE '%markov%' OR table_name LIKE '%transition%')
        """))
        
        markov_tables = result.fetchall()
        print("Markov/transition tables:")
        if markov_tables:
            for row in markov_tables:
                print(f"  {row[0]}")
        else:
            print("  No Markov/transition tables found")
        
        # Check all tables to see what's available
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        
        all_tables = result.fetchall()
        print("\nAll tables in database:")
        for row in all_tables:
            print(f"  {row[0]}")

if __name__ == '__main__':
    check_tables()