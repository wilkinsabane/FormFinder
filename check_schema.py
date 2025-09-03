#!/usr/bin/env python3
"""
Check database table schemas
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check teams table columns
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'teams' 
            ORDER BY ordinal_position
        """)).fetchall()
        
        print("Teams table columns:")
        for row in result:
            print(f"  {row[0]}")
        
        print("\nFixtures table columns:")
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'fixtures' 
            ORDER BY ordinal_position
        """)).fetchall()
        
        for row in result:
            print(f"  {row[0]}")
        
        # Just show first few rows to understand the data
        print("\nFirst few fixtures (showing all columns):")
        result = session.execute(text("""
            SELECT * FROM fixtures LIMIT 3
        """)).fetchall()
        
        if result:
            print(f"  Sample row: {result[0]}")
        else:
            print("  No fixtures found")

if __name__ == "__main__":
    main()