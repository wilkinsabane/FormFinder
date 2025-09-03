#!/usr/bin/env python3
"""
Check the schema of the leagues table.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check if we're using SQLite or PostgreSQL
        db_url = str(session.bind.url)
        
        if 'sqlite' in db_url:
            query = text('PRAGMA table_info(leagues);')
        else:
            query = text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'leagues' 
                ORDER BY ordinal_position;
            """)
        
        result = session.execute(query).fetchall()
        
        print("Leagues table columns:")
        for row in result:
            print(f"  {row}")
        
        # Also check if is_cup column exists specifically
        try:
            test_query = text("SELECT is_cup FROM leagues LIMIT 1")
            session.execute(test_query)
            print("\nis_cup column exists: YES")
        except Exception as e:
            print(f"\nis_cup column exists: NO ({e})")

if __name__ == "__main__":
    main()