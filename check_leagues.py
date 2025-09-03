#!/usr/bin/env python3
"""Check available leagues in the database."""

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text

def main():
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT league_id, COUNT(*) as fixture_count 
            FROM fixtures 
            WHERE home_score IS NOT NULL 
            GROUP BY league_id 
            ORDER BY fixture_count DESC
        """))
        
        print("Available leagues in database:")
        for row in result:
            print(f"League ID: {row[0]}, Fixtures: {row[1]}")

if __name__ == "__main__":
    main()