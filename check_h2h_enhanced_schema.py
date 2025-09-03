#!/usr/bin/env python3
"""Check h2h_cache_enhanced table schema."""

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text

def main():
    load_config()
    engine = create_engine(get_config().get_database_url())
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'h2h_cache_enhanced' 
            ORDER BY ordinal_position
        """))
        
        columns = result.fetchall()
        print("h2h_cache_enhanced table columns:")
        for col in columns:
            print(f"  {col[0]}: {col[1]} ({'nullable' if col[2] == 'YES' else 'not null'})")

if __name__ == "__main__":
    main()