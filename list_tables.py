#!/usr/bin/env python3
"""List all tables in the database."""

import os
from sqlalchemy import create_engine, text
from formfinder.config import load_config, get_config

def list_tables():
    """List all tables in the database."""
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        # Get all tables
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """))
        tables = [row[0] for row in result.fetchall()]
        print("Current database tables:")
        for table in tables:
            print(f"  - {table}")
        
        return tables

if __name__ == "__main__":
    list_tables()