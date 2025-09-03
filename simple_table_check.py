#!/usr/bin/env python3
"""Simple script to check if pre_computed_features table exists."""

import psycopg2
from formfinder.config import load_config, get_config

def check_table_exists():
    """Check if pre_computed_features table exists using direct psycopg2 connection."""
    try:
        load_config()
        config = get_config()
        
        # Direct database connection
        db_config = config.database.postgresql
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.username,
            password=db_config.password
        )
        
        cur = conn.cursor()
        
        # Check if pre_computed_features table exists
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'pre_computed_features'"
        )
        result = cur.fetchall()
        
        table_exists = len(result) > 0
        print(f"pre_computed_features table exists: {table_exists}")
        
        if table_exists:
            # Check row count
            cur.execute("SELECT COUNT(*) FROM pre_computed_features")
            count = cur.fetchone()[0]
            print(f"Number of rows: {count}")
            
            # Check columns
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'pre_computed_features' ORDER BY ordinal_position"
            )
            columns = [row[0] for row in cur.fetchall()]
            print(f"Number of columns: {len(columns)}")
            print(f"First 10 columns: {columns[:10]}")
        
        cur.close()
        conn.close()
        
        return table_exists
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    check_table_exists()