#!/usr/bin/env python3
"""
Script to drop the duplicate ix_alert_type index from PostgreSQL database.
"""

import psycopg2
from psycopg2 import sql

def drop_duplicate_index():
    """Drop the duplicate ix_alert_type index."""
    # Database connection parameters from config.yaml
    conn_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'formfinder',
        'user': 'wilkins',
        'password': 'Holmes&7watson'
    }
    
    try:
        # Connect to PostgreSQL
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Drop the index if it exists
        print("Dropping ix_alert_type index...")
        cursor.execute("DROP INDEX IF EXISTS ix_alert_type;")
        
        # Commit the transaction
        conn.commit()
        print("Successfully dropped ix_alert_type index.")
        
    except psycopg2.Error as e:
        print(f"PostgreSQL error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    drop_duplicate_index()