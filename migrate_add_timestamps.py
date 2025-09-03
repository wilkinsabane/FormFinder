#!/usr/bin/env python3

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime

def add_timestamp_columns():
    """Add created_at and updated_at columns to pre_computed_features table."""
    load_config()
    
    with get_db_session() as session:
        try:
            # Add created_at column
            print("Adding created_at column...")
            session.execute(text(
                "ALTER TABLE pre_computed_features "
                "ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()"
            ))
            
            # Add updated_at column
            print("Adding updated_at column...")
            session.execute(text(
                "ALTER TABLE pre_computed_features "
                "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()"
            ))
            
            # Update existing rows to have current timestamp
            print("Updating existing rows with current timestamp...")
            session.execute(text(
                "UPDATE pre_computed_features "
                "SET created_at = COALESCE(created_at, NOW()), "
                "    updated_at = COALESCE(updated_at, NOW()) "
                "WHERE created_at IS NULL OR updated_at IS NULL"
            ))
            
            session.commit()
            print("Successfully added timestamp columns to pre_computed_features table.")
            
        except Exception as e:
            session.rollback()
            print(f"Error adding timestamp columns: {e}")
            raise

if __name__ == "__main__":
    add_timestamp_columns()