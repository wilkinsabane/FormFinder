#!/usr/bin/env python3

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_precomputed_features_columns():
    """Check what columns exist in the pre_computed_features table."""
    load_config()
    
    with get_db_session() as session:
        result = session.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'pre_computed_features' "
            "ORDER BY column_name"
        ))
        columns = [row[0] for row in result.fetchall()]
        
        print("Columns in pre_computed_features table:")
        for col in columns:
            print(f"  - {col}")
        
        # Check specifically for created_at and updated_at
        missing_columns = []
        required_columns = ['created_at', 'updated_at']
        
        for col in required_columns:
            if col not in columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"\nMissing columns: {missing_columns}")
        else:
            print("\nAll required timestamp columns are present.")

if __name__ == "__main__":
    check_precomputed_features_columns()