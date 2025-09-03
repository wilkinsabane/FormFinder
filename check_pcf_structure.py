#!/usr/bin/env python3
"""
Check the actual structure of pre_computed_features table.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    # Load configuration first
    load_config()
    
    with get_db_session() as session:
        # Check the structure of pre_computed_features table
        columns_query = text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'pre_computed_features'
            ORDER BY ordinal_position
        """)
        
        columns = session.execute(columns_query).fetchall()
        
        print("Pre-computed features table structure:")
        for col in columns:
            print(f"  {col.column_name}: {col.data_type} ({'NULL' if col.is_nullable == 'YES' else 'NOT NULL'})")
        
        # Get a sample of actual data
        sample_query = text("""
            SELECT *
            FROM pre_computed_features
            ORDER BY fixture_id DESC
            LIMIT 3
        """)
        
        sample_data = session.execute(sample_query).fetchall()
        
        print(f"\nSample data ({len(sample_data)} records):")
        if sample_data:
            # Get column names from the result
            column_names = list(sample_data[0]._mapping.keys())
            print(f"Columns: {column_names}")
            
            for row in sample_data:
                print(f"  Fixture {row.fixture_id}: {dict(row._mapping)}")
        
        # Count total records
        count_query = text("SELECT COUNT(*) FROM pre_computed_features")
        total_count = session.execute(count_query).scalar()
        print(f"\nTotal records in pre_computed_features: {total_count}")
        
        # Check for any null values in key columns
        if columns:
            print("\nChecking for null values in key columns:")
            for col in columns:
                if col.column_name not in ['fixture_id', 'created_at', 'updated_at']:
                    null_count_query = text(f"SELECT COUNT(*) FROM pre_computed_features WHERE {col.column_name} IS NULL")
                    null_count = session.execute(null_count_query).scalar()
                    if null_count > 0:
                        print(f"  {col.column_name}: {null_count} null values")

if __name__ == "__main__":
    main()