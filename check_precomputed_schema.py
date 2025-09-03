#!/usr/bin/env python3
"""
Check the actual schema of pre_computed_features table.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
import pandas as pd

def check_precomputed_schema():
    """Check the schema of pre_computed_features table."""
    load_config()
    
    with get_db_session() as session:
        conn = session.get_bind().raw_connection()
        
        # Get table schema
        query1 = """
        SELECT 
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'pre_computed_features'
        ORDER BY ordinal_position
        """
        
        df1 = pd.read_sql(query1, conn)
        print("=== Pre-Computed Features Table Schema ===")
        print(df1.to_string(index=False))
        print()
        
        # Get sample data
        query2 = """
        SELECT *
        FROM pre_computed_features
        LIMIT 5
        """
        
        df2 = pd.read_sql(query2, conn)
        print("=== Sample Data ===")
        print(df2.to_string(index=False))
        print()
        
        # Count total records
        query3 = "SELECT COUNT(*) as total_records FROM pre_computed_features"
        df3 = pd.read_sql(query3, conn)
        print("=== Total Records ===")
        print(df3.to_string(index=False))
        
        conn.close()

if __name__ == "__main__":
    check_precomputed_schema()