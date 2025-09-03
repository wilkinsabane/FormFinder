#!/usr/bin/env python3
"""Check if the data collection training separation migration has been applied."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from formfinder.config import load_config
    from formfinder.database import get_db_session
    from sqlalchemy import text
    
    def check_migration_status():
        """Check if the pre_computed_features table exists."""
        try:
            load_config()
            
            with get_db_session() as session:
                # Check if pre_computed_features table exists
                result = session.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = 'pre_computed_features'"
                )).fetchall()
                
                table_exists = len(result) > 0
                print(f"pre_computed_features table exists: {table_exists}")
                
                if table_exists:
                    # Check if table has data
                    count_result = session.execute(text("SELECT COUNT(*) FROM pre_computed_features")).fetchone()
                    row_count = count_result[0] if count_result else 0
                    print(f"Number of rows in pre_computed_features: {row_count}")
                    
                    # Check table structure
                    columns_result = session.execute(text(
                        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'pre_computed_features' ORDER BY ordinal_position"
                    )).fetchall()
                    print(f"Table has {len(columns_result)} columns")
                    
                    # Show first few columns
                    if columns_result:
                        print("First 10 columns:")
                        for i, (col_name, data_type) in enumerate(columns_result[:10]):
                            print(f"  {col_name}: {data_type}")
                else:
                    print("Migration needs to be applied.")
                    
                return table_exists
            
        except Exception as e:
            print(f"Error checking migration status: {e}")
            return False
    
    if __name__ == "__main__":
        check_migration_status()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct directory and dependencies are installed.")
except Exception as e:
    print(f"Unexpected error: {e}")