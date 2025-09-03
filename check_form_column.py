#!/usr/bin/env python3
"""Check form column in standings table."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # First check the actual schema of standings table
        print("Checking standings table schema...")
        schema_info = session.execute(text(
            """SELECT column_name, data_type, is_nullable, column_default
               FROM information_schema.columns 
               WHERE table_name = 'standings'
               ORDER BY ordinal_position"""
        )).fetchall()
        
        print("\nStandings table columns:")
        for col_name, data_type, is_nullable, col_default in schema_info:
            print(f"  {col_name}: {data_type} (nullable: {is_nullable}, default: {col_default})")
        
        # Check form column values
        print("\nChecking form column in standings table...")
        
        # Get total count
        total_count = session.execute(text(
            "SELECT COUNT(*) FROM standings"
        )).fetchone()[0]
        print(f"Total standings records: {total_count}")
        
        # Check if form column exists
        form_column_exists = any(col[0] == 'form' for col in schema_info)
        
        if form_column_exists:
            # Check distinct form values
            distinct_values = session.execute(text(
                "SELECT DISTINCT form FROM standings ORDER BY form"
            )).fetchall()
            print(f"\nDistinct form values: {[row[0] for row in distinct_values]}")
            
            # Count records with form = NULL
            null_count = session.execute(text(
                "SELECT COUNT(*) FROM standings WHERE form IS NULL"
            )).fetchone()[0]
            print(f"Records with form = NULL: {null_count}")
            
            # Count records with form != NULL
            non_null_count = session.execute(text(
                "SELECT COUNT(*) FROM standings WHERE form IS NOT NULL"
            )).fetchone()[0]
            print(f"Records with form != NULL: {non_null_count}")
            
            # Check some sample records with form data
            print("\nSample standings with form data:")
            sample_data = session.execute(text(
                "SELECT team_id, form FROM standings LIMIT 10"
            )).fetchall()
            
            for team_id, form in sample_data:
                print(f"Team {team_id}: form='{form}'")
        else:
            print("\nForm column does not exist in standings table!")

if __name__ == '__main__':
    main()