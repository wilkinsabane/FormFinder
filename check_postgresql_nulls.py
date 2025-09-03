#!/usr/bin/env python3
"""
Check null values in pre_computed_features table in PostgreSQL database.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
import psycopg2
from psycopg2.extras import RealDictCursor

def main():
    """Check null values in pre_computed_features table."""
    try:
        # Load configuration
        load_config()
        
        # Get database connection details from config
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        db_config = config['database']['postgresql']
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['username'],
            password=db_config['password']
        )
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if pre_computed_features table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'pre_computed_features'
            );
        """)
        
        result = cursor.fetchone()
        table_exists = result['exists'] if result else False
        
        if not table_exists:
            print("❌ pre_computed_features table does not exist")
            return
        
        print("✅ pre_computed_features table exists")
        
        # Get total row count
        cursor.execute("SELECT COUNT(*) FROM pre_computed_features;")
        result = cursor.fetchone()
        total_rows = result['count'] if result else 0
        print(f"📊 Total rows: {total_rows}")
        
        if total_rows == 0:
            print("⚠️  Table is empty")
            return
        
        # Check specific columns for null values
        target_columns = [
            'home_attack_strength',
            'home_defense_strength', 
            'away_attack_strength',
            'away_defense_strength',
            'home_form_diff',
            'away_form_diff',
            'home_team_form_score',
            'away_team_form_score'
        ]
        
        print("\n🔍 Checking null values in key columns:")
        print("-" * 50)
        
        for column in target_columns:
            # Check if column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'pre_computed_features'
                    AND column_name = %s
                );
            """, (column,))
            
            result = cursor.fetchone()
            column_exists = result['exists'] if result else False
            
            if not column_exists:
                print(f"❌ {column}: Column does not exist")
                continue
            
            # Count null values
            cursor.execute(f"SELECT COUNT(*) FROM pre_computed_features WHERE {column} IS NULL;")
            result = cursor.fetchone()
            null_count = result['count'] if result else 0
            
            # Count non-null values
            cursor.execute(f"SELECT COUNT(*) FROM pre_computed_features WHERE {column} IS NOT NULL;")
            result = cursor.fetchone()
            non_null_count = result['count'] if result else 0
            
            null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0
            
            status = "❌" if null_count > 0 else "✅"
            print(f"{status} {column}: {null_count}/{total_rows} null ({null_percentage:.1f}%)")
            
            # Show sample non-null values if any exist
            if non_null_count > 0:
                cursor.execute(f"SELECT {column} FROM pre_computed_features WHERE {column} IS NOT NULL LIMIT 3;")
                samples = [row[column] for row in cursor.fetchall()]
                print(f"   Sample values: {samples}")
        
        # Get table schema info
        print("\n📋 Table schema:")
        print("-" * 50)
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'pre_computed_features'
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        for col in columns:
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            print(f"  {col['column_name']}: {col['data_type']} {nullable}")
        
        # Show sample rows
        print("\n📄 Sample rows:")
        print("-" * 50)
        cursor.execute("SELECT * FROM pre_computed_features LIMIT 3;")
        rows = cursor.fetchall()
        
        if rows:
            for i, row in enumerate(rows, 1):
                print(f"Row {i}:")
                for key, value in row.items():
                    print(f"  {key}: {value}")
                print()
        else:
            print("No rows found")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()