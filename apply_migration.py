#!/usr/bin/env python3
"""Apply the data collection training separation migration."""

import psycopg2
from formfinder.config import load_config, get_config
from pathlib import Path

def apply_migration():
    """Apply the migration SQL file."""
    try:
        load_config()
        config = get_config()
        
        # Read migration file
        migration_file = Path("migrations/20250101_data_collection_training_separation.sql")
        if not migration_file.exists():
            print(f"Migration file not found: {migration_file}")
            return False
            
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        # Direct database connection
        db_config = config.database.postgresql
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.username,
            password=db_config.password
        )
        
        # Set autocommit to handle DDL statements
        conn.autocommit = True
        cur = conn.cursor()
        
        print("Applying migration...")
        
        # Split SQL into individual statements and execute
        statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements):
            try:
                print(f"Executing statement {i+1}/{len(statements)}...")
                cur.execute(statement)
                print(f"Statement {i+1} executed successfully")
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    print(f"Statement {i+1} skipped (already exists): {e}")
                    continue
                else:
                    print(f"Error in statement {i+1}: {e}")
                    print(f"Statement: {statement[:100]}...")
                    # Continue with other statements
                    continue
        
        cur.close()
        conn.close()
        
        print("Migration completed!")
        return True
        
    except Exception as e:
        print(f"Error applying migration: {e}")
        return False

if __name__ == "__main__":
    apply_migration()