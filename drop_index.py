#!/usr/bin/env python3
"""Drop the duplicate ix_alert_type index."""

import os
from sqlalchemy import create_engine, text
from formfinder.config import load_config

def main():
    # Load configuration
    load_config()
    
    # Get database URL from environment
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("DATABASE_URL not found in environment variables")
        return
    
    # Create engine and drop index
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        try:
            conn.execute(text('DROP INDEX IF EXISTS ix_alert_type'))
            conn.commit()
            print("✅ Index ix_alert_type dropped successfully")
        except Exception as e:
            print(f"❌ Error dropping index: {e}")

if __name__ == '__main__':
    main()