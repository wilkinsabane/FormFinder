import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
load_config()

from sqlalchemy import create_engine, text

config = get_config()
engine = create_engine(config.get_database_url())

with engine.connect() as conn:
    # Check fixtures table columns
    result = conn.execute(text("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'fixtures' 
        ORDER BY ordinal_position
    """))
    
    print("Fixtures table columns:")
    for row in result:
        print(f"  {row.column_name}: {row.data_type}")
    
    # Check if there are any fixtures
    result = conn.execute(text("SELECT COUNT(*) as count FROM fixtures"))
    count = result.scalar()
    print(f"\nTotal fixtures in database: {count}")
    
    if count > 0:
        # Show sample data
        result = conn.execute(text("SELECT * FROM fixtures LIMIT 3"))
        print("\nSample fixtures data:")
        for row in result:
            print(f"  {dict(row._mapping)}")