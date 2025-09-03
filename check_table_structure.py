#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

load_config()

with get_db_session() as db:
    result = db.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'pre_computed_features' 
        ORDER BY ordinal_position
    """))
    
    columns = [row[0] for row in result.fetchall()]
    print("Columns in pre_computed_features table:")
    for col in columns:
        print(f"  - {col}")