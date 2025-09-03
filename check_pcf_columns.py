#!/usr/bin/env python3
"""Check pre_computed_features table columns."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    with get_db_session() as db:
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'pre_computed_features' 
            ORDER BY ordinal_position
        """))
        
        columns = [row[0] for row in result]
        print(f'Total columns in pre_computed_features: {len(columns)}')
        print('\nColumns:')
        for i, col in enumerate(columns, 1):
            print(f'{i:2d}. {col}')

if __name__ == '__main__':
    main()