#!/usr/bin/env python3
"""Temporary script to truncate pre_computed_features table."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    # Load configuration first
    load_config()
    
    with get_db_session() as session:
        session.execute(text('TRUNCATE TABLE pre_computed_features'))
        session.commit()
        print('pre_computed_features table truncated successfully')

if __name__ == '__main__':
    main()