#!/usr/bin/env python3
"""Check database schema for h2h_cache table and predictions columns."""

import os
from sqlalchemy import create_engine, text
from formfinder.config import load_config, get_config

def check_schema():
    """Check if migration has been applied."""
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        # Check if h2h_cache table exists
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'h2h_cache'
        """))
        h2h_exists = result.fetchone() is not None
        print(f"h2h_cache table exists: {h2h_exists}")
        
        # Check predictions table columns
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'predictions'
            AND column_name IN ('predicted_total_goals', 'over_2_5_probability')
        """))
        pred_columns = [row[0] for row in result.fetchall()]
        print(f"Predictions table goal columns: {pred_columns}")
        
        # Check if old W/D/L columns still exist
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'predictions'
            AND column_name IN ('home_win_probability', 'draw_probability', 'away_win_probability')
        """))
        old_columns = [row[0] for row in result.fetchall()]
        print(f"Old W/D/L columns still present: {old_columns}")
        
        return h2h_exists and len(pred_columns) == 2

if __name__ == "__main__":
    success = check_schema()
    print(f"\nMigration status: {'Complete' if success else 'Incomplete'}")