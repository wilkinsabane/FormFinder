#!/usr/bin/env python3
"""Clear pre_computed_features table to force regeneration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text

def clear_precomputed_features():
    """Clear all pre-computed features to force regeneration."""
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        conn.execute(text('TRUNCATE TABLE pre_computed_features'))
        conn.commit()
        print('✅ Cleared pre_computed_features table')

if __name__ == "__main__":
    clear_precomputed_features()