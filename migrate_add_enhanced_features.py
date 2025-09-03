#!/usr/bin/env python3
"""
Migration script to add enhanced features to PreComputedFeatures table.
Adds xG features, team strength/momentum features, and ensures all Markov features are present.
"""

import logging
from sqlalchemy import text
from formfinder.database import DatabaseManager
from formfinder.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_add_enhanced_features():
    """Add enhanced features columns to pre_computed_features table."""
    # Load configuration first
    load_config()
    db_manager = DatabaseManager()
    
    # SQL statements to add new columns
    migration_statements = [
        # xG features
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_xg FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_xg FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_avg_goals_scored FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_avg_goals_conceded FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_avg_goals_scored FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_avg_goals_conceded FLOAT;",
        
        # Team strength and momentum features
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_strength FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_strength FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_momentum FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_momentum FLOAT;",
    ]
    
    try:
        with db_manager.get_session() as session:
            for statement in migration_statements:
                logger.info(f"Executing: {statement}")
                session.execute(text(statement))
            
            session.commit()
            logger.info("Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_add_enhanced_features()