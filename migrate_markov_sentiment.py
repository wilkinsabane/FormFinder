#!/usr/bin/env python3
"""
Migration script to add Markov and sentiment columns to pre_computed_features table.
"""

import logging
from sqlalchemy import text
from formfinder.database import DatabaseManager
from formfinder.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_markov_sentiment_features():
    """Add Markov and sentiment features columns to pre_computed_features table."""
    # Load configuration first
    load_config()
    db_manager = DatabaseManager()
    
    # SQL statements to add new columns
    migration_statements = [
        # Sentiment features
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_sentiment FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_sentiment FLOAT;",
        
        # Markov features
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_markov_momentum FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_markov_momentum FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_state_stability FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_state_stability FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_transition_entropy FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_transition_entropy FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_performance_volatility FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_performance_volatility FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_current_state VARCHAR(20);",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_current_state VARCHAR(20);",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_state_duration INTEGER;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_state_duration INTEGER;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_expected_next_state VARCHAR(20);",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_expected_next_state VARCHAR(20);",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS home_team_state_confidence FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS away_team_state_confidence FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS markov_match_prediction_confidence FLOAT;",
        "ALTER TABLE pre_computed_features ADD COLUMN IF NOT EXISTS markov_outcome_probabilities TEXT;",
        
        # Note: Skipping check constraints as PostgreSQL doesn't support IF NOT EXISTS for constraints
        # These can be added manually if needed
    ]
    
    try:
        with db_manager.get_session() as session:
            for statement in migration_statements:
                logger.info(f"Executing: {statement[:100]}...")
                session.execute(text(statement))
            
            session.commit()
            logger.info("Migration completed successfully!")
            logger.info("Added Markov and sentiment columns to pre_computed_features table.")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_markov_sentiment_features()