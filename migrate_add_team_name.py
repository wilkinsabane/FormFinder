#!/usr/bin/env python3
"""
Migration script to add team_name column to high_form_teams table.

This script adds the team_name column to the high_form_teams table and
populates it with team names from the teams table.
"""

import os
import sys
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import init_database, get_db_session
from formfinder.config import load_config, get_config
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_team_name_column():
    """Add team_name column to high_form_teams table."""
    try:
        with get_db_session() as session:
            # Check if team_name column already exists
            try:
                session.execute("SELECT team_name FROM high_form_teams LIMIT 1")
                logger.info("team_name column already exists in high_form_teams table")
                return True
            except Exception:
                logger.info("Adding team_name column to high_form_teams table...")
                
                # Add the team_name column
                session.execute(text("""
                    ALTER TABLE high_form_teams 
                    ADD COLUMN team_name VARCHAR(255) NOT NULL DEFAULT 'Unknown Team'
                """))
                session.commit()
                logger.info("team_name column added successfully")
                
                # Populate team_name column with actual team names
                logger.info("Populating team_name column with team names...")
                session.execute(text("""
                    UPDATE high_form_teams 
                    SET team_name = (
                        SELECT t.name 
                        FROM teams t 
                        WHERE t.id = high_form_teams.team_id
                    )
                """))
                session.commit()
                logger.info("team_name column populated successfully")
                
                return True
                
    except Exception as e:
        logger.error(f"Error adding team_name column: {e}")
        return False


def main():
    """Main migration function."""
    logger.info("Starting migration to add team_name column...")
    
    # Load configuration first
    load_config()
    
    # Initialize database
    init_database()
    
    # Add team_name column
    success = add_team_name_column()
    
    if success:
        logger.info("Migration completed successfully!")
    else:
        logger.error("Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()