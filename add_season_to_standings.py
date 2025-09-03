#!/usr/bin/env python3
"""
Database Migration: Add Season Column to Standings Table

This script adds a season column to the standings table and updates the unique constraint
to prevent data overwrites across different seasons.

Author: FormFinder2 Team
Created: 2025-01-27
Purpose: Fix standings table to differentiate between seasons
"""

import logging
from sqlalchemy import text, MetaData, Table, Column, String, UniqueConstraint, Index
from sqlalchemy.exc import SQLAlchemyError
from formfinder.config import load_config
from formfinder.database import get_db_session, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_season_column_to_standings():
    """Add season column to standings table and update constraints."""
    try:
        with get_db_session() as session:
            logger.info("Starting standings table migration...")
            
            # Step 1: Add season column
            logger.info("Adding season column to standings table...")
            session.execute(text("""
                ALTER TABLE standings 
                ADD COLUMN season VARCHAR(20)
            """))
            session.commit()
            logger.info("✓ Season column added successfully")
            
            # Step 2: Update existing records with season data
            logger.info("Updating existing standings with season data...")
            
            # Get season data from leagues table and update standings
            update_query = text("""
                UPDATE standings 
                SET season = (
                    SELECT l.season 
                    FROM leagues l 
                    WHERE l.league_pk = standings.league_id 
                    LIMIT 1
                )
                WHERE season IS NULL
            """)
            
            result = session.execute(update_query)
            session.commit()
            logger.info(f"✓ Updated {result.rowcount} standings records with season data")
            
            # Step 3: Set season as NOT NULL
            logger.info("Setting season column as NOT NULL...")
            session.execute(text("""
                ALTER TABLE standings 
                ALTER COLUMN season SET NOT NULL
            """))
            session.commit()
            logger.info("✓ Season column set as NOT NULL")
            
            # Step 4: Drop old unique constraint
            logger.info("Dropping old unique constraint...")
            try:
                session.execute(text("""
                    ALTER TABLE standings 
                    DROP CONSTRAINT uq_standing_league_team
                """))
                session.commit()
                logger.info("✓ Old unique constraint dropped")
            except SQLAlchemyError as e:
                logger.warning(f"Could not drop old constraint (may not exist): {e}")
            
            # Step 5: Add new unique constraint with season
            logger.info("Adding new unique constraint with season...")
            session.execute(text("""
                ALTER TABLE standings 
                ADD CONSTRAINT uq_standing_league_team_season 
                UNIQUE (league_id, team_id, season)
            """))
            session.commit()
            logger.info("✓ New unique constraint added")
            
            # Step 6: Add index for better query performance
            logger.info("Adding index for season queries...")
            try:
                session.execute(text("""
                    CREATE INDEX ix_standing_league_season 
                    ON standings (league_id, season)
                """))
                session.commit()
                logger.info("✓ Season index added")
            except SQLAlchemyError as e:
                logger.warning(f"Could not create index (may already exist): {e}")
            
            logger.info("🎉 Standings table migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

def verify_migration():
    """Verify the migration was successful."""
    try:
        with get_db_session() as session:
            logger.info("Verifying migration...")
            
            # Check if season column exists
            result = session.execute(text("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'standings' AND column_name = 'season'
            """))
            
            column_info = result.fetchone()
            if column_info:
                logger.info(f"✓ Season column exists: {column_info[0]} ({column_info[1]}, nullable: {column_info[2]})")
            else:
                logger.error("✗ Season column not found")
                return False
            
            # Check unique constraint
            result = session.execute(text("""
                SELECT constraint_name 
                FROM information_schema.table_constraints 
                WHERE table_name = 'standings' 
                AND constraint_type = 'UNIQUE' 
                AND constraint_name = 'uq_standing_league_team_season'
            """))
            
            constraint_info = result.fetchone()
            if constraint_info:
                logger.info(f"✓ New unique constraint exists: {constraint_info[0]}")
            else:
                logger.error("✗ New unique constraint not found")
                return False
            
            # Check sample data
            result = session.execute(text("""
                SELECT COUNT(*) as total_standings,
                       COUNT(DISTINCT season) as unique_seasons,
                       COUNT(CASE WHEN season IS NULL THEN 1 END) as null_seasons
                FROM standings
            """))
            
            stats = result.fetchone()
            logger.info(f"✓ Standings stats: {stats[0]} total, {stats[1]} unique seasons, {stats[2]} null seasons")
            
            if stats[2] > 0:
                logger.warning(f"Found {stats[2]} standings with null seasons")
            
            logger.info("✅ Migration verification completed")
            return True
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def main():
    """Main migration function."""
    load_config()
    
    logger.info("=== Standings Table Season Migration ===")
    logger.info("This will add a season column to the standings table")
    logger.info("and update the unique constraint to prevent overwrites.")
    
    try:
        # Run migration
        add_season_column_to_standings()
        
        # Verify migration
        if verify_migration():
            logger.info("🎉 Migration completed successfully!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Update save_standings_to_database method to include season")
            logger.info("2. Update _get_team_position methods to consider season")
            logger.info("3. Test with different season data")
        else:
            logger.error("❌ Migration verification failed")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        logger.error("Please check the error and try again")

if __name__ == "__main__":
    main()