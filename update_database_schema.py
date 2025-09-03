#!/usr/bin/env python3
"""
Database schema update script for FormFinder.

This script adds missing sentiment analysis columns to the predictions table
and ensures the database schema is up to date.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from formfinder.config import load_config, get_config
from formfinder.database import get_db_manager, init_database
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_predictions_table():
    """Update the predictions table to include sentiment analysis columns."""
    logger.info("Updating predictions table schema...")
    
    try:
        # Load configuration
        load_config()
        db_manager = get_db_manager()
        
        # SQL to add missing columns
        add_columns_sql = """
        DO $$
        BEGIN
            -- Add home_team_sentiment column if it doesn't exist
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='predictions' AND column_name='home_team_sentiment'
            ) THEN
                ALTER TABLE predictions ADD COLUMN home_team_sentiment FLOAT;
                RAISE NOTICE 'Added column: home_team_sentiment';
            END IF;
            
            -- Add away_team_sentiment column if it doesn't exist
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='predictions' AND column_name='away_team_sentiment'
            ) THEN
                ALTER TABLE predictions ADD COLUMN away_team_sentiment FLOAT;
                RAISE NOTICE 'Added column: away_team_sentiment';
            END IF;
            
            -- Add sentiment_articles_analyzed column if it doesn't exist
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='predictions' AND column_name='sentiment_articles_analyzed'
            ) THEN
                ALTER TABLE predictions ADD COLUMN sentiment_articles_analyzed INTEGER;
                RAISE NOTICE 'Added column: sentiment_articles_analyzed';
            END IF;
        END $$;
        """
        
        # Execute the SQL
        with db_manager.engine.connect() as conn:
            conn.execute(text(add_columns_sql))
            conn.commit()
            
          
        logger.info("✅ Predictions table updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating predictions table: {e}")
        return False


def update_fixtures_table():
    """Update the fixtures table to include missing columns."""
    logger.info("Updating fixtures table schema...")
    
    try:
        # Load configuration
        load_config()
        db_manager = get_db_manager()
        
        # SQL to add missing columns
        add_columns_sql = """
        DO $$
        BEGIN
            -- Add api_fixture_id column if it doesn't exist
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='api_fixture_id'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN api_fixture_id VARCHAR(50) UNIQUE;
                RAISE NOTICE 'Added column: api_fixture_id';
            END IF;
            
            -- Add extra time score columns if they don't exist
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='home_score_et'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN home_score_et INTEGER;
                RAISE NOTICE 'Added column: home_score_et';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='away_score_et'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN away_score_et INTEGER;
                RAISE NOTICE 'Added column: away_score_et';
            END IF;
            
            -- Add penalty score columns if they don't exist
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='home_score_pen'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN home_score_pen INTEGER;
                RAISE NOTICE 'Added column: home_score_pen';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='away_score_pen'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN away_score_pen INTEGER;
                RAISE NOTICE 'Added column: away_score_pen';
            END IF;
            
            -- Add other missing columns
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='referee'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN referee VARCHAR(255);
                RAISE NOTICE 'Added column: referee';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='venue'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN venue VARCHAR(255);
                RAISE NOTICE 'Added column: venue';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='attendance'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN attendance INTEGER;
                RAISE NOTICE 'Added column: attendance';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='winner'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN winner VARCHAR(10);
                RAISE NOTICE 'Added column: winner';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='has_extra_time'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN has_extra_time BOOLEAN DEFAULT FALSE;
                RAISE NOTICE 'Added column: has_extra_time';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='has_penalties'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN has_penalties BOOLEAN DEFAULT FALSE;
                RAISE NOTICE 'Added column: has_penalties';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='minute'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN minute INTEGER;
                RAISE NOTICE 'Added column: minute';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='stadium_id'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN stadium_id INTEGER;
                RAISE NOTICE 'Added column: stadium_id';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='stadium_name'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN stadium_name VARCHAR(255);
                RAISE NOTICE 'Added column: stadium_name';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='stadium_city'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN stadium_city VARCHAR(255);
                RAISE NOTICE 'Added column: stadium_city';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='home_formation'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN home_formation VARCHAR(20);
                RAISE NOTICE 'Added column: home_formation';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='away_formation'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN away_formation VARCHAR(20);
                RAISE NOTICE 'Added column: away_formation';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='created_at'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                RAISE NOTICE 'Added column: created_at';
            END IF;
            
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='fixtures' AND column_name='updated_at'
            ) THEN
                ALTER TABLE fixtures ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                RAISE NOTICE 'Added column: updated_at';
            END IF;
        END $$;
        """
        
        # Execute the SQL
        with db_manager.engine.connect() as conn:
            conn.execute(text(add_columns_sql))
            conn.commit()
            
        logger.info("✅ Fixtures table updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating fixtures table: {e}")
        return False


def verify_schema():
    """Verify that all required columns exist in the predictions table."""
    logger.info("Verifying database schema...")
    
    try:
        db_manager = get_db_manager()
        
        # Check columns
        check_sql = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name='predictions' 
        AND column_name IN ('home_team_sentiment', 'away_team_sentiment', 'sentiment_articles_analyzed')
        ORDER BY column_name;
        """
        
        with db_manager.engine.connect() as conn:
            result = conn.execute(text(check_sql))
            columns = result.fetchall()
            
        if len(columns) == 3:
            logger.info("✅ All sentiment columns verified in predictions table")
            for col_name, data_type in columns:
                logger.info(f"   - {col_name}: {data_type}")
            return True
        else:
            logger.warning(f"⚠️  Found {len(columns)} sentiment columns, expected 3")
            for col_name, data_type in columns:
                logger.info(f"   - {col_name}: {data_type}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error verifying schema: {e}")
        return False


def main():
    """Main function to update the database schema."""
    logger.info("🚀 FormFinder Database Schema Update")
    logger.info("=" * 40)
    
    try:
        # Load configuration
        load_config()
        logger.info("✅ Configuration loaded")
        
        # Update the schema
        predictions_success = update_predictions_table()
        fixtures_success = update_fixtures_table()
        
        if predictions_success and fixtures_success:
            # Verify the changes
            if verify_schema():
                logger.info("\n🎉 Database schema update completed successfully!")
                return True
            else:
                logger.warning("\n⚠️  Schema update completed but verification failed")
                return False
        else:
            logger.error("\n❌ Schema update failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)