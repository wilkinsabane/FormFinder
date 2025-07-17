#!/usr/bin/env python3
"""
Test PostgreSQL connection with new credentials.
"""

import logging
from formfinder.config import load_config, get_config
from formfinder.database import get_db_session, League, Team, Standing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_postgresql_connection():
    """Test PostgreSQL connection and basic operations."""
    try:
        # Load configuration
        load_config("config.yaml")
        config = get_config()
        
        logger.info(f"Testing connection to PostgreSQL database: {config.database.postgresql.database}")
        logger.info(f"Host: {config.database.postgresql.host}")
        logger.info(f"Username: {config.database.postgresql.username}")
        
        # Test database connection
        with get_db_session() as session:
            # Test basic query
            leagues_count = session.query(League).count()
            teams_count = session.query(Team).count()
            standings_count = session.query(Standing).count()
            
            logger.info("✅ PostgreSQL connection successful!")
            logger.info(f"Database contents:")
            logger.info(f"  - Leagues: {leagues_count}")
            logger.info(f"  - Teams: {teams_count}")
            logger.info(f"  - Standings: {standings_count}")
            
            # Test a sample query
            if leagues_count > 0:
                sample_leagues = session.query(League).limit(3).all()
                logger.info("Sample leagues:")
                for league in sample_leagues:
                    logger.info(f"  - {league.name} ({league.country}) - Season: {league.season}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {str(e)}")
        logger.error("Please check:")
        logger.error("  1. PostgreSQL server is running")
        logger.error("  2. Database 'formfinder' exists")
        logger.error("  3. User 'wilkins' has access to the database")
        logger.error("  4. Host 'your_postgres_host' is accessible")
        logger.error("  5. Password is correct")
        return False

if __name__ == "__main__":
    logger.info("=== PostgreSQL Connection Test ===")
    success = test_postgresql_connection()
    
    if success:
        logger.info("\n🎉 PostgreSQL setup is ready!")
        logger.info("You can now run: python main.py")
    else:
        logger.info("\n⚠️  Please fix the connection issues before proceeding.")