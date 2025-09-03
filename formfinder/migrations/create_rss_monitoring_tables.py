"""Database migration to create RSS monitoring tables for PostgreSQL

This migration creates the necessary tables for RSS feed monitoring,
content quality tracking, and sentiment analysis metrics.
"""

import logging
from sqlalchemy import text
from ..database import get_db_session

logger = logging.getLogger(__name__)


def create_rss_monitoring_tables():
    """Create RSS monitoring tables in PostgreSQL database."""
    
    # SQL statements to create tables
    create_tables_sql = [
        # RSS Feed Health table
        """
        CREATE TABLE IF NOT EXISTS rss_feed_health (
            id SERIAL PRIMARY KEY,
            feed_url VARCHAR(500) NOT NULL,
            check_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            is_successful BOOLEAN NOT NULL,
            response_time FLOAT,
            articles_count INTEGER,
            error_message TEXT,
            http_status_code INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # RSS Content Quality table
        """
        CREATE TABLE IF NOT EXISTS rss_content_quality (
            id SERIAL PRIMARY KEY,
            feed_url VARCHAR(500) NOT NULL,
            article_url VARCHAR(1000) NOT NULL,
            recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            quality_score FLOAT NOT NULL,
            team_matches INTEGER DEFAULT 0,
            is_duplicate BOOLEAN DEFAULT FALSE,
            content_length INTEGER,
            has_image BOOLEAN DEFAULT FALSE,
            source_name VARCHAR(200),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # RSS Sentiment Analysis table
        """
        CREATE TABLE IF NOT EXISTS rss_sentiment_analysis (
            id SERIAL PRIMARY KEY,
            query VARCHAR(500) NOT NULL,
            analysis_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            source_type VARCHAR(50) NOT NULL,
            home_team VARCHAR(100),
            away_team VARCHAR(100),
            home_sentiment FLOAT,
            away_sentiment FLOAT,
            home_confidence FLOAT,
            away_confidence FLOAT,
            articles_analyzed INTEGER DEFAULT 0,
            processing_time FLOAT,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # RSS Cached Articles table
        """
        CREATE TABLE IF NOT EXISTS rss_cached_articles (
            id SERIAL PRIMARY KEY,
            url VARCHAR(1000) NOT NULL UNIQUE,
            title VARCHAR(500) NOT NULL,
            content TEXT,
            published_date TIMESTAMP WITH TIME ZONE,
            source_url VARCHAR(500),
            source_name VARCHAR(200),
            tags TEXT[],
            image_url VARCHAR(1000),
            cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    # Index creation statements for performance
    create_indexes_sql = [
        # Feed Health indexes
        "CREATE INDEX IF NOT EXISTS idx_rss_feed_health_url ON rss_feed_health(feed_url)",
        "CREATE INDEX IF NOT EXISTS idx_rss_feed_health_check_time ON rss_feed_health(check_time)",
        "CREATE INDEX IF NOT EXISTS idx_rss_feed_health_success ON rss_feed_health(is_successful)",
        
        # Content Quality indexes
        "CREATE INDEX IF NOT EXISTS idx_rss_content_quality_feed_url ON rss_content_quality(feed_url)",
        "CREATE INDEX IF NOT EXISTS idx_rss_content_quality_recorded_at ON rss_content_quality(recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_rss_content_quality_article_url ON rss_content_quality(article_url)",
        "CREATE INDEX IF NOT EXISTS idx_rss_content_quality_duplicate ON rss_content_quality(is_duplicate)",
        
        # Sentiment Analysis indexes
        "CREATE INDEX IF NOT EXISTS idx_rss_sentiment_analysis_query ON rss_sentiment_analysis(query)",
        "CREATE INDEX IF NOT EXISTS idx_rss_sentiment_analysis_time ON rss_sentiment_analysis(analysis_time)",
        "CREATE INDEX IF NOT EXISTS idx_rss_sentiment_analysis_source ON rss_sentiment_analysis(source_type)",
        "CREATE INDEX IF NOT EXISTS idx_rss_sentiment_analysis_success ON rss_sentiment_analysis(success)",
        "CREATE INDEX IF NOT EXISTS idx_rss_sentiment_analysis_teams ON rss_sentiment_analysis(home_team, away_team)",
        
        # Cached Articles indexes
        "CREATE INDEX IF NOT EXISTS idx_rss_cached_articles_url ON rss_cached_articles(url)",
        "CREATE INDEX IF NOT EXISTS idx_rss_cached_articles_source ON rss_cached_articles(source_url)",
        "CREATE INDEX IF NOT EXISTS idx_rss_cached_articles_cached_at ON rss_cached_articles(cached_at)",
        "CREATE INDEX IF NOT EXISTS idx_rss_cached_articles_expires_at ON rss_cached_articles(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_rss_cached_articles_published ON rss_cached_articles(published_date)"
    ]
    
    try:
        with get_db_session() as session:
            logger.info("Creating RSS monitoring tables...")
            
            # Create tables
            for sql in create_tables_sql:
                logger.debug(f"Executing: {sql[:50]}...")
                session.execute(text(sql))
            
            # Create indexes
            logger.info("Creating indexes for RSS monitoring tables...")
            for sql in create_indexes_sql:
                logger.debug(f"Executing: {sql}")
                session.execute(text(sql))
            
            session.commit()
            logger.info("Successfully created RSS monitoring tables and indexes")
            return True
            
    except Exception as e:
        logger.error(f"Error creating RSS monitoring tables: {e}")
        return False


def drop_rss_monitoring_tables():
    """Drop RSS monitoring tables (for rollback)."""
    
    drop_tables_sql = [
        "DROP TABLE IF EXISTS rss_sentiment_analysis CASCADE",
        "DROP TABLE IF EXISTS rss_content_quality CASCADE",
        "DROP TABLE IF EXISTS rss_feed_health CASCADE",
        "DROP TABLE IF EXISTS rss_cached_articles CASCADE"
    ]
    
    try:
        with get_db_session() as session:
            logger.info("Dropping RSS monitoring tables...")
            
            for sql in drop_tables_sql:
                logger.debug(f"Executing: {sql}")
                session.execute(text(sql))
            
            session.commit()
            logger.info("Successfully dropped RSS monitoring tables")
            return True
            
    except Exception as e:
        logger.error(f"Error dropping RSS monitoring tables: {e}")
        return False


def verify_rss_monitoring_tables():
    """Verify that RSS monitoring tables exist and are accessible."""
    
    table_names = [
        'rss_feed_health',
        'rss_content_quality', 
        'rss_sentiment_analysis',
        'rss_cached_articles'
    ]
    
    try:
        with get_db_session() as session:
            logger.info("Verifying RSS monitoring tables...")
            
            for table_name in table_names:
                result = session.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"
                ), {'table_name': table_name}).scalar()
                
                if result:
                    logger.info(f"✓ Table '{table_name}' exists")
                else:
                    logger.error(f"✗ Table '{table_name}' does not exist")
                    return False
            
            logger.info("All RSS monitoring tables verified successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error verifying RSS monitoring tables: {e}")
        return False


if __name__ == "__main__":
    # Load configuration first
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from formfinder.config import load_config
    load_config()
    
    # Run migration when script is executed directly
    logging.basicConfig(level=logging.INFO)
    
    print("Creating RSS monitoring tables...")
    if create_rss_monitoring_tables():
        print("✓ Tables created successfully")
        
        print("Verifying tables...")
        if verify_rss_monitoring_tables():
            print("✓ All tables verified")
        else:
            print("✗ Table verification failed")
    else:
        print("✗ Failed to create tables")