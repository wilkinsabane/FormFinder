#!/usr/bin/env python3
"""
Migration script to add sentiment analysis tables to existing databases.
This script will create the new sentiment-related tables in your existing database.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import Base, get_db_manager
from formfinder.config import load_config


def create_sentiment_tables():
    """Create the new sentiment analysis tables."""
    
    print("Creating sentiment analysis tables...")
    
    try:
        # Initialize database manager
        db_manager = get_db_manager()
        engine = db_manager.engine
        
        # Create all tables (including new ones)
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")
        
        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        new_tables = [
            'sentiment_sources',
            'sentiment_data',
            'team_sentiment_scores',
            'team_form_analysis',
            'combined_analysis',
            'alert_thresholds',
            'notification_logs',
            'processing_status'
        ]
        
        existing_new_tables = [table for table in new_tables if table in table_names]
        
        if existing_new_tables:
            print(f"✅ Created tables: {', '.join(existing_new_tables)}")
        else:
            print("⚠️  No new tables were created (may already exist)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False


def setup_default_sentiment_sources():
    """Insert default sentiment sources."""
    
    from formfinder.database import SentimentSource, get_db_manager
    
    db_manager = get_db_manager()
    engine = db_manager.engine
    session = db_manager.get_session()
    
    try:
        # Check if sources already exist
        existing_sources = session.query(SentimentSource).count()
        if existing_sources > 0:
            print("ℹ️  Sentiment sources already exist, skipping insertion.")
            return True
            
        # Insert default sources
        default_sources = [
            SentimentSource(source_name="NewsAPI", source_type="news"),
            SentimentSource(source_name="Twitter", source_type="social"),
            SentimentSource(source_name="Reddit", source_type="social"),
        ]
        
        session.add_all(default_sources)
        session.commit()
        
        print("✅ Default sentiment sources added successfully!")
        return True
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error adding sentiment sources: {e}")
        return False
    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Sentiment Analysis Tables Migration")
    print("=" * 60)
    
    # Load configuration
    load_config()
    
    # Create tables
    tables_created = create_sentiment_tables()
    
    if tables_created:
        # Add default sources
        sources_added = setup_default_sentiment_sources()
        
        if sources_added:
            print("\n🎉 Migration completed successfully!")
            print("\nNext steps:")
            print("1. Run: python test_sentiment_integration.py")
            print("2. Update your config.yaml with NewsAPI key if needed")
            print("3. Restart your application")
        else:
            print("\n⚠️  Migration completed with warnings.")
    else:
        print("\n❌ Migration failed.")