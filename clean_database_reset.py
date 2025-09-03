#!/usr/bin/env python3
"""
Clean Database Reset for FormFinder.

This script provides a more controlled database reset by:
1. Connecting directly to PostgreSQL
2. Dropping all tables in the formfinder database
3. Recreating the schema
4. Optionally fetching fresh data

Usage:
    python clean_database_reset.py [--yes] [--no-fetch]

Arguments:
    --yes: Skip confirmation prompt
    --no-fetch: Don't fetch new data after reset
"""

import sys
import logging
import click
from sqlalchemy import text
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from formfinder.database import DatabaseManager
from formfinder.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def drop_all_tables(db_manager):
    """Drop all tables in the database."""
    logger.info("🗑️  Dropping all tables...")
    
    try:
        with db_manager.get_session() as session:
            # Get list of all tables
            result = session.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """))
            
            tables = [row[0] for row in result]
            
            if not tables:
                logger.info("No tables found to drop")
                return True
            
            # Drop tables in correct order to handle foreign keys
            drop_order = [
                'predictions', 'high_form_teams', 'standings', 
                'fixtures', 'teams', 'leagues', 'data_fetch_logs'
            ]
            
            for table in drop_order:
                if table in tables:
                    try:
                        session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                        logger.info(f"Dropped table: {table}")
                    except Exception as e:
                        logger.warning(f"Could not drop {table}: {e}")
            
            session.commit()
            logger.info("✅ All tables dropped successfully")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error dropping tables: {e}")
        return False


def create_schema(db_manager):
    """Create the database schema."""
    logger.info("🏗️  Creating database schema...")
    
    try:
        db_manager.create_tables()
        logger.info("✅ Schema created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating schema: {e}")
        return False


def reset_sequences(db_manager):
    """Reset all sequences to start from 1."""
    logger.info("🔄 Resetting database sequences...")
    
    try:
        with db_manager.get_session() as session:
            # Get all sequences
            result = session.execute(text("""
                SELECT sequence_name 
                FROM information_schema.sequences 
                WHERE sequence_schema = 'public'
            """))
            
            sequences = [row[0] for row in result]
            
            for seq in sequences:
                try:
                    session.execute(text(f"ALTER SEQUENCE {seq} RESTART WITH 1"))
                    logger.info(f"Reset sequence: {seq}")
                except Exception as e:
                    logger.warning(f"Could not reset {seq}: {e}")
            
            session.commit()
            logger.info("✅ Sequences reset successfully")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error resetting sequences: {e}")
        return False


def get_database_stats(db_manager):
    """Get current database statistics."""
    try:
        with db_manager.get_session() as session:
            # Count tables
            result = session.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """))
            tables = [row[0] for row in result]
            
            stats = {"tables": len(tables), "table_names": tables}
            
            # Count records in each table
            for table in tables:
                try:
                    count_result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = count_result.scalar()
                    stats[table] = count
                except:
                    stats[table] = 0
            
            return stats
            
    except Exception as e:
        logger.error(f"Could not get database stats: {e}")
        return {"tables": 0, "table_names": []}


def fetch_fresh_data():
    """Fetch fresh data using the main pipeline."""
    logger.info("📊 Fetching fresh data...")
    
    try:
        import subprocess
        
        result = subprocess.run([
            sys.executable, 'main.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Fresh data fetched successfully")
            if result.stdout:
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Data fetching failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error fetching data: {e}")
        return False


@click.command()
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
@click.option('--no-fetch', is_flag=True, help='Don\'t fetch new data after reset')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(yes, no_fetch, verbose):
    """Clean reset of FormFinder database."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🗄️  FormFinder Database Clean Reset")
    print("=" * 50)
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager()
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)
    
    # Show current status
    stats = get_database_stats(db_manager)
    print(f"📊 Current database has {stats['tables']} tables")
    if stats['table_names']:
        for table in stats['table_names']:
            count = stats.get(table, 0)
            print(f"  • {table}: {count} records")
    
    if not yes:
        print("\n⚠️  WARNING: This will permanently delete all data!")
        if not click.confirm("Are you sure you want to proceed?"):
            print("❌ Reset cancelled")
            return
    
    # Perform reset
    print("\n🔄 Starting database reset...")
    
    # Step 1: Drop all tables
    if not drop_all_tables(db_manager):
        print("❌ Failed to drop tables")
        sys.exit(1)
    
    # Step 2: Create new schema
    if not create_schema(db_manager):
        print("❌ Failed to create schema")
        sys.exit(1)
    
    # Step 3: Reset sequences
    if not reset_sequences(db_manager):
        print("⚠️  Failed to reset sequences (continuing...)")
    
    # Step 4: Fetch fresh data (optional)
    if not no_fetch:
        print("\n📊 Fetching fresh data...")
        if not fetch_fresh_data():
            print("⚠️  Data fetching failed - you can retry with: python main.py")
    
    # Show final status
    stats = get_database_stats(db_manager)
    print(f"\n✅ Reset completed!")
    print(f"📊 Database now has {stats['tables']} tables")
    
    print("\n📋 Next steps:")
    print("  • Run dashboard: python dashboard.py")
    print("  • Check status: python -m formfinder.cli db status")
    print("  • Fetch data: python main.py")


if __name__ == "__main__":
    main()