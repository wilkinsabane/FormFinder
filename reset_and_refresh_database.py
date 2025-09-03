#!/usr/bin/env python3
"""
Database Reset and Refresh Script for FormFinder.

This script completely resets the database by:
1. Dropping all existing tables
2. Recreating the schema
3. Fetching fresh data from APIs
4. Populating the database with new data

Usage:
    python reset_and_refresh_database.py [--force] [--skip-fetch]

Arguments:
    --force: Skip confirmation prompt
    --skip-fetch: Only reset database, don't fetch new data
"""

import sys
import logging
import click
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_database_connection():
    """Test database connection before proceeding."""
    try:
        from formfinder.database import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.get_session().close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def reset_database():
    """Reset the database using CLI commands."""
    logger.info("🔄 Starting database reset...")
    
    try:
        # Use the CLI reset command
        result = subprocess.run([
            sys.executable, '-m', 'formfinder.cli', 'db', 'reset'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("✅ Database reset completed successfully")
            if result.stdout:
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Database reset failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}")
        return False


def fetch_fresh_data():
    """Fetch fresh data from APIs."""
    logger.info("📊 Fetching fresh data from APIs...")
    
    try:
        # Run the main pipeline to fetch new data
        result = subprocess.run([
            sys.executable, 'main.py', '--fetch-only'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
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


def migrate_csv_data():
    """Migrate any existing CSV data to the database."""
    logger.info("📁 Migrating CSV data to database...")
    
    try:
        # Run the CSV migration script
        result = subprocess.run([
            sys.executable, 'DB Migration Files/migrate_csv_to_db.py'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("✅ CSV data migration completed")
            if result.stdout:
                logger.info(result.stdout)
            return True
        else:
            logger.warning(f"⚠️ CSV migration had issues: {result.stderr}")
            # Don't fail the entire process if CSV migration fails
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ CSV migration failed: {e}")
        return True  # Continue even if CSV migration fails


def show_database_status():
    """Display current database status."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'formfinder.cli', 'db', 'status'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("📋 Current database status:")
            logger.info(result.stdout)
            return True
        else:
            logger.warning("Could not get database status")
            return False
            
    except Exception as e:
        logger.warning(f"Could not get database status: {e}")
        return False


@click.command()
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
@click.option('--skip-fetch', is_flag=True, help='Only reset database, don\'t fetch new data')
@click.option('--skip-csv', is_flag=True, help='Skip CSV migration step')
def main(force, skip_fetch, skip_csv):
    """Reset and refresh the FormFinder database."""
    
    print("🗄️  FormFinder Database Reset and Refresh")
    print("=" * 50)
    
    if not force:
        # Show current status before reset
        show_database_status()
        
        print("\n⚠️  WARNING: This will completely reset your database!")
        print("All existing data will be permanently deleted.")
        
        if not click.confirm("Do you want to proceed with the reset?"):
            print("❌ Reset cancelled by user")
            return
    
    # Check database connection
    if not check_database_connection():
        print("❌ Cannot proceed - database connection failed")
        sys.exit(1)
    
    # Step 1: Reset database
    if not reset_database():
        print("❌ Database reset failed")
        sys.exit(1)
    
    if not skip_fetch:
        # Step 2: Fetch fresh data
        if not fetch_fresh_data():
            print("❌ Data fetching failed")
            print("You can retry data fetching with: python main.py")
            sys.exit(1)
    
    if not skip_csv:
        # Step 3: Migrate CSV data (optional)
        migrate_csv_data()
    
    # Show final status
    print("\n🎉 Database reset and refresh completed!")
    show_database_status()
    
    print("\n📋 Next steps:")
    print("  • Run the dashboard: python dashboard.py")
    print("  • Check data: python -m formfinder.cli db status")
    print("  • Fetch more data: python main.py")


if __name__ == "__main__":
    main()