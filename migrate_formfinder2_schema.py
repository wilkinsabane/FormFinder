#!/usr/bin/env python3
"""
FormFinder2 Database Schema Migration Script

This script migrates the existing FormFinder database to support the new FormFinder2
architecture with enhanced features for data separation, training, and monitoring.

New tables added:
- pre_computed_features: Store pre-computed features for training
- feature_computation_queue: Task queue for feature computation
- health_checks: System health monitoring results
- alerts: System alerts and notifications
- performance_metrics: Performance monitoring data
- scheduled_jobs: Scheduled job definitions and history

Usage:
    python migrate_formfinder2_schema.py [--dry-run] [--backup]
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from formfinder.config import get_config
from formfinder.database import (
    Base, PreComputedFeatures, FeatureComputationQueue, HealthChecks,
    Alerts, PerformanceMetrics, ScheduledJobs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FormFinder2Migration:
    """Handle FormFinder2 database schema migration."""
    
    def __init__(self, config=None):
        """Initialize migration with database configuration."""
        self.config = config or get_config()
        self.database_url = self.config.get_database_url()
        self.engine = create_engine(self.database_url)
        self.inspector = inspect(self.engine)
        
        # New tables to be created
        self.new_tables = [
            'pre_computed_features',
            'feature_computation_queue', 
            'health_checks',
            'alerts',
            'performance_metrics',
            'scheduled_jobs'
        ]
        
        logger.info(f"Initialized migration for database: {self.database_url}")
    
    def backup_database(self, backup_path=None):
        """Create a backup of the database before migration."""
        if not backup_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"formfinder_backup_{timestamp}.db"
        
        if self.database_url.startswith('sqlite'):
            # Extract database path from SQLite URL
            db_path = self.database_url.replace('sqlite:///', '')
            if os.path.exists(db_path):
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backed up to: {backup_path}")
                return backup_path
            else:
                logger.warning(f"Database file not found: {db_path}")
                return None
        else:
            logger.warning("Backup not implemented for non-SQLite databases")
            logger.info("Please create a manual backup before proceeding")
            return None
    
    def check_existing_tables(self):
        """Check which tables already exist in the database."""
        existing_tables = self.inspector.get_table_names()
        logger.info(f"Found {len(existing_tables)} existing tables")
        
        # Check for core FormFinder tables
        core_tables = ['leagues', 'teams', 'fixtures', 'standings', 'predictions']
        missing_core = [table for table in core_tables if table not in existing_tables]
        
        if missing_core:
            logger.error(f"Missing core FormFinder tables: {missing_core}")
            logger.error("This doesn't appear to be a valid FormFinder database")
            return False
        
        logger.info("Core FormFinder tables found ✓")
        
        # Check which new tables already exist
        existing_new = [table for table in self.new_tables if table in existing_tables]
        if existing_new:
            logger.warning(f"Some new tables already exist: {existing_new}")
        
        return True
    
    def create_new_tables(self, dry_run=False):
        """Create the new FormFinder2 tables."""
        if dry_run:
            logger.info("DRY RUN: Would create the following new tables:")
            for table in self.new_tables:
                logger.info(f"  - {table}")
            return True
        
        try:
            # Create only the new tables
            new_table_classes = [
                PreComputedFeatures,
                FeatureComputationQueue,
                HealthChecks,
                Alerts,
                PerformanceMetrics,
                ScheduledJobs
            ]
            
            for table_class in new_table_classes:
                table_name = table_class.__tablename__
                if table_name not in self.inspector.get_table_names():
                    logger.info(f"Creating table: {table_name}")
                    table_class.__table__.create(self.engine)
                    logger.info(f"✓ Created table: {table_name}")
                else:
                    logger.info(f"Table already exists: {table_name}")
            
            logger.info("All new tables created successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {e}")
            return False
    
    def verify_migration(self):
        """Verify that the migration was successful."""
        logger.info("Verifying migration...")
        
        existing_tables = self.inspector.get_table_names()
        
        # Check that all new tables exist
        missing_tables = [table for table in self.new_tables if table not in existing_tables]
        if missing_tables:
            logger.error(f"Migration verification failed. Missing tables: {missing_tables}")
            return False
        
        # Check table structures
        for table in self.new_tables:
            try:
                columns = self.inspector.get_columns(table)
                logger.info(f"✓ Table '{table}' has {len(columns)} columns")
            except Exception as e:
                logger.error(f"Error checking table '{table}': {e}")
                return False
        
        logger.info("Migration verification successful ✓")
        return True
    
    def add_initial_data(self, dry_run=False):
        """Add initial data to new tables."""
        if dry_run:
            logger.info("DRY RUN: Would add initial data to new tables")
            return True
        
        try:
            with self.engine.connect() as conn:
                # Add initial scheduled jobs
                initial_jobs = [
                    {
                        'job_name': 'daily_feature_computation',
                        'job_type': 'feature_computation',
                        'cron_expression': '0 2 * * *',  # Daily at 2 AM
                        'is_enabled': True,
                        'job_parameters': '{"leagues": "all", "days_back": 30}',
                        'timeout_seconds': 7200,  # 2 hours
                        'max_retries': 2
                    },
                    {
                        'job_name': 'weekly_model_training',
                        'job_type': 'model_training',
                        'cron_expression': '0 3 * * 0',  # Weekly on Sunday at 3 AM
                        'is_enabled': True,
                        'job_parameters': '{"retrain_threshold": 0.05}',
                        'timeout_seconds': 10800,  # 3 hours
                        'max_retries': 1
                    },
                    {
                        'job_name': 'hourly_health_check',
                        'job_type': 'health_check',
                        'cron_expression': '0 * * * *',  # Every hour
                        'is_enabled': True,
                        'job_parameters': '{"checks": ["database", "api", "disk", "memory"]}',
                        'timeout_seconds': 300,  # 5 minutes
                        'max_retries': 3
                    }
                ]
                
                for job in initial_jobs:
                    # Check if job already exists
                    result = conn.execute(
                        text("SELECT COUNT(*) FROM scheduled_jobs WHERE job_name = :name"),
                        {'name': job['job_name']}
                    ).scalar()
                    
                    if result == 0:
                        conn.execute(
                            text("""
                                INSERT INTO scheduled_jobs 
                                (job_name, job_type, cron_expression, is_enabled, 
                                 job_parameters, timeout_seconds, max_retries, created_at, updated_at)
                                VALUES 
                                (:job_name, :job_type, :cron_expression, :is_enabled,
                                 :job_parameters, :timeout_seconds, :max_retries, :now, :now)
                            """),
                            {
                                **job,
                                'now': datetime.utcnow()
                            }
                        )
                        logger.info(f"Added initial job: {job['job_name']}")
                    else:
                        logger.info(f"Job already exists: {job['job_name']}")
                
                conn.commit()
                logger.info("Initial data added successfully")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error adding initial data: {e}")
            return False
    
    def run_migration(self, dry_run=False, backup=True):
        """Run the complete migration process."""
        logger.info("🚀 Starting FormFinder2 database migration")
        
        # Step 1: Check existing database
        if not self.check_existing_tables():
            logger.error("Pre-migration checks failed")
            return False
        
        # Step 2: Create backup
        if backup and not dry_run:
            backup_path = self.backup_database()
            if backup_path:
                logger.info(f"Backup created: {backup_path}")
        
        # Step 3: Create new tables
        if not self.create_new_tables(dry_run):
            logger.error("Table creation failed")
            return False
        
        # Step 4: Add initial data
        if not self.add_initial_data(dry_run):
            logger.error("Initial data insertion failed")
            return False
        
        # Step 5: Verify migration
        if not dry_run:
            if not self.verify_migration():
                logger.error("Migration verification failed")
                return False
        
        logger.info("🎉 FormFinder2 migration completed successfully!")
        
        if not dry_run:
            logger.info("\nNext steps:")
            logger.info("1. Update your configuration to use the new features")
            logger.info("2. Run feature pre-computation: python -m formfinder.main features")
            logger.info("3. Start the scheduler: python -m formfinder.main daemon")
            logger.info("4. Monitor system health: python -m formfinder.main health")
        
        return True
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'engine'):
            self.engine.dispose()


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description='Migrate FormFinder database to FormFinder2 schema'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true', 
        help='Skip database backup (not recommended)'
    )
    parser.add_argument(
        '--config-file',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config_file:
            # Custom config loading would go here
            config = get_config()
        else:
            config = get_config()
        
        # Run migration
        migration = FormFinder2Migration(config)
        
        success = migration.run_migration(
            dry_run=args.dry_run,
            backup=not args.no_backup
        )
        
        migration.close()
        
        if success:
            logger.info("Migration completed successfully")
            sys.exit(0)
        else:
            logger.error("Migration failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()