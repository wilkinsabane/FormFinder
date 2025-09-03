#!/usr/bin/env python3
"""
Simple Database Reset for FormFinder.

This script provides a clean database reset using direct SQLAlchemy connection.
"""

import os
import sys
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_database_url():
    """Get database URL from configuration."""
    # Try environment variables first
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url
    
    # Try .env file
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('DATABASE_URL='):
                    return line.split('=', 1)[1].strip().strip('"\'')
    
    # Try Streamlit secrets
    secrets_path = Path('.streamlit/secrets.toml')
    if secrets_path.exists():
        try:
            import tomllib
            with open(secrets_path, 'rb') as f:
                secrets = tomllib.load(f)
                db_url = secrets.get('DB_URI')
                if db_url:
                    return db_url
        except ImportError:
            # Handle older Python versions
            pass
    
    # Default PostgreSQL connection
    return "postgresql://wilkins:Holmes&7watson@localhost:5432/formfinder"


def reset_database():
    """Reset the database using direct SQLAlchemy connection."""
    try:
        from sqlalchemy import create_engine, text
        
        # Get database URL
        database_url = get_database_url()
        logger.info(f"Connecting to database...")
        
        # Create engine
        engine = create_engine(database_url)
        
        # Connect and reset
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Get all tables
                result = conn.execute(text("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                """))
                
                tables = [row[0] for row in result]
                
                if tables:
                    logger.info(f"Found {len(tables)} tables: {', '.join(tables)}")
                    
                    # Drop tables in correct order (handle foreign keys)
                    drop_order = [
                        'predictions', 'high_form_teams', 'standings', 
                        'fixtures', 'teams', 'leagues', 'data_fetch_logs'
                    ]
                    
                    dropped_count = 0
                    for table in drop_order:
                        if table in tables:
                            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                            logger.info(f"Dropped table: {table}")
                            dropped_count += 1
                    
                    if dropped_count == 0:
                        # If no specific tables found, drop all
                        for table in tables:
                            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                            logger.info(f"Dropped table: {table}")
                
                # Reset sequences
                result = conn.execute(text("""
                    SELECT sequence_name 
                    FROM information_schema.sequences 
                    WHERE sequence_schema = 'public'
                """))
                
                sequences = [row[0] for row in result]
                for seq in sequences:
                    conn.execute(text(f"ALTER SEQUENCE {seq} RESTART WITH 1"))
                    logger.info(f"Reset sequence: {seq}")
                
                trans.commit()
                logger.info("✅ Database reset completed successfully")
                return True
                
            except Exception as e:
                trans.rollback()
                logger.error(f"❌ Error during reset: {e}")
                return False
                
    except ImportError:
        logger.error("SQLAlchemy not found. Install with: pip install sqlalchemy")
        return False
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def create_tables():
    """Create database tables using the formfinder models."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Import and load configuration
        from formfinder.config import load_config
        load_config()
        
        from formfinder.database import DatabaseManager, Base
        
        # Create tables
        db = DatabaseManager()
        Base.metadata.create_all(db.engine)
        logger.info("✅ Tables created successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Creating tables manually...")
        return create_tables_manual()
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False


def create_tables_manual():
    """Create tables manually without formfinder imports."""
    try:
        from sqlalchemy import create_engine, text
        
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        # Create tables SQL based on formfinder schema
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS leagues (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            country VARCHAR(100),
            season INTEGER,
            start_date DATE,
            end_date DATE,
            current_matchday INTEGER,
            api_id INTEGER UNIQUE
        );

        CREATE TABLE IF NOT EXISTS teams (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            short_name VARCHAR(10),
            tla VARCHAR(5),
            crest_url TEXT,
            founded INTEGER,
            venue_name VARCHAR(100),
            venue_capacity INTEGER,
            league_id INTEGER REFERENCES leagues(id),
            api_id INTEGER UNIQUE
        );

        CREATE TABLE IF NOT EXISTS fixtures (
            id SERIAL PRIMARY KEY,
            matchday INTEGER,
            utc_date TIMESTAMP,
            status VARCHAR(20),
            matchday INTEGER,
            stage VARCHAR(50),
            group_name VARCHAR(50),
            last_updated TIMESTAMP,
            home_team_id INTEGER REFERENCES teams(id),
            away_team_id INTEGER REFERENCES teams(id),
            league_id INTEGER REFERENCES leagues(id),
            api_id INTEGER UNIQUE
        );

        CREATE TABLE IF NOT EXISTS standings (
            id SERIAL PRIMARY KEY,
            stage VARCHAR(50),
            type VARCHAR(20),
            group_name VARCHAR(50),
            position INTEGER,
            points INTEGER,
            played_games INTEGER,
            won INTEGER,
            draw INTEGER,
            lost INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            goal_difference INTEGER,
            team_id INTEGER REFERENCES teams(id),
            league_id INTEGER REFERENCES leagues(id)
        );

        CREATE TABLE IF NOT EXISTS high_form_teams (
            id SERIAL PRIMARY KEY,
            team_name VARCHAR(100),
            league_name VARCHAR(100),
            form_score FLOAT,
            last_updated TIMESTAMP,
            team_id INTEGER REFERENCES teams(id),
            league_id INTEGER REFERENCES leagues(id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            fixture_id INTEGER REFERENCES fixtures(id),
            predicted_winner VARCHAR(100),
            confidence FLOAT,
            prediction_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS data_fetch_logs (
            id SERIAL PRIMARY KEY,
            operation VARCHAR(50),
            league_name VARCHAR(100),
            status VARCHAR(20),
            records_processed INTEGER,
            error_message TEXT,
            fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with engine.connect() as conn:
            conn.execute(text(create_tables_sql))
            conn.commit()
            
        logger.info("✅ Tables created manually")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating tables manually: {e}")
        return False


def check_connection():
    """Test database connection."""
    try:
        from sqlalchemy import create_engine, text
        
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def show_status():
    """Show current database status."""
    try:
        from sqlalchemy import create_engine, text
        
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Get table count
            result = conn.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """))
            
            tables = [row[0] for row in result]
            
            print(f"📊 Database Status:")
            print(f"  URL: {database_url}")
            print(f"  Tables found: {len(tables)}")
            
            if tables:
                print(f"  Table names: {', '.join(sorted(tables))}")
                
                # Count records in each table
                for table in sorted(tables):
                    try:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                        print(f"    {table}: {count} records")
                    except Exception as e:
                        print(f"    {table}: error counting - {e}")
            else:
                print("  ✅ Database is clean (no tables)")
                
    except Exception as e:
        print(f"❌ Could not get status: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple database reset for FormFinder')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--reset', action='store_true', help='Reset database')
    parser.add_argument('--create', action='store_true', help='Create tables')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        sys.exit(0)
    
    if args.reset:
        print("🗄️  FormFinder Database Reset")
        print("=" * 40)
        
        # Check connection
        if not check_connection():
            print("❌ Cannot connect to database")
            sys.exit(1)
        
        # Show current status
        show_status()
        
        if not args.yes:
            response = input("\n⚠️  This will delete all data. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Cancelled")
                sys.exit(0)
        
        # Reset database
        if reset_database():
            print("✅ Database reset completed")
            
            if args.create:
                if create_tables():
                    print("✅ Tables created successfully")
                else:
                    print("⚠️  Table creation failed")
            
            # Show final status
            show_status()
            
            print("\n📋 Next steps:")
            print("  • Create tables: python simple_db_reset.py --create")
            print("  • Reset + create: python simple_db_reset.py --reset --create")
            print("  • Fetch data: python main.py")
        else:
            print("❌ Database reset failed")
            sys.exit(1)
    
    elif args.create:
        if create_tables():
            print("✅ Tables created successfully")
        else:
            print("❌ Table creation failed")
    
    else:
        # Default: show status
        show_status()
        print("\nUsage:")
        print("  python simple_db_reset.py --status    # Show current status")
        print("  python simple_db_reset.py --reset     # Reset database")
        print("  python simple_db_reset.py --create    # Create tables")
        print("  python simple_db_reset.py --reset --create --yes  # Full reset")