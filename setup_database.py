#!/usr/bin/env python3
"""
Database setup script for FormFinder.
Automatically sets up PostgreSQL using Docker and migrates existing data.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Docker found: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Docker not found")
            return False
    except FileNotFoundError:
        logger.error("❌ Docker not installed")
        return False

def check_docker_compose():
    """Check if Docker Compose is available."""
    try:
        # Try docker compose (newer syntax)
        result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Docker Compose found: {result.stdout.strip()}")
            return 'docker compose'
        else:
            # Try docker-compose (older syntax)
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Docker Compose found: {result.stdout.strip()}")
                return 'docker-compose'
            else:
                logger.error("❌ Docker Compose not found")
                return None
    except FileNotFoundError:
        logger.error("❌ Docker Compose not installed")
        return None

def start_postgresql(compose_cmd):
    """Start PostgreSQL using Docker Compose."""
    logger.info("🚀 Starting PostgreSQL with Docker...")
    
    try:
        if compose_cmd == 'docker compose':
            cmd = ['docker', 'compose', '-f', 'docker-compose.postgres.yml', 'up', '-d']
        else:
            cmd = ['docker-compose', '-f', 'docker-compose.postgres.yml', 'up', '-d']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ PostgreSQL started successfully")
            logger.info("Waiting for PostgreSQL to be ready...")
            
            # Wait for PostgreSQL to be ready
            for i in range(30):  # Wait up to 30 seconds
                try:
                    health_cmd = ['docker', 'exec', 'formfinder_postgres', 'pg_isready', '-U', 'wilkins', '-d', 'formfinder']
                    health_result = subprocess.run(health_cmd, capture_output=True, text=True)
                    if health_result.returncode == 0:
                        logger.info("✅ PostgreSQL is ready!")
                        return True
                except:
                    pass
                
                time.sleep(1)
                print(".", end="", flush=True)
            
            logger.warning("⚠️  PostgreSQL may not be fully ready yet, but continuing...")
            return True
        else:
            logger.error(f"❌ Failed to start PostgreSQL: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error starting PostgreSQL: {str(e)}")
        return False

def test_connection():
    """Test PostgreSQL connection."""
    logger.info("🔍 Testing PostgreSQL connection...")
    
    try:
        result = subprocess.run([sys.executable, 'test_postgresql_connection.py'], 
                              capture_output=True, text=True)
        
        if "✅ PostgreSQL connection successful!" in result.stdout:
            logger.info("✅ Connection test passed!")
            return True
        else:
            logger.error("❌ Connection test failed")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing connection: {str(e)}")
        return False

def migrate_data():
    """Migrate existing CSV data to PostgreSQL."""
    logger.info("📊 Migrating existing CSV data to PostgreSQL...")
    
    try:
        result = subprocess.run([sys.executable, 'migrate_csv_to_db.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Data migration completed successfully!")
            # Extract summary from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'leagues' in line.lower() or 'teams' in line.lower() or 'standings' in line.lower():
                    logger.info(f"  {line.strip()}")
            return True
        else:
            logger.error("❌ Data migration failed")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during migration: {str(e)}")
        return False

def show_next_steps():
    """Show next steps to the user."""
    logger.info("\n🎉 Database setup completed successfully!")
    logger.info("\n📋 Next steps:")
    logger.info("  1. Run: python main.py")
    logger.info("  2. Your FormFinder is now using PostgreSQL!")
    logger.info("\n🔧 Useful commands:")
    logger.info("  • Check database status: docker ps")
    logger.info("  • View database logs: docker logs formfinder_postgres")
    logger.info("  • Connect to database: docker exec -it formfinder_postgres psql -U wilkins -d formfinder")
    logger.info("  • Stop database: docker compose -f docker-compose.postgres.yml down")
    logger.info("  • Restart database: docker compose -f docker-compose.postgres.yml restart")

def main():
    """Main setup function."""
    logger.info("🚀 FormFinder PostgreSQL Setup")
    logger.info("=" * 40)
    
    # Check prerequisites
    if not check_docker():
        logger.error("\n❌ Docker is required. Please install Docker Desktop:")
        logger.error("   https://www.docker.com/products/docker-desktop")
        return False
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        logger.error("\n❌ Docker Compose is required. Please install Docker Compose.")
        return False
    
    # Check if files exist
    required_files = ['docker-compose.postgres.yml', 'test_postgresql_connection.py', 'migrate_csv_to_db.py']
    for file in required_files:
        if not Path(file).exists():
            logger.error(f"❌ Required file not found: {file}")
            return False
    
    # Start PostgreSQL
    if not start_postgresql(compose_cmd):
        return False
    
    # Test connection
    if not test_connection():
        logger.error("\n❌ Setup failed. Please check the logs above.")
        return False
    
    # Migrate data
    if not migrate_data():
        logger.warning("⚠️  Data migration failed, but PostgreSQL is running.")
        logger.warning("   You can try running 'python migrate_csv_to_db.py' manually.")
    
    # Show next steps
    show_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)