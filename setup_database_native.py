#!/usr/bin/env python3
"""
Native PostgreSQL Setup Script for FormFinder
Sets up PostgreSQL database without Docker
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)

def run_command(command, shell=True, capture_output=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            capture_output=capture_output, 
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_postgresql_installed():
    """Check if PostgreSQL is installed"""
    print_step(1, "Checking PostgreSQL Installation")
    
    # Check if psql command is available
    success, stdout, stderr = run_command("psql --version")
    
    if success:
        version = stdout.strip()
        print(f"✅ PostgreSQL found: {version}")
        return True
    else:
        print("❌ PostgreSQL not found in PATH")
        return False

def get_installation_instructions():
    """Get OS-specific installation instructions"""
    system = platform.system().lower()
    
    print_step(2, "PostgreSQL Installation Instructions")
    
    if system == "windows":
        print("Windows Installation Options:")
        print("\n1. Official Installer (Recommended):")
        print("   - Download from: https://www.postgresql.org/download/windows/")
        print("   - Run installer as Administrator")
        print("   - Remember the postgres user password!")
        print("   - Default port: 5432")
        
        print("\n2. Chocolatey:")
        print("   choco install postgresql")
        
        print("\n3. Scoop:")
        print("   scoop install postgresql")
        
    elif system == "darwin":  # macOS
        print("macOS Installation Options:")
        print("\n1. Homebrew (Recommended):")
        print("   brew install postgresql@15")
        print("   brew services start postgresql@15")
        
        print("\n2. Postgres.app:")
        print("   - Download from: https://postgresapp.com/")
        print("   - Drag to Applications and launch")
        
    else:  # Linux
        print("Linux Installation:")
        print("\nUbuntu/Debian:")
        print("   sudo apt update")
        print("   sudo apt install postgresql postgresql-contrib")
        print("   sudo systemctl start postgresql")
        print("   sudo systemctl enable postgresql")
        
        print("\nCentOS/RHEL/Fedora:")
        print("   sudo dnf install postgresql postgresql-server postgresql-contrib")
        print("   sudo postgresql-setup --initdb")
        print("   sudo systemctl start postgresql")
        print("   sudo systemctl enable postgresql")

def check_postgresql_service():
    """Check if PostgreSQL service is running"""
    print_step(3, "Checking PostgreSQL Service Status")
    
    system = platform.system().lower()
    
    if system == "windows":
        # Check Windows service
        success, stdout, stderr = run_command('sc query "postgresql-x64-15"')
        if success and "RUNNING" in stdout:
            print("✅ PostgreSQL service is running")
            return True
        else:
            print("❌ PostgreSQL service is not running")
            print("\nTo start PostgreSQL on Windows:")
            print('   net start "postgresql-x64-15"')
            print("   Or use Services.msc to start the service")
            return False
            
    elif system == "darwin":  # macOS
        success, stdout, stderr = run_command("brew services list | grep postgresql")
        if success and "started" in stdout:
            print("✅ PostgreSQL service is running")
            return True
        else:
            print("❌ PostgreSQL service is not running")
            print("\nTo start PostgreSQL on macOS:")
            print("   brew services start postgresql@15")
            return False
            
    else:  # Linux
        success, stdout, stderr = run_command("systemctl is-active postgresql")
        if success and "active" in stdout.strip():
            print("✅ PostgreSQL service is running")
            return True
        else:
            print("❌ PostgreSQL service is not running")
            print("\nTo start PostgreSQL on Linux:")
            print("   sudo systemctl start postgresql")
            return False

def test_postgres_connection():
    """Test connection to PostgreSQL as postgres user"""
    print_step(4, "Testing PostgreSQL Connection")
    
    # Try to connect as postgres user
    success, stdout, stderr = run_command('psql -U postgres -c "SELECT version();"')
    
    if success:
        print("✅ Successfully connected to PostgreSQL")
        return True
    else:
        print("❌ Failed to connect to PostgreSQL")
        print(f"Error: {stderr}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check if 'postgres' user exists")
        print("3. Verify authentication settings in pg_hba.conf")
        return False

def create_database_and_user():
    """Create the FormFinder database and user"""
    print_step(5, "Creating Database and User")
    
    sql_commands = [
        "CREATE USER wilkins WITH PASSWORD 'Holmes&7watson';",
        "CREATE DATABASE formfinder OWNER wilkins;",
        "GRANT ALL PRIVILEGES ON DATABASE formfinder TO wilkins;"
    ]
    
    print("Creating user 'wilkins' and database 'formfinder'...")
    
    for i, command in enumerate(sql_commands, 1):
        print(f"  {i}. {command.split()[0]} {command.split()[1]}...")
        
        success, stdout, stderr = run_command(
            f'psql -U postgres -c "{command}"'
        )
        
        if not success:
            if "already exists" in stderr:
                print(f"    ⚠️  Already exists (skipping)")
            else:
                print(f"    ❌ Failed: {stderr}")
                return False
        else:
            print(f"    ✅ Success")
    
    # Grant additional privileges
    print("\nGranting additional privileges...")
    additional_commands = [
        "\\c formfinder",
        "GRANT ALL ON SCHEMA public TO wilkins;",
        "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO wilkins;",
        "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO wilkins;",
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO wilkins;",
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO wilkins;"
    ]
    
    # Create a temporary SQL file for complex commands
    sql_file = Path("temp_setup.sql")
    with open(sql_file, "w") as f:
        f.write("\\c formfinder\n")
        for cmd in additional_commands[1:]:
            f.write(cmd + "\n")
    
    success, stdout, stderr = run_command(f'psql -U postgres -f "{sql_file}"')
    sql_file.unlink()  # Clean up temp file
    
    if success:
        print("✅ Database and user setup completed")
        return True
    else:
        print(f"❌ Failed to set additional privileges: {stderr}")
        return False

def test_formfinder_connection():
    """Test connection with FormFinder credentials"""
    print_step(6, "Testing FormFinder Database Connection")
    
    success, stdout, stderr = run_command(
        'psql -h localhost -U wilkins -d formfinder -c "SELECT version();"',
        capture_output=True
    )
    
    if success:
        print("✅ Successfully connected to FormFinder database")
        return True
    else:
        print("❌ Failed to connect to FormFinder database")
        print(f"Error: {stderr}")
        return False

def test_python_connection():
    """Test Python connection to PostgreSQL"""
    print_step(7, "Testing Python Database Connection")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            database="formfinder",
            user="wilkins",
            password="Holmes&7watson",
            port=5432
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print("✅ Python connection successful")
        print(f"   Database version: {version[:50]}...")
        return True
        
    except ImportError:
        print("❌ psycopg2 not installed")
        print("   Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ Python connection failed: {e}")
        return False

def check_config_file():
    """Check and update config.yaml"""
    print_step(8, "Checking Configuration File")
    
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    # Read current config
    with open(config_file, "r") as f:
        content = f.read()
    
    # Check if PostgreSQL is configured
    if 'type: "postgresql"' in content and 'host: "localhost"' in content:
        print("✅ config.yaml is properly configured for PostgreSQL")
        return True
    else:
        print("⚠️  config.yaml needs to be updated for PostgreSQL")
        print("\nRequired settings:")
        print("  database:")
        print('    type: "postgresql"')
        print('    host: "localhost"')
        print('    username: "wilkins"')
        print('    password: "Holmes&7watson"')
        print('    database_name: "formfinder"')
        return False

def run_migration():
    """Run data migration if available"""
    print_step(9, "Running Data Migration")
    
    migration_script = Path("migrate_csv_to_db.py")
    
    if not migration_script.exists():
        print("⚠️  migrate_csv_to_db.py not found (skipping migration)")
        return True
    
    print("Running data migration...")
    success, stdout, stderr = run_command("python migrate_csv_to_db.py")
    
    if success:
        print("✅ Data migration completed")
        if stdout:
            print(f"   Output: {stdout[:200]}...")
        return True
    else:
        print(f"❌ Data migration failed: {stderr}")
        return False

def main():
    """Main setup function"""
    print_header("FormFinder Native PostgreSQL Setup")
    print("This script will help you set up PostgreSQL for FormFinder without Docker.")
    print("\nPrerequisites:")
    print("- PostgreSQL must be installed on your system")
    print("- You need administrative access to create databases")
    print("- Python psycopg2 package should be installed")
    
    # Step 1: Check if PostgreSQL is installed
    if not check_postgresql_installed():
        get_installation_instructions()
        print("\n❌ Please install PostgreSQL first, then run this script again.")
        return False
    
    # Step 2: Check if PostgreSQL service is running
    if not check_postgresql_service():
        print("\n❌ Please start PostgreSQL service first, then run this script again.")
        return False
    
    # Step 3: Test postgres connection
    if not test_postgres_connection():
        print("\n❌ Cannot connect to PostgreSQL. Please check your installation.")
        return False
    
    # Step 4: Create database and user
    if not create_database_and_user():
        print("\n❌ Failed to create database and user.")
        return False
    
    # Step 5: Test FormFinder connection
    if not test_formfinder_connection():
        print("\n❌ Cannot connect to FormFinder database.")
        return False
    
    # Step 6: Test Python connection
    if not test_python_connection():
        print("\n⚠️  Python connection test failed. Install psycopg2-binary if needed.")
    
    # Step 7: Check config file
    config_ok = check_config_file()
    
    # Step 8: Run migration
    migration_ok = run_migration()
    
    # Final summary
    print_header("Setup Summary")
    
    if config_ok and migration_ok:
        print("🎉 PostgreSQL setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the connection: python test_postgresql_connection.py")
        print("2. Run FormFinder: python main.py")
        
        print("\nDatabase Details:")
        print("  Host: localhost")
        print("  Database: formfinder")
        print("  Username: wilkins")
        print("  Password: Holmes&7watson")
        print("  Port: 5432")
        
        return True
    else:
        print("⚠️  Setup completed with warnings.")
        if not config_ok:
            print("- Please update config.yaml manually")
        if not migration_ok:
            print("- Data migration may need to be run manually")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)