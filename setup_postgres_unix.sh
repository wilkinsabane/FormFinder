#!/bin/bash

# FormFinder PostgreSQL Setup Script for macOS and Linux
# This script helps set up PostgreSQL database for FormFinder

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_step() {
    echo -e "\n${BLUE}[Step $1]${NC} $2"
    echo "--------------------------------------------------"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "ubuntu"
        elif command -v yum &> /dev/null || command -v dnf &> /dev/null; then
            echo "redhat"
        else
            echo "linux"
        fi
    else
        echo "unknown"
    fi
}

# Check if PostgreSQL is installed
check_postgresql() {
    print_step 1 "Checking PostgreSQL Installation"
    
    if command -v psql &> /dev/null; then
        version=$(psql --version)
        print_success "PostgreSQL found: $version"
        return 0
    else
        print_error "PostgreSQL not found in PATH"
        return 1
    fi
}

# Show installation instructions
show_installation_instructions() {
    local os=$(detect_os)
    
    print_step 2 "PostgreSQL Installation Instructions"
    
    case $os in
        "macos")
            echo "macOS Installation Options:"
            echo ""
            echo "1. Homebrew (Recommended):"
            echo "   brew install postgresql@15"
            echo "   brew services start postgresql@15"
            echo ""
            echo "2. Postgres.app:"
            echo "   - Download from: https://postgresapp.com/"
            echo "   - Drag to Applications and launch"
            ;;
        "ubuntu")
            echo "Ubuntu/Debian Installation:"
            echo ""
            echo "   sudo apt update"
            echo "   sudo apt install postgresql postgresql-contrib"
            echo "   sudo systemctl start postgresql"
            echo "   sudo systemctl enable postgresql"
            ;;
        "redhat")
            echo "CentOS/RHEL/Fedora Installation:"
            echo ""
            echo "   sudo dnf install postgresql postgresql-server postgresql-contrib"
            echo "   sudo postgresql-setup --initdb"
            echo "   sudo systemctl start postgresql"
            echo "   sudo systemctl enable postgresql"
            ;;
        *)
            echo "Please install PostgreSQL for your operating system:"
            echo "https://www.postgresql.org/download/"
            ;;
    esac
}

# Check if PostgreSQL service is running
check_postgresql_service() {
    print_step 3 "Checking PostgreSQL Service Status"
    
    local os=$(detect_os)
    
    case $os in
        "macos")
            if brew services list | grep postgresql | grep -q "started"; then
                print_success "PostgreSQL service is running"
                return 0
            else
                print_error "PostgreSQL service is not running"
                echo "To start PostgreSQL on macOS:"
                echo "   brew services start postgresql@15"
                return 1
            fi
            ;;
        "ubuntu"|"redhat"|"linux")
            if systemctl is-active --quiet postgresql; then
                print_success "PostgreSQL service is running"
                return 0
            else
                print_error "PostgreSQL service is not running"
                echo "To start PostgreSQL on Linux:"
                echo "   sudo systemctl start postgresql"
                return 1
            fi
            ;;
        *)
            print_warning "Cannot check service status on this OS"
            return 0
            ;;
    esac
}

# Test PostgreSQL connection
test_postgres_connection() {
    print_step 4 "Testing PostgreSQL Connection"
    
    local os=$(detect_os)
    
    # Try different connection methods based on OS
    if [[ "$os" == "macos" ]]; then
        # macOS with Homebrew usually allows direct connection
        if psql postgres -c "SELECT version();" &> /dev/null; then
            print_success "Successfully connected to PostgreSQL"
            return 0
        fi
    fi
    
    # Try connecting as postgres user (Linux)
    if sudo -u postgres psql -c "SELECT version();" &> /dev/null 2>&1; then
        print_success "Successfully connected to PostgreSQL"
        return 0
    fi
    
    # Try direct connection as current user
    if psql -U postgres -c "SELECT version();" &> /dev/null 2>&1; then
        print_success "Successfully connected to PostgreSQL"
        return 0
    fi
    
    print_error "Failed to connect to PostgreSQL"
    echo "Troubleshooting:"
    echo "1. Make sure PostgreSQL is running"
    echo "2. Check if 'postgres' user exists"
    echo "3. Verify authentication settings in pg_hba.conf"
    return 1
}

# Create database and user
create_database_and_user() {
    print_step 5 "Creating Database and User"
    
    local os=$(detect_os)
    local psql_cmd
    
    # Determine the correct psql command based on OS
    if [[ "$os" == "macos" ]]; then
        psql_cmd="psql postgres"
    else
        psql_cmd="sudo -u postgres psql"
    fi
    
    echo "Creating user 'wilkins' and database 'formfinder'..."
    
    # Create user
    echo "1. Creating user 'wilkins'..."
    if $psql_cmd -c "CREATE USER wilkins WITH PASSWORD 'Holmes&7watson';" 2>/dev/null; then
        print_success "User 'wilkins' created successfully"
    else
        print_warning "User 'wilkins' already exists (skipping)"
    fi
    
    # Create database
    echo "2. Creating database 'formfinder'..."
    if $psql_cmd -c "CREATE DATABASE formfinder OWNER wilkins;" 2>/dev/null; then
        print_success "Database 'formfinder' created successfully"
    else
        print_warning "Database 'formfinder' already exists (skipping)"
    fi
    
    # Grant privileges
    echo "3. Granting privileges..."
    $psql_cmd -c "GRANT ALL PRIVILEGES ON DATABASE formfinder TO wilkins;" &> /dev/null
    
    # Create temporary SQL file for additional privileges
    cat > temp_privileges.sql << EOF
\\c formfinder
GRANT ALL ON SCHEMA public TO wilkins;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO wilkins;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO wilkins;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO wilkins;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO wilkins;
EOF
    
    $psql_cmd -f temp_privileges.sql &> /dev/null
    rm -f temp_privileges.sql
    
    print_success "Database and user setup completed"
}

# Test FormFinder database connection
test_formfinder_connection() {
    print_step 6 "Testing FormFinder Database Connection"
    
    if psql -h localhost -U wilkins -d formfinder -c "SELECT version();" &> /dev/null; then
        print_success "Successfully connected to FormFinder database"
        return 0
    else
        print_error "Failed to connect to FormFinder database"
        return 1
    fi
}

# Test Python connection
test_python_connection() {
    print_step 7 "Testing Python Database Connection"
    
    # Check if psycopg2 is installed
    if ! python3 -c "import psycopg2" &> /dev/null; then
        print_warning "psycopg2 not installed"
        echo "Installing psycopg2-binary..."
        if pip3 install psycopg2-binary; then
            print_success "psycopg2-binary installed successfully"
        else
            print_error "Failed to install psycopg2-binary"
            return 1
        fi
    fi
    
    # Test Python connection
    if python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        database='formfinder',
        user='wilkins',
        password='Holmes&7watson',
        port=5432
    )
    print('Python connection successful')
    conn.close()
except Exception as e:
    print(f'Connection failed: {e}')
    exit(1)
" &> /dev/null; then
        print_success "Python connection test successful"
        return 0
    else
        print_error "Python connection test failed"
        return 1
    fi
}

# Check config file
check_config_file() {
    print_step 8 "Checking Configuration File"
    
    if [[ ! -f "config.yaml" ]]; then
        print_error "config.yaml not found"
        return 1
    fi
    
    if grep -q 'type: "postgresql"' config.yaml && grep -q 'host: "localhost"' config.yaml; then
        print_success "config.yaml is properly configured for PostgreSQL"
        return 0
    else
        print_warning "config.yaml needs to be updated for PostgreSQL"
        echo ""
        echo "Required settings:"
        echo "  database:"
        echo '    type: "postgresql"'
        echo '    host: "localhost"'
        echo '    username: "wilkins"'
        echo '    password: "Holmes&7watson"'
        echo '    database_name: "formfinder"'
        return 1
    fi
}

# Run migration
run_migration() {
    print_step 9 "Running Data Migration"
    
    if [[ ! -f "migrate_csv_to_db.py" ]]; then
        print_warning "migrate_csv_to_db.py not found (skipping migration)"
        return 0
    fi
    
    echo "Running data migration..."
    if python3 migrate_csv_to_db.py; then
        print_success "Data migration completed"
        return 0
    else
        print_error "Data migration failed"
        return 1
    fi
}

# Main function
main() {
    print_header "FormFinder Native PostgreSQL Setup"
    echo "This script will help you set up PostgreSQL for FormFinder without Docker."
    echo ""
    echo "Prerequisites:"
    echo "- PostgreSQL must be installed on your system"
    echo "- You need administrative access to create databases"
    echo "- Python psycopg2 package should be installed"
    
    local os=$(detect_os)
    echo "Detected OS: $os"
    
    # Step 1: Check if PostgreSQL is installed
    if ! check_postgresql; then
        show_installation_instructions
        print_error "Please install PostgreSQL first, then run this script again."
        exit 1
    fi
    
    # Step 2: Check if PostgreSQL service is running
    if ! check_postgresql_service; then
        print_error "Please start PostgreSQL service first, then run this script again."
        exit 1
    fi
    
    # Step 3: Test postgres connection
    if ! test_postgres_connection; then
        print_error "Cannot connect to PostgreSQL. Please check your installation."
        exit 1
    fi
    
    # Step 4: Create database and user
    if ! create_database_and_user; then
        print_error "Failed to create database and user."
        exit 1
    fi
    
    # Step 5: Test FormFinder connection
    if ! test_formfinder_connection; then
        print_error "Cannot connect to FormFinder database."
        exit 1
    fi
    
    # Step 6: Test Python connection
    if ! test_python_connection; then
        print_warning "Python connection test failed. Install psycopg2-binary if needed."
    fi
    
    # Step 7: Check config file
    config_ok=true
    if ! check_config_file; then
        config_ok=false
    fi
    
    # Step 8: Run migration
    migration_ok=true
    if ! run_migration; then
        migration_ok=false
    fi
    
    # Final summary
    print_header "Setup Summary"
    
    if $config_ok && $migration_ok; then
        print_success "PostgreSQL setup completed successfully!"
        echo ""
        echo "Next steps:"
        echo "1. Test the connection: python3 test_postgresql_connection.py"
        echo "2. Run FormFinder: python3 main.py"
        echo ""
        echo "Database Details:"
        echo "  Host: localhost"
        echo "  Database: formfinder"
        echo "  Username: wilkins"
        echo "  Password: Holmes&7watson"
        echo "  Port: 5432"
        echo ""
        echo "Useful Commands:"
        echo "- Connect to database: psql -h localhost -U wilkins -d formfinder"
        echo "- Check service status: systemctl status postgresql (Linux) or brew services list (macOS)"
        echo "- View logs: sudo journalctl -u postgresql (Linux) or brew services list (macOS)"
    else
        print_warning "Setup completed with warnings."
        if ! $config_ok; then
            echo "- Please update config.yaml manually"
        fi
        if ! $migration_ok; then
            echo "- Data migration may need to be run manually"
        fi
    fi
}

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi