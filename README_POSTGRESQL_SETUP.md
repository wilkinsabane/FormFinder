# PostgreSQL Setup Guide for FormFinder

This guide provides multiple options for setting up PostgreSQL for the FormFinder application without using Docker.

## Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**
```cmd
# Run the automated setup script
setup_postgres_windows.bat
```

**macOS/Linux:**
```bash
# Make script executable and run
chmod +x setup_postgres_unix.sh
./setup_postgres_unix.sh
```

**Cross-platform Python script:**
```bash
python setup_database_native.py
```

### Option 2: Manual Setup

Follow the detailed instructions in `setup_postgresql_native.md`

## Files Overview

| File | Purpose | Platform |
|------|---------|----------|
| `setup_postgresql_native.md` | Comprehensive manual setup guide | All |
| `setup_database_native.py` | Automated Python setup script | All |
| `setup_postgres_windows.bat` | Windows batch script | Windows |
| `setup_postgres_unix.sh` | Shell script for Unix systems | macOS/Linux |
| `test_postgresql_connection.py` | Connection testing script | All |

## Prerequisites

1. **PostgreSQL Installation**: PostgreSQL must be installed on your system
2. **Python Dependencies**: `psycopg2-binary` package
3. **Administrative Access**: Required to create databases and users
4. **Network Access**: PostgreSQL service must be running and accessible

## Installation by Operating System

### Windows

#### Method 1: Official Installer (Recommended)
1. Download from: https://www.postgresql.org/download/windows/
2. Run installer as Administrator
3. Set password for `postgres` user (remember this!)
4. Use default port (5432)
5. Add PostgreSQL bin directory to PATH

#### Method 2: Package Managers
```powershell
# Chocolatey
choco install postgresql

# Scoop
scoop install postgresql
```

### macOS

#### Method 1: Homebrew (Recommended)
```bash
brew install postgresql@15
brew services start postgresql@15

# Add to PATH
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Method 2: Postgres.app
1. Download from: https://postgresapp.com/
2. Drag to Applications folder
3. Launch and initialize

### Linux

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### CentOS/RHEL/Fedora
```bash
sudo dnf install postgresql postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

## Database Setup

### 1. Connect to PostgreSQL

**Windows:**
```cmd
psql -U postgres
```

**macOS/Linux:**
```bash
# Method 1: Switch to postgres user
sudo -u postgres psql

# Method 2: Direct connection
psql postgres
```

### 2. Create Database and User

```sql
-- Create user
CREATE USER wilkins WITH PASSWORD 'Holmes&7watson';

-- Create database
CREATE DATABASE formfinder OWNER wilkins;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE formfinder TO wilkins;

-- Connect to formfinder database
\c formfinder

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO wilkins;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO wilkins;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO wilkins;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO wilkins;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO wilkins;

-- Exit
\q
```

## Configuration

### Update config.yaml

Ensure your `config.yaml` has the following database configuration:

```yaml
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  username: "wilkins"
  password: "Holmes&7watson"
  database_name: "formfinder"
```

### Install Python Dependencies

```bash
pip install psycopg2-binary
```

## Testing

### 1. Test PostgreSQL Connection

```bash
# Test with psql
psql -h localhost -U wilkins -d formfinder -c "SELECT version();"
```

### 2. Test Python Connection

```bash
# Run the test script
python test_postgresql_connection.py
```

### 3. Test FormFinder Integration

```bash
# Run data migration
python migrate_csv_to_db.py

# Run FormFinder
python main.py
```

## Troubleshooting

### Common Issues

#### "psql: command not found"
**Solution:** Add PostgreSQL bin directory to PATH
- Windows: `C:\Program Files\PostgreSQL\15\bin`
- macOS: `/opt/homebrew/opt/postgresql@15/bin`
- Linux: Usually `/usr/bin` (already in PATH)

#### "peer authentication failed"
**Solution:** Edit `pg_hba.conf` and change `peer` to `md5`
```bash
# Find config file
psql -U postgres -c "SHOW hba_file;"

# Edit the file and change:
# local   all   all   peer
# to:
# local   all   all   md5

# Restart PostgreSQL
sudo systemctl restart postgresql  # Linux
brew services restart postgresql@15  # macOS
net restart postgresql-x64-15  # Windows
```

#### "password authentication failed"
**Solutions:**
1. Verify username and password
2. Check if user exists: `\du` in psql
3. Reset password: `ALTER USER wilkins PASSWORD 'Holmes&7watson';`

#### "connection refused"
**Solutions:**
1. Check if PostgreSQL is running
2. Verify port (default 5432)
3. Check firewall settings
4. Verify host configuration

#### "database does not exist"
**Solution:** Create the database
```sql
CREATE DATABASE formfinder OWNER wilkins;
```

### Service Management

#### Windows
```cmd
# Check status
sc query "postgresql-x64-15"

# Start service
net start "postgresql-x64-15"

# Stop service
net stop "postgresql-x64-15"
```

#### macOS
```bash
# Check status
brew services list | grep postgresql

# Start service
brew services start postgresql@15

# Stop service
brew services stop postgresql@15

# Restart service
brew services restart postgresql@15
```

#### Linux
```bash
# Check status
sudo systemctl status postgresql

# Start service
sudo systemctl start postgresql

# Stop service
sudo systemctl stop postgresql

# Restart service
sudo systemctl restart postgresql

# Enable auto-start
sudo systemctl enable postgresql
```

### Log Files

#### Windows
- Location: `C:\Program Files\PostgreSQL\15\data\log\`
- View: Use text editor or `type` command

#### macOS
```bash
# View logs
tail -f /opt/homebrew/var/log/postgresql@15.log

# Or check with brew
brew services list
```

#### Linux
```bash
# View logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Or use journalctl
sudo journalctl -u postgresql -f
```

## Security Best Practices

### 1. Change Default Passwords
```sql
ALTER USER postgres PASSWORD 'new_secure_password';
```

### 2. Restrict Network Access
Edit `pg_hba.conf` to limit connections:
```
# Only allow local connections
host    all    all    127.0.0.1/32    md5
host    all    all    ::1/128         md5
```

### 3. Enable SSL (Production)
Edit `postgresql.conf`:
```ini
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

### 4. Regular Backups
```bash
# Create backup
pg_dump -h localhost -U wilkins formfinder > formfinder_backup.sql

# Restore backup
psql -h localhost -U wilkins formfinder < formfinder_backup.sql
```

## Performance Optimization

### Basic Tuning
Edit `postgresql.conf`:
```ini
# Memory settings (adjust based on your system)
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connection settings
max_connections = 100

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

### Create Indexes
```sql
-- Performance indexes for FormFinder
CREATE INDEX IF NOT EXISTS idx_standings_league_season ON standings(league_id, season);
CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(match_date);
CREATE INDEX IF NOT EXISTS idx_teams_league ON teams(league_id);
CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league_id);
```

## Migration from SQLite

If you're migrating from SQLite:

1. **Backup existing data:**
   ```bash
   cp formfinder.db formfinder.db.backup
   ```

2. **Update configuration:**
   Change `config.yaml` from SQLite to PostgreSQL

3. **Run migration:**
   ```bash
   python migrate_csv_to_db.py
   ```

4. **Verify data:**
   ```bash
   python test_postgresql_connection.py
   ```

## Quick Command Reference

```bash
# Connect to database
psql -h localhost -U wilkins -d formfinder

# List databases
psql -U postgres -l

# List users
psql -U postgres -c "\du"

# Check PostgreSQL version
psql --version

# Create backup
pg_dump -h localhost -U wilkins formfinder > backup.sql

# Restore backup
psql -h localhost -U wilkins formfinder < backup.sql

# Monitor connections
psql -h localhost -U wilkins -d formfinder -c "SELECT * FROM pg_stat_activity;"
```

## Support

If you encounter issues:

1. **Check the logs** (see Log Files section)
2. **Verify service status** (see Service Management section)
3. **Test connections** step by step
4. **Review configuration files**
5. **Check firewall and network settings**

For FormFinder-specific issues, ensure:
- `config.yaml` is correctly configured
- Python dependencies are installed
- Database tables are created
- Data migration completed successfully

## Next Steps

After successful setup:

1. **Test the connection:** `python test_postgresql_connection.py`
2. **Migrate data:** `python migrate_csv_to_db.py`
3. **Run FormFinder:** `python main.py`
4. **Set up monitoring** (optional)
5. **Configure backups** (recommended)
6. **Optimize performance** (as needed)

This native PostgreSQL setup provides full control and is ideal for production environments or when Docker is not available.