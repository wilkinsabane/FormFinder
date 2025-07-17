# Native PostgreSQL Setup Guide (No Docker)

This guide will help you install and configure PostgreSQL directly on your system without using Docker.

## Installation by Operating System

### Windows

#### Option 1: Official Installer (Recommended)
1. Download PostgreSQL from: https://www.postgresql.org/download/windows/
2. Run the installer as Administrator
3. During installation:
   - Choose installation directory (default: `C:\Program Files\PostgreSQL\15`)
   - Set password for `postgres` superuser (remember this!)
   - Choose port (default: 5432)
   - Select locale (default is fine)

#### Option 2: Chocolatey
```powershell
# Install Chocolatey first if not installed
# Then install PostgreSQL
choco install postgresql
```

#### Option 3: Scoop
```powershell
scoop install postgresql
```

### macOS

#### Option 1: Homebrew (Recommended)
```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Option 2: Postgres.app
1. Download from: https://postgresapp.com/
2. Drag to Applications folder
3. Launch and initialize

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check status
sudo systemctl status postgresql
```

### Linux (CentOS/RHEL/Fedora)

```bash
# Install PostgreSQL
sudo dnf install postgresql postgresql-server postgresql-contrib
# or for older versions: sudo yum install postgresql postgresql-server postgresql-contrib

# Initialize database
sudo postgresql-setup --initdb

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

## Database and User Setup

### Step 1: Connect to PostgreSQL

#### Windows
```cmd
# Open Command Prompt as Administrator
# Navigate to PostgreSQL bin directory
cd "C:\Program Files\PostgreSQL\15\bin"

# Connect as postgres superuser
psql -U postgres
```

#### macOS/Linux
```bash
# Switch to postgres user (Linux)
sudo -u postgres psql

# Or connect directly (macOS with Homebrew)
psql postgres
```

### Step 2: Create Database and User

Once connected to PostgreSQL, run these SQL commands:

```sql
-- Create the user 'wilkins'
CREATE USER wilkins WITH PASSWORD 'Holmes&7watson';

-- Create the database 'formfinder'
CREATE DATABASE formfinder OWNER wilkins;

-- Grant all privileges on the database
GRANT ALL PRIVILEGES ON DATABASE formfinder TO wilkins;

-- Connect to the formfinder database
\c formfinder

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO wilkins;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO wilkins;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO wilkins;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO wilkins;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO wilkins;

-- Exit PostgreSQL
\q
```

### Step 3: Configure PostgreSQL (if needed)

#### Find Configuration Files

```sql
-- Connect to PostgreSQL and find config locations
psql -U postgres
SHOW config_file;
SHOW hba_file;
\q
```

#### Common Configuration Locations

**Windows:**
- `C:\Program Files\PostgreSQL\15\data\postgresql.conf`
- `C:\Program Files\PostgreSQL\15\data\pg_hba.conf`

**macOS (Homebrew):**
- `/opt/homebrew/var/postgresql@15/postgresql.conf`
- `/opt/homebrew/var/postgresql@15/pg_hba.conf`

**Linux:**
- `/etc/postgresql/15/main/postgresql.conf`
- `/etc/postgresql/15/main/pg_hba.conf`

#### Enable Local Connections (if needed)

Edit `pg_hba.conf` to ensure local connections are allowed:

```
# Add or modify these lines:
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
```

Restart PostgreSQL after changes:

**Windows:**
```cmd
# In Services.msc, restart "postgresql-x64-15" service
# Or use command line:
net stop postgresql-x64-15
net start postgresql-x64-15
```

**macOS:**
```bash
brew services restart postgresql@15
```

**Linux:**
```bash
sudo systemctl restart postgresql
```

## Verification

### Test Connection

```bash
# Test connection with new user
psql -h localhost -U wilkins -d formfinder

# If successful, you should see:
# formfinder=>

# Test a simple query
SELECT version();

# Exit
\q
```

### Test with Python

```python
# Test connection with Python
python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        database='formfinder',
        user='wilkins',
        password='Holmes&7watson'
    )
    print('✅ Connection successful!')
    conn.close()
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
```

## Troubleshooting

### Common Issues

1. **"psql: command not found"**
   - Add PostgreSQL bin directory to PATH
   - Windows: Add `C:\Program Files\PostgreSQL\15\bin` to PATH
   - macOS: `export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"`

2. **"peer authentication failed"**
   - Edit `pg_hba.conf` and change `peer` to `md5` for local connections
   - Restart PostgreSQL

3. **"password authentication failed"**
   - Verify username and password
   - Check if user exists: `\du` in psql

4. **"database does not exist"**
   - Create database: `CREATE DATABASE formfinder;`

5. **"connection refused"**
   - Check if PostgreSQL is running
   - Verify port (default 5432)
   - Check firewall settings

### Check PostgreSQL Status

**Windows:**
```cmd
# Check if service is running
sc query postgresql-x64-15

# Or in Services.msc, look for "postgresql-x64-15"
```

**macOS:**
```bash
brew services list | grep postgresql
```

**Linux:**
```bash
sudo systemctl status postgresql
```

### View PostgreSQL Logs

**Windows:**
- Check: `C:\Program Files\PostgreSQL\15\data\log\`

**macOS:**
```bash
tail -f /opt/homebrew/var/log/postgresql@15.log
```

**Linux:**
```bash
sudo tail -f /var/log/postgresql/postgresql-15-main.log
```

## Security Best Practices

1. **Change Default Passwords**
   ```sql
   ALTER USER postgres PASSWORD 'new_secure_password';
   ```

2. **Restrict Network Access**
   - Only allow connections from necessary IPs
   - Use SSL in production

3. **Regular Backups**
   ```bash
   # Create backup
   pg_dump -h localhost -U wilkins formfinder > formfinder_backup.sql
   
   # Restore backup
   psql -h localhost -U wilkins formfinder < formfinder_backup.sql
   ```

4. **Monitor Connections**
   ```sql
   -- View active connections
   SELECT * FROM pg_stat_activity WHERE datname = 'formfinder';
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

# Logging
log_statement = 'all'  # For development only
log_duration = on
```

### Create Indexes

```sql
-- Connect to formfinder database
\c formfinder

-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_standings_league_season ON standings(league_id, season);
CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(match_date);
CREATE INDEX IF NOT EXISTS idx_teams_league ON teams(league_id);
CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league_id);
```

## Next Steps

1. **Verify Installation**: Run the test commands above
2. **Update FormFinder Config**: Ensure `config.yaml` has correct settings
3. **Test Connection**: Run `python test_postgresql_connection.py`
4. **Migrate Data**: Run `python migrate_csv_to_db.py`
5. **Run FormFinder**: Execute `python main.py`

## Quick Command Reference

```bash
# Start PostgreSQL
# Windows: net start postgresql-x64-15
# macOS: brew services start postgresql@15
# Linux: sudo systemctl start postgresql

# Connect to database
psql -h localhost -U wilkins -d formfinder

# Create backup
pg_dump -h localhost -U wilkins formfinder > backup.sql

# Restore backup
psql -h localhost -U wilkins formfinder < backup.sql

# Check PostgreSQL version
psql --version

# View databases
psql -U postgres -l
```

This native installation gives you full control over PostgreSQL and is ideal for production environments or when Docker is not available.