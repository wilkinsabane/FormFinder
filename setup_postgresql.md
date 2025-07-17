# PostgreSQL Setup Guide for FormFinder

## Overview

This guide will help you set up PostgreSQL for FormFinder, including database creation, user setup, and proper configuration.

## Prerequisites

1. **Install PostgreSQL**
   - Download from: https://www.postgresql.org/download/
   - Or use Docker (recommended for development)

## Option 1: Docker Setup (Recommended)

### 1. Create Docker Compose for PostgreSQL

```yaml
# docker-compose.postgres.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    container_name: formfinder_postgres
    environment:
      POSTGRES_DB: formfinder
      POSTGRES_USER: wilkins
      POSTGRES_PASSWORD: Holmes&7watson
      POSTGRES_HOST_AUTH_METHOD: md5
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U wilkins -d formfinder"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
```

### 2. Start PostgreSQL with Docker

```bash
# Start PostgreSQL
docker-compose -f docker-compose.postgres.yml up -d

# Check if it's running
docker-compose -f docker-compose.postgres.yml ps

# View logs
docker-compose -f docker-compose.postgres.yml logs postgres
```

### 3. Update config.yaml for Docker

```yaml
database:
  type: "postgresql"
  postgresql:
    host: "localhost"  # Changed from "your_postgres_host"
    port: 5432
    database: "formfinder"
    username: "wilkins"
    password: "Holmes&7watson"
```

## Option 2: Local PostgreSQL Installation

### 1. Install PostgreSQL

**Windows:**
```bash
# Download installer from postgresql.org
# Or use chocolatey
choco install postgresql
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Create Database and User

```bash
# Connect as postgres superuser
sudo -u postgres psql

# Or on Windows (run as Administrator)
psql -U postgres
```

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

-- Exit
\q
```

### 3. Update config.yaml for Local Installation

```yaml
database:
  type: "postgresql"
  postgresql:
    host: "localhost"  # or your actual hostname/IP
    port: 5432
    database: "formfinder"
    username: "wilkins"
    password: "Holmes&7watson"
```

## Option 3: Cloud PostgreSQL (Production)

### Popular Cloud Providers:

1. **AWS RDS**
2. **Google Cloud SQL**
3. **Azure Database for PostgreSQL**
4. **DigitalOcean Managed Databases**
5. **Heroku Postgres**

### Example for AWS RDS:

```yaml
database:
  type: "postgresql"
  postgresql:
    host: "your-rds-endpoint.amazonaws.com"
    port: 5432
    database: "formfinder"
    username: "wilkins"
    password: "Holmes&7watson"
```

## Database Initialization

### 1. Create Tables

The FormFinder application will automatically create tables when you run it for the first time. The database schema is defined in `formfinder/database.py`.

### 2. Migrate Existing Data

After setting up PostgreSQL, run the migration script:

```bash
python migrate_csv_to_db.py
```

## Testing the Connection

### 1. Test Basic Connection

```bash
python test_postgresql_connection.py
```

### 2. Manual Connection Test

```bash
# Using psql
psql -h localhost -U wilkins -d formfinder

# Using Python
python -c "import psycopg2; conn = psycopg2.connect(host='localhost', database='formfinder', user='wilkins', password='Holmes&7watson'); print('Connection successful!')"
```

## Troubleshooting

### Common Issues:

1. **"could not translate host name"**
   - Solution: Change `your_postgres_host` to `localhost` or actual hostname

2. **"password authentication failed"**
   - Solution: Verify username/password in PostgreSQL

3. **"database does not exist"**
   - Solution: Create the database using the SQL commands above

4. **"connection refused"**
   - Solution: Ensure PostgreSQL is running and accepting connections

5. **"permission denied"**
   - Solution: Grant proper privileges to the user

### Check PostgreSQL Status:

```bash
# Linux/macOS
sudo systemctl status postgresql

# Windows (as Administrator)
sc query postgresql-x64-15

# Docker
docker-compose -f docker-compose.postgres.yml ps
```

### View PostgreSQL Logs:

```bash
# Linux
sudo tail -f /var/log/postgresql/postgresql-*.log

# Docker
docker-compose -f docker-compose.postgres.yml logs -f postgres
```

## Security Best Practices

1. **Use Environment Variables for Credentials**

```yaml
# config.yaml
database:
  postgresql:
    host: ${POSTGRES_HOST:-localhost}
    username: ${POSTGRES_USER:-wilkins}
    password: ${POSTGRES_PASSWORD}
```

2. **Create .env file**

```bash
# .env
POSTGRES_HOST=localhost
POSTGRES_USER=wilkins
POSTGRES_PASSWORD=Holmes&7watson
```

3. **Restrict Network Access**
   - Configure `pg_hba.conf` for specific IP ranges
   - Use SSL connections in production

4. **Regular Backups**

```bash
# Create backup
pg_dump -h localhost -U wilkins formfinder > formfinder_backup.sql

# Restore backup
psql -h localhost -U wilkins formfinder < formfinder_backup.sql
```

## Performance Optimization

1. **Create Indexes**

```sql
-- Add indexes for better query performance
CREATE INDEX idx_standings_league_season ON standings(league_id, season);
CREATE INDEX idx_fixtures_date ON fixtures(match_date);
CREATE INDEX idx_teams_league ON teams(league_id);
```

2. **Connection Pooling**

Consider using PgBouncer for connection pooling in production.

## Next Steps

1. Choose your preferred setup option (Docker recommended for development)
2. Update `config.yaml` with correct hostname
3. Run `python test_postgresql_connection.py` to verify
4. Run `python migrate_csv_to_db.py` to migrate data
5. Run `python main.py` to start using FormFinder with PostgreSQL

## Quick Start Commands

```bash
# Option 1: Docker (Recommended)
docker-compose -f docker-compose.postgres.yml up -d

# Option 2: Local PostgreSQL
# (Install PostgreSQL first, then create database/user as shown above)

# Update config and test
python test_postgresql_connection.py
python migrate_csv_to_db.py
python main.py
```