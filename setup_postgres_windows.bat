@echo off
REM FormFinder PostgreSQL Setup Script for Windows
REM This script helps set up PostgreSQL database for FormFinder

echo ========================================
echo FormFinder PostgreSQL Setup (Windows)
echo ========================================
echo.

REM Check if PostgreSQL is installed
echo [Step 1] Checking PostgreSQL installation...
psql --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PostgreSQL not found in PATH
    echo.
    echo Please install PostgreSQL first:
    echo 1. Download from: https://www.postgresql.org/download/windows/
    echo 2. Run installer as Administrator
    echo 3. Remember the postgres user password
    echo 4. Add PostgreSQL bin directory to PATH
    echo.
    echo Common PostgreSQL bin location:
    echo "C:\Program Files\PostgreSQL\17\bin"
    echo.
    pause
    exit /b 1
)
echo PostgreSQL found!
echo.

REM Check if PostgreSQL service is running
echo [Step 2] Checking PostgreSQL service...
sc query "postgresql-x64-17" | find "RUNNING" >nul 2>&1
if %errorlevel% neq 0 (
    echo PostgreSQL service is not running
    echo Starting PostgreSQL service...
    net start "postgresql-x64-17"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to start PostgreSQL service
        echo Please start it manually:
        echo - Open Services.msc
        echo - Find "postgresql-x64-17" service
        echo - Right-click and select "Start"
        pause
        exit /b 1
    )
)
echo PostgreSQL service is running!
echo.

REM Test connection to PostgreSQL
echo [Step 3] Testing PostgreSQL connection...
psql -U postgres -c "SELECT version();" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Cannot connect to PostgreSQL
    echo Please check:
    echo 1. PostgreSQL service is running
    echo 2. postgres user password is correct
    echo 3. Authentication settings in pg_hba.conf
    pause
    exit /b 1
)
echo Connection successful!
echo.

REM Create database and user
echo [Step 4] Creating FormFinder database and user...
echo Creating user 'wilkins'...
psql -U postgres -c "CREATE USER wilkins WITH PASSWORD 'Holmes&7watson';" 2>nul
if %errorlevel% equ 0 (
    echo User 'wilkins' created successfully
) else (
    echo User 'wilkins' already exists (skipping)
)

echo Creating database 'formfinder'...
psql -U postgres -c "CREATE DATABASE formfinder OWNER wilkins;" 2>nul
if %errorlevel% equ 0 (
    echo Database 'formfinder' created successfully
) else (
    echo Database 'formfinder' already exists (skipping)
)

echo Granting privileges...
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE formfinder TO wilkins;" >nul 2>&1

REM Create temporary SQL file for additional privileges
echo \c formfinder > temp_privileges.sql
echo GRANT ALL ON SCHEMA public TO wilkins; >> temp_privileges.sql
echo GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO wilkins; >> temp_privileges.sql
echo GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO wilkins; >> temp_privileges.sql
echo ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO wilkins; >> temp_privileges.sql
echo ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO wilkins; >> temp_privileges.sql

psql -U postgres -f temp_privileges.sql >nul 2>&1
del temp_privileges.sql
echo.

REM Test FormFinder database connection
echo [Step 5] Testing FormFinder database connection...
psql -h localhost -U wilkins -d formfinder -c "SELECT version();" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Cannot connect to FormFinder database
    echo Please check the setup manually
    pause
    exit /b 1
)
echo FormFinder database connection successful!
echo.

REM Check Python psycopg2
echo [Step 6] Checking Python PostgreSQL driver...
python -c "import psycopg2; print('psycopg2 is installed')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: psycopg2 not installed
    echo Installing psycopg2-binary...
    pip install psycopg2-binary
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install psycopg2-binary
        echo Please install manually: pip install psycopg2-binary
    )
) else (
    echo psycopg2 is already installed
)
echo.

REM Test Python connection
echo [Step 7] Testing Python database connection...
python -c "import psycopg2; conn = psycopg2.connect(host='localhost', database='formfinder', user='wilkins', password='Holmes&7watson'); print('Python connection successful'); conn.close()" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Python connection test failed
    echo This might be due to missing psycopg2 or connection issues
) else (
    echo Python connection test successful!
)
echo.

REM Final summary
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Database Details:
echo   Host: localhost
echo   Database: formfinder
echo   Username: wilkins
echo   Password: Holmes&7watson
echo   Port: 5432
echo.
echo Next Steps:
echo 1. Update config.yaml if needed
echo 2. Test connection: python test_postgresql_connection.py
echo 3. Run migration: python migrate_csv_to_db.py
echo 4. Start FormFinder: python main.py
echo.
echo Useful Commands:
echo - Connect to database: psql -h localhost -U wilkins -d formfinder
echo - Check service status: sc query "postgresql-x64-15"
echo - Start service: net start "postgresql-x64-15"
echo - Stop service: net stop "postgresql-x64-15"
echo.
pause