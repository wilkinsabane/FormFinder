# FormFinder Database Reset Guide

This guide provides step-by-step instructions to completely reset your FormFinder database and start fresh with clean data.

## 🎯 Why Reset the Database?

Resetting your database can help:
- **Remove duplicate data** and inconsistencies
- **Start fresh** with clean, accurate data
- **Resolve data corruption** issues
- **Optimize performance** by removing accumulated cruft
- **Test with clean datasets**

## ⚠️ Before You Begin

**Important**: This process will permanently delete all existing data. Make sure you:
1. **Backup any important data** if needed
2. **Close any running applications** (dashboard, scripts)
3. **Ensure PostgreSQL is running**

## 🚀 Quick Reset Options

### Option 1: Clean Reset (Recommended)

This is the most thorough reset method that directly interacts with PostgreSQL:

```bash
# Interactive mode (with confirmation)
python clean_database_reset.py

# Automatic mode (skip confirmation)
python clean_database_reset.py --yes

# Reset without fetching new data
python clean_database_reset.py --yes --no-fetch
```

### Option 2: CLI Reset

Use the built-in CLI commands for a controlled reset:

```bash
# Reset database (drop and recreate tables)
python -m formfinder.cli db reset

# Check status after reset
python -m formfinder.cli db status

# Fetch fresh data
python main.py
```

### Option 3: Full Pipeline Reset

This option resets everything and immediately fetches new data:

```bash
# Interactive reset with data fetching
python reset_and_refresh_database.py

# Automatic reset with data fetching
python reset_and_refresh_database.py --force

# Reset only (skip data fetching)
python reset_and_refresh_database.py --force --skip-fetch
```

## 🔧 Manual Reset Steps

If you prefer manual control, follow these steps:

### Step 1: Check Current Status
```bash
python -m formfinder.cli db status
```

### Step 2: Reset Database
```bash
python -c "
from formfinder.database import DatabaseManager
db = DatabaseManager()
db.drop_tables()
db.create_tables()
print('Database reset complete!')
"
```

### Step 3: Fetch Fresh Data
```bash
# Fetch all leagues and data
python main.py

# Or fetch specific leagues
python main.py --leagues 2021,2014,2002
```

### Step 4: Migrate CSV Data (Optional)
```bash
python DB\ Migration\ Files/migrate_csv_to_db.py
```

## 📊 Verification Steps

After reset, verify everything is working:

1. **Check database status**:
   ```bash
   python -m formfinder.cli db status
   ```

2. **Test database connection**:
   ```bash
   python test_postgresql_connection.py
   ```

3. **Run the dashboard**:
   ```bash
   python dashboard.py
   ```

4. **Check data freshness**:
   ```bash
   python -m formfinder.cli db status
   ```

## 🛠️ Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check if PostgreSQL is running
docker ps  # If using Docker
# or
pg_isready -h localhost -U wilkins -d formfinder

# Start PostgreSQL
docker compose -f docker-compose.postgres.yml up -d
```

### Permission Issues
```bash
# Ensure PostgreSQL user has proper permissions
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE formfinder TO wilkins;"
```

### Data Fetching Issues
```bash
# Check API token configuration
cat config.yaml | grep token

# Test API connection
python test_api.py
```

## 🔄 Regular Maintenance

To keep your database clean:

1. **Regular resets** (monthly recommended)
2. **Monitor for duplicates** using dashboard logs
3. **Clean up old data** periodically
4. **Use fresh API tokens** to avoid rate limits

## 📋 Quick Commands Reference

| Task | Command |
|------|---------|
| **Clean Reset** | `python clean_database_reset.py --yes` |
| **CLI Reset** | `python -m formfinder.cli db reset` |
| **Status Check** | `python -m formfinder.cli db status` |
| **Fetch Data** | `python main.py` |
| **Test Connection** | `python test_postgresql_connection.py` |
| **Run Dashboard** | `python dashboard.py` |

## 🆘 Support

If you encounter issues:

1. **Check logs**: Look in `data/logs/` directory
2. **Verify configuration**: Ensure `config.yaml` is correct
3. **Test database**: Use `test_postgresql_connection.py`
4. **Check PostgreSQL**: Ensure service is running
5. **Review API limits**: Check your API provider's rate limits

## ✅ Success Indicators

After a successful reset, you should see:
- ✅ Database status shows 0 records across all tables
- ✅ Dashboard loads without errors
- ✅ Fresh data appears in the dashboard
- ✅ No duplicate entries in team/league lists
- ✅ Recent match data is current and accurate