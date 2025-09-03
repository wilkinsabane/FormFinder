# Database Table Data Clearer

A modern, robust, and user-friendly Python script for clearing data from specific database tables without dropping the table structure. This tool provides interactive table selection with detailed summaries, safety confirmations, and comprehensive logging.

## Features

### 🎯 Core Functionality
- **Interactive Table Selection**: Browse and select tables with detailed information
- **Data Preservation**: Clears table data while maintaining table structure and indexes
- **Safety First**: Multiple confirmation prompts and dependency checks
- **Comprehensive Logging**: Detailed logs for audit trails and debugging
- **Backup Support**: Automatic backup creation before clearing data
- **Cross-Database**: Works with both SQLite and PostgreSQL

### 📊 Table Information Display
- Row counts with comma formatting
- Storage size estimates (B, KB, MB, GB)
- Column details with types and constraints
- Index information
- Foreign key relationships
- Dependency analysis

### 🔧 Advanced Features
- **Dry-run Mode**: Preview what would be deleted without making changes
- **Command-line Interface**: Non-interactive mode for automation
- **Dependency Detection**: Warns about foreign key relationships
- **Sequence Reset**: Automatically resets auto-increment sequences
- **Progress Indicators**: Real-time progress during operations

## Installation

No additional installation required - the script uses existing project dependencies. Ensure you have the required packages:

```bash
pip install rich sqlalchemy
```

## Usage

### Interactive Mode (Recommended)
```bash
python clear_table_data.py
```

This launches an interactive interface where you can:
1. View all tables with summaries
2. Select a table to clear
3. Review detailed information
4. Confirm the operation

### Command-Line Options

#### Clear Specific Table
```bash
python clear_table_data.py --table fixtures
```

#### Dry Run (Preview Only)
```bash
python clear_table_data.py --table fixtures --dry-run
```

#### List All Tables
```bash
python clear_table_data.py --list
```

#### Skip Confirmations (Use with Caution)
```bash
python clear_table_data.py --table fixtures --confirm
```

#### Verbose Logging
```bash
python clear_table_data.py --table fixtures --verbose
```

## Examples

### Example 1: Interactive Selection
```bash
$ python clear_table_data.py

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                 Database Table Data Clearer               ┃
┃                                                           ┃
┃ Select a table to clear all data from. The table          ┃
┃ structure will be preserved.                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Available Tables            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ # │ Table Name    │ Rows    │ Size  │
├───┼───────────────┼─────────┼───────┤
│ 1 │ fixtures      │ 15,234  │ 2.3 MB│
│ 2 │ leagues       │ 245     │ 45 KB │
│ 3 │ predictions   │ 8,921   │ 1.1 MB│
│ 4 │ teams         │ 512     │ 89 KB │
└───┴───────────────┴─────────┴───────┘

Enter table number (or 'q' to quit): 1
```

### Example 2: Dry Run Mode
```bash
$ python clear_table_data.py --table predictions --dry-run

Table: predictions
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                  Summary for predictions                      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Property        │ Value                                      │
│─────────────────┼────────────────────────────────────────────│
│ Row Count       │ 8,921                                      │
│ Size Estimate   │ 1.1 MB                                     │
│ Columns         │ 23                                         │
│ Indexes         │ 4                                          │
│ Foreign Keys    │ 1                                          │
└─────────────────┴────────────────────────────────────────────┘

✓ Dry run complete: Would delete 8,921 rows
```

### Example 3: List All Tables
```bash
$ python clear_table_data.py --list

Database Tables Summary

Table: fixtures
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    Summary for fixtures                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Property        │ Value                                      │
│─────────────────┼────────────────────────────────────────────│
│ Row Count       │ 15,234                                     │
│ Size Estimate   │ 2.3 MB                                     │
│ Columns         │ 23                                         │
│ Indexes         │ 5                                          │
│ Foreign Keys    │ 2                                          │
└─────────────────┴────────────────────────────────────────────┘

Columns:
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Name               ┃ Type                             ┃ Nullable ┃ Primary Key ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ id                 ┃ INTEGER                          │ ✗        │ ✓           │
│ league_id          ┃ INTEGER                          │ ✗        │ ✗           │
│ home_team_id       ┃ INTEGER                          │ ✗        │ ✗           │
│ away_team_id       ┃ INTEGER                          │ ✗        │ ✗           │
│ match_date         ┃ DATETIME                         │ ✗        │ ✗           │
└────────────────────┴──────────────────────────────────┴──────────┴─────────────┘
```

## Safety Features

### 🔍 Dependency Checking
The script automatically detects foreign key relationships and warns about dependent tables:

```
⚠️  WARNING: Dependencies found!
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Dependent Tables           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Dependent Table │ Foreign Key       │
│─────────────────┼───────────────────│
│ predictions     │ fk_fixture_id     │
└─────────────────┴───────────────────┘
```

### 💾 Automatic Backups
Before clearing any data, the script creates JSON backups:
- Location: `data/backups/`
- Naming: `{table_name}_backup_{timestamp}.json`
- Format: JSON array with all table rows

### ✅ Confirmation Prompts
Multiple confirmation steps prevent accidental data loss:
1. Table selection confirmation
2. Dependency warning (if applicable)
3. Final deletion confirmation

## Database Support

### SQLite
- Uses `PRAGMA foreign_keys=OFF/ON` for safe deletion
- Creates JSON backups
- Resets sequences automatically

### PostgreSQL
- Uses transaction-safe deletion
- Handles sequences with `ALTER SEQUENCE`
- Supports materialized views

## Configuration

The script uses your existing `config.yaml` database configuration. No additional setup required.

## Log Files

All operations are logged to:
- Location: `data/logs/`
- Format: `table_clearer_YYYYMMDD_HHMMSS.log`
- Contains: Timestamps, table names, row counts, errors, and performance metrics

## Error Handling

### Common Issues and Solutions

#### "Table not found"
- Ensure the database is properly initialized
- Check if the table name is spelled correctly
- Verify database connection in `config.yaml`

#### "Foreign key constraint violation"
- Review dependency warnings before proceeding
- Consider clearing dependent tables first
- Use `--dry-run` to identify issues

#### "Permission denied"
- Check file permissions for backup directory
- Ensure write access to `data/` folder
- Verify database user permissions

## Performance

### Large Tables
- Progress indicators for tables > 1000 rows
- Estimated completion times
- Memory-efficient batch processing

### Optimization Tips
- Use `--dry-run` for large tables first
- Clear during low-traffic periods
- Monitor disk space for backups

## Testing

### Test Mode
Use the testing database configuration:

```bash
# Set test environment
export TESTING=true
python clear_table_data.py --table test_table --dry-run
```

### Verification
After clearing data:
1. Check table structure is intact: `\d tablename` (PostgreSQL) or `.schema tablename` (SQLite)
2. Verify indexes exist: Check with database client
3. Confirm sequences reset: Check auto-increment values

## Troubleshooting

### Debug Mode
Enable verbose logging:
```bash
python clear_table_data.py --table fixtures --verbose
```

### Common Commands
```bash
# Check available tables
python clear_table_data.py --list

# Test with dry run
python clear_table_data.py --table predictions --dry-run

# Force clear without prompts (dangerous)
python clear_table_data.py --table fixtures --confirm
```

## Support

For issues or questions:
1. Check the log files in `data/logs/`
2. Use `--dry-run` to test operations
3. Review the database configuration in `config.yaml`
4. Create backups before major operations

## License

This script is part of the FormFinder project and follows the same license terms.