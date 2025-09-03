# Duplicate Handling in Historical Data Fetcher

## Overview

The `historical_fetcher.py` script implements robust duplicate detection and prevention mechanisms to ensure data integrity when fetching and storing historical match data.

## Duplicate Detection Mechanisms

### 1. League-Level Duplicate Prevention

**Location**: Lines 157-175 in `historical_fetcher.py`

```python
# Query leagues that have fixtures in the database for the current season
existing_data = session.query(League.id, func.count(Fixture.id).label('match_count')).\
    join(Fixture, League.id == Fixture.league_id).\
    filter(League.season == args.season).\
    group_by(League.id).all()

for league_id, match_count in existing_data:
    if match_count > 0:
        existing_leagues.add(league_id)
        logger.info(f'League ID {league_id} already has {match_count} matches in the database for season {args.season}')
```

**Purpose**: Prevents re-processing entire leagues that already have data when `--skip-existing` flag is used.

**Behavior**: 
- Checks if a league already has fixtures in the database for the specified season
- If `--skip-existing` is enabled and the league has data, skips the entire league
- Logs the number of existing matches for transparency

### 2. Match-Level Duplicate Detection

**Location**: Lines 318-340 in `historical_fetcher.py`

```python
# Get existing match dates for this league to avoid duplicates
existing_matches = set()
db_matches = session.query(Fixture.home_team_id, Fixture.away_team_id, Fixture.match_date).\
    filter(Fixture.league_id == league_id).all()

for home_id, away_id, match_date in db_matches:
    match_key = f"{home_id}_{away_id}_{match_date.strftime('%Y-%m-%d')}"
    existing_matches.add(match_key)

# Filter out matches that already exist in the database
new_matches = []
for match in matches:
    match_date = datetime.strptime(match.date, '%Y-%m-%d')
    match_key = f"{match.home_team_id}_{match.away_team_id}_{match.date}"
    
    if match_key not in existing_matches:
        new_matches.append(match)
    else:
        logger.debug(f'Skipping duplicate match: {match.home_team_name} vs {match.away_team_name} on {match.date}')
```

**Purpose**: Prevents duplicate match records at the individual fixture level.

**Key Components**:

1. **Match Key Generation**: Creates unique identifiers using `home_team_id_away_team_id_date` format
2. **Existing Match Query**: Retrieves all existing fixtures for the league from the database
3. **Duplicate Filtering**: Compares fetched matches against existing ones using the match key
4. **Logging**: Records skipped duplicates at debug level for audit purposes

### 3. File-Level Duplicate Prevention

**Location**: Lines 856-866 in `DataFetcher.py`

```python
# Skip if file exists and is recent (less than 24 hours old)
if os.path.exists(historical_file):
    file_age = time.time() - os.path.getmtime(historical_file)
    if file_age < 24 * 3600 and os.path.getsize(historical_file) > 200:
        self.logger.info(f"Recent historical data exists for {league_name}, skipping")
        return
```

**Purpose**: Prevents re-fetching data when recent CSV files already exist.

**Behavior**:
- Checks if CSV file exists and is less than 24 hours old
- Verifies file size is reasonable (> 200 bytes)
- Skips fetching if recent data exists

## Duplicate Handling Flow

1. **Pre-Processing Check**: 
   - Check if league already has data (when `--skip-existing` is used)
   - Skip entire league if data exists

2. **Data Fetching**:
   - Fetch matches from API regardless of existing data
   - This ensures we have the latest data for comparison

3. **Database Save Phase**:
   - Query existing fixtures for the league
   - Generate match keys for all existing fixtures
   - Filter new matches to exclude duplicates
   - Save only new, non-duplicate matches

4. **Logging and Reporting**:
   - Log number of duplicates filtered
   - Report new matches added vs total matches fetched
   - Provide transparency in the summary statistics

## Benefits

1. **Data Integrity**: Prevents duplicate records in the database
2. **Performance**: Avoids unnecessary database writes for existing data
3. **Transparency**: Clear logging of what was skipped and why
4. **Flexibility**: Can be controlled via command-line flags
5. **Robustness**: Works at multiple levels (league, match, file)

## Usage Examples

```bash
# Skip leagues that already have data
python historical_fetcher.py --leagues 39,40 --skip-existing

# Force re-fetch but still filter duplicates at match level
python historical_fetcher.py --leagues 39,40
```

## Related Components

- **Database Models**: `Fixture`, `League` models define the data structure
- **Session Management**: Uses `get_db_session()` for database operations
- **Logging**: Comprehensive logging at info and debug levels
- **Summary Statistics**: Tracks and reports duplicate filtering results