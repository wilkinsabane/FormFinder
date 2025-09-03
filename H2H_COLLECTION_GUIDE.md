# H2H Collection Service Guide

The H2H Collection Service has been enhanced to support multiple ways of inputting team IDs for head-to-head data collection.

## Usage Options

### 1. Configuration File Mode (Recommended)

Use a JSON configuration file to define team pairs organized by leagues:

```bash
python formfinder/h2h_collection_service.py --config-file h2h_teams_config.json --duration 60
```

**Configuration File Format (`h2h_teams_config.json`):**
```json
{
  "leagues": {
    "203": {
      "name": "Premier League",
      "team_pairs": [
        {
          "team1_id": 2689,
          "team2_id": 2693,
          "priority": "HIGH",
          "description": "Active fixture teams"
        },
        {
          "team1_id": 3258,
          "team2_id": 3259,
          "priority": "MEDIUM",
          "description": "Velez vs Zeljeznicar"
        }
      ]
    },
    "204": {
      "name": "La Liga",
      "team_pairs": [
        {
          "team1_id": 4061,
          "team2_id": 4063,
          "priority": "HIGH",
          "description": "Herediano vs Alajuelense"
        }
      ]
    }
  }
}
```

### 2. Manual Input Mode

Interactively input team pairs through the command line:

```bash
python formfinder/h2h_collection_service.py --manual --duration 30
```

You'll be prompted to enter:
- Team 1 ID
- Team 2 ID  
- League ID
- Priority (HIGH/MEDIUM/LOW)
- Whether to add more pairs

### 3. Default Test Mode

Run with default test team pairs:

```bash
python formfinder/h2h_collection_service.py --duration 45
```

## Command Line Arguments

- `--config-file`: Path to JSON configuration file
- `--manual`: Enable manual team ID input mode
- `--duration`: Service runtime in seconds (default: 60)

## Available Team and League IDs

You can find real team and league IDs by running:
```bash
python check_teams.py
```

This will show you:
- Available teams in the database
- Team IDs found in recent fixtures
- League information

## Automated Configuration Generation

For convenience, you can automatically generate team pair configurations based on your available leagues:

```bash
python generate_h2h_config.py
```

This script will:
- Read league IDs from `free_leagues.txt`
- Query the database for teams and recent fixtures in those leagues
- Generate optimized team pairs with priority levels
- Create `h2h_teams_config_auto.json` with real data
- Process up to 15 leagues with 10 team pairs each

The auto-generated config includes:
- **HIGH priority**: Recent fixture matchups (teams that played recently)
- **MEDIUM priority**: Active team combinations within the same league
- **LOW priority**: Additional team pairs for comprehensive coverage

### Using Auto-Generated Config

```bash
# Generate the configuration
python generate_h2h_config.py

# Use it with the H2H service
python formfinder/h2h_collection_service.py --config-file h2h_teams_config_auto.json --duration 60
```

Based on the database query, here are some real team IDs you can use:

**Teams:**
- 3258: Velez
- 3259: Zeljeznicar  
- 2920: Dinamo Zagreb
- 2928: HNK Gorica
- 4061: Herediano
- 4063: Alajuelense
- 3262: Borac Banja Luka
- 3263: Zrinjski
- 2749: (Additional team from fixtures)
- 2689, 2693: (Teams found in fixtures)

**Common League IDs:**
- 203: Premier League
- 204: La Liga
- 205: Serie A

## Priority Levels

- **HIGH**: Processed first, for urgent/important matchups
- **MEDIUM**: Standard priority
- **LOW**: Processed last, for background collection

## Service Features

- **Multi-worker processing**: 3 concurrent workers for efficient data collection
- **Priority queue**: Tasks processed based on priority level
- **Cache management**: Intelligent caching with freshness validation
- **Real-time monitoring**: Live status updates during execution
- **Graceful shutdown**: Clean service termination
- **Error handling**: Robust error recovery and logging

## Example Workflows

### Quick Test with Real Teams
```bash
# Use the provided config file
python formfinder/h2h_collection_service.py --config-file h2h_teams_config.json --duration 30
```

### Interactive Session
```bash
# Manual input for custom team pairs
python formfinder/h2h_collection_service.py --manual --duration 60
```

### Background Collection
```bash
# Long-running collection with config file
python formfinder/h2h_collection_service.py --config-file h2h_teams_config.json --duration 300
```

## Notes

- The service automatically handles database connections and API rate limiting
- Cache freshness is validated before processing
- Failed tasks are logged but don't stop the service
- Use Ctrl+C to stop the service early if needed
- Monitor logs for collection status and any errors