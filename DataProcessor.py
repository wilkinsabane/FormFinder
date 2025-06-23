import pandas as pd
import logging
import os
import json

# Ensure sdata_init_config.json is in the same directory or provide correct path
# It's better to pass these as arguments or load them inside the class/main
try:
    with open('sdata_init_config.json', 'r') as f:
        PROJECT_CONFIG = json.load(f)
    RECENT_PERIOD = PROJECT_CONFIG['recent_period']
    WIN_RATE_THRESHOLD = PROJECT_CONFIG['win_rate_threshold']
except FileNotFoundError:
    logging.warning("sdata_init_config.json not found. Using default values for RECENT_PERIOD and WIN_RATE_THRESHOLD.")
    RECENT_PERIOD = 10
    WIN_RATE_THRESHOLD = 0.7
except json.JSONDecodeError:
    logging.error("Error decoding sdata_init_config.json. Using default values.")
    RECENT_PERIOD = 10
    WIN_RATE_THRESHOLD = 0.7

# Configure logging (ensure this path is correct if running from cron)
# Consider using an absolute path or a logs directory consistent with DataFetcher
LOG_PROCESSOR_DIR = 'data/logs' # Aligning with DataFetcher's log dir structure
os.makedirs(LOG_PROCESSOR_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_PROCESSOR_DIR,'data_processor.log'), # Changed path
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class DataProcessor:
    """A class to process historical match data and identify high-form teams."""
    
    def __init__(self, recent_period=10, win_rate_threshold=0.7):
        """Initialize the DataProcessor with configurable parameters."""
        self.recent_period = recent_period
        self.win_rate_threshold = win_rate_threshold
        logging.info(f"Initialized DataProcessor with recent_period={self.recent_period}, win_rate_threshold={self.win_rate_threshold}")

    def load_matches(self, filepath):
        """Load historical match data from a CSV file."""
        try:
            df = pd.read_csv(filepath)
            
            # Ensure 'date' column exists before trying to convert
            if 'date' in df.columns:
                # FIXED: Try multiple date formats to handle both DataFetcher output and other formats
                original_count = len(df)
                
                # First, try the DataFetcher standard format (YYYY-MM-DD)
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
                
                # If that didn't work for all dates, try other common formats
                mask_invalid = df['date'].isnull()
                if mask_invalid.any():
                    logging.info(f"Trying alternative date formats for {mask_invalid.sum()} dates")
                    
                    # Try DD/MM/YYYY format
                    df.loc[mask_invalid, 'date'] = pd.to_datetime(
                        df.loc[mask_invalid, 'date'], 
                        format='%d/%m/%Y', 
                        errors='coerce'
                    )
                    
                    # Try general pandas inference for remaining dates
                    mask_still_invalid = df['date'].isnull()
                    if mask_still_invalid.any():
                        df.loc[mask_still_invalid, 'date'] = pd.to_datetime(
                            df.loc[mask_still_invalid, 'date'], 
                            errors='coerce'
                        )
                
                # Log results of date parsing
                valid_dates = df['date'].notna().sum()
                invalid_dates = original_count - valid_dates
                
                logging.info(f"Date parsing results: {valid_dates} valid dates, {invalid_dates} invalid dates")
                
                if invalid_dates > 0:
                    logging.warning(f"Found {invalid_dates} rows with unparseable dates")
                    # Show sample of invalid dates for debugging
                    invalid_sample = df[df['date'].isnull()]['date'].head(5).tolist()
                    logging.warning(f"Sample invalid dates: {invalid_sample}")
                
                # Optionally drop rows with invalid dates if they're critical for analysis
                # df.dropna(subset=['date'], inplace=True)
            else:
                logging.warning(f"'date' column not found in {filepath}")
            
            logging.info(f"Loaded {len(df)} matches from {filepath}")
            return df
        except pd.errors.EmptyDataError:
            logging.warning(f"No data or empty file: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Failed to load matches from {filepath}: {e}")
            return pd.DataFrame()

    def calculate_win_rate(self, team_id, matches_df): # Renamed matches to matches_df
        """Calculate the win rate for a team based on their last N games."""
        # Filter for matches involving the team
        team_matches = matches_df[
            (matches_df['home_team_id'] == team_id) | (matches_df['away_team_id'] == team_id)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        if 'date' not in team_matches.columns:
            logging.warning(f"No 'date' column found for team {team_id}. Cannot calculate win rate.")
            return 0.0
            
        # FIXED: Better handling of null dates
        valid_date_matches = team_matches.dropna(subset=['date'])
        
        if len(valid_date_matches) == 0:
            logging.warning(f"No matches with valid dates for team {team_id}. Cannot calculate win rate.")
            return 0.0

        # Sort by date and select the most recent N games
        team_matches_sorted = valid_date_matches.sort_values(by='date', ascending=False).head(self.recent_period)
        
        if len(team_matches_sorted) == 0:
            return 0.0
        
        wins = 0
        valid_games_for_win_rate = 0
        for _, match in team_matches_sorted.iterrows():
            if match['status'] != 'finished':
                continue
            
            # Convert scores to numeric, coercing errors. N/A or invalid scores become NaN.
            home_score = pd.to_numeric(match.get('home_score'), errors='coerce')
            away_score = pd.to_numeric(match.get('away_score'), errors='coerce')
            
            if pd.isna(home_score) or pd.isna(away_score):
                continue
            
            valid_games_for_win_rate += 1

            if (match['home_team_id'] == team_id and home_score > away_score) or \
               (match['away_team_id'] == team_id and away_score > home_score):
                wins += 1
        
        if valid_games_for_win_rate == 0:
            return 0.0
        
        win_rate = wins / valid_games_for_win_rate
        logging.debug(f"Team {team_id}: win rate {win_rate:.2f} over {valid_games_for_win_rate} valid recent games.")
        return win_rate

    def process_league(self, filepath):
        """Process matches for a league and identify high-performing teams."""
        matches = self.load_matches(filepath)
        if matches.empty:
            logging.info(f"No matches loaded from {filepath}, cannot process league.")
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate']) # Return empty df with columns
        
        # FIXED: Convert team IDs to string first, then to numeric to handle mixed types
        home_team_ids = pd.to_numeric(matches['home_team_id'].astype(str), errors='coerce').dropna()
        away_team_ids = pd.to_numeric(matches['away_team_id'].astype(str), errors='coerce').dropna()
        
        all_team_ids = pd.concat([home_team_ids, away_team_ids]).astype(int).unique()
        
        high_form_teams_data = []
        
        for team_id in all_team_ids:
            win_rate = self.calculate_win_rate(team_id, matches)
            if win_rate >= self.win_rate_threshold: # Changed to >= to include threshold
                # Attempt to get team_name, prioritize home_team_name then away_team_name
                team_name_series = matches.loc[matches['home_team_id'] == team_id, 'home_team_name']
                if team_name_series.empty:
                    team_name_series = matches.loc[matches['away_team_id'] == team_id, 'away_team_name']
                
                team_name = team_name_series.iloc[0] if not team_name_series.empty else f"TeamId-{team_id}"
                
                high_form_teams_data.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'win_rate': win_rate
                })
        
        high_form_df = pd.DataFrame(high_form_teams_data, columns=['team_id', 'team_name', 'win_rate'])
        if not high_form_df.empty:
            logging.info(f"Identified {len(high_form_df)} high-form teams from {filepath}")
        else:
            logging.info(f"No high-form teams met threshold from {filepath}")
        return high_form_df

    def save_high_form_teams(self, high_form_df, output_filepath):
        """Save the high-form teams to a CSV file. Creates an empty file if df is empty."""
        expected_columns = ['team_id', 'team_name', 'win_rate']
        if high_form_df.empty:
            logging.info(f"No high-form teams identified. Saving empty file with headers to {output_filepath}")
            pd.DataFrame(columns=expected_columns).to_csv(output_filepath, index=False)
        else:
            # Ensure DataFrame has the expected columns before saving
            df_to_save = high_form_df.reindex(columns=expected_columns)
            df_to_save.to_csv(output_filepath, index=False)
            logging.info(f"Saved {len(df_to_save)} high-form teams to {output_filepath}")

    @staticmethod
    def extract_league_id(filename):
        """Extract league ID from a filename: league_{league_id}_...csv"""
        # Example: league_123_2023-2024_historical_matches.csv -> 123
        # Example: league_456_upcoming_fixtures.csv -> 456
        if filename.startswith('league_') and filename.endswith('.csv'):
            parts = filename.split('_')
            if len(parts) > 1:
                try:
                    return int(parts[1])
                except ValueError:
                    logging.warning(f"Could not parse league_id from {parts[1]} in {filename}")
                    pass
        logging.warning(f"Cannot extract league_id from filename: {filename}")
        return None

    def run(self, input_filepaths, output_dir='processed_data'):
        """Process historical match files and generate high-form team files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        for filepath in input_filepaths:
            filename = os.path.basename(filepath)
            logging.info(f"Processing historical file: {filename}")
            league_id = self.extract_league_id(filename) # Expects league_{id}_...
            
            if league_id is None:
                logging.warning(f"Skipping {filename} as league_id could not be extracted.")
                continue
            
            high_form_df = self.process_league(filepath) # Returns df, possibly empty but with columns
            output_filepath = os.path.join(output_dir, f"league_{league_id}_high_form_teams.csv")
            self.save_high_form_teams(high_form_df, output_filepath) # Will now always save a file

def get_historical_files(directory='data/historical'):
    """Retrieve all historical match CSV files from the specified directory."""
    if not os.path.exists(directory):
        logging.error(f"Historical data directory {directory} does not exist")
        return []
    
    # Expecting filenames like: league_{id}_{season}_historical_matches.csv
    csv_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.startswith('league_') and f.endswith('_historical_matches.csv') and os.path.isfile(os.path.join(directory, f))
    ]
    logging.info(f"Found {len(csv_files)} historical CSV files in {directory}")
    return csv_files

if __name__ == "__main__":
    historical_input_files = get_historical_files() # Corrected variable name
    if not historical_input_files:
        logging.warning("No input historical files found by get_historical_files(), DataProcessor exiting.")
    else:
        # Use RECENT_PERIOD and WIN_RATE_THRESHOLD loaded from config or defaults
        processor = DataProcessor(recent_period=RECENT_PERIOD, win_rate_threshold=WIN_RATE_THRESHOLD)
        processor.run(historical_input_files)
