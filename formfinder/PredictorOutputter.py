import pandas as pd
import os
import json
from datetime import datetime
import logging

# Define the log directory (aligning with DataFetcher's log dir structure)
LOG_PREDICTOR_DIR = 'data/logs' # Changed to data/logs
os.makedirs(LOG_PREDICTOR_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_PREDICTOR_DIR, 'predictor_outputter.log'), # Changed path
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class PredictorOutputter:
    """A class to predict outcomes and output results."""

    def __init__(self, leagues_filepath='leagues.json'):
        self.leagues_data = self._load_leagues_data(leagues_filepath)

    def _load_leagues_data(self, filepath):
        """Loads league data and creates mappings from league_id to country and league name."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            league_data = {}
            for league in data.get('results', []):
                league_data[league['id']] = {
                    'country': league['country']['name'],
                    'name': league['name']
                }
            
            logging.info(f"Loaded league data for {len(league_data)} leagues.")
            return league_data
        except FileNotFoundError:
            logging.error(f"Leagues file not found at {filepath}. League data will not be available.")
            return {}
        except Exception as e:
            logging.error(f"Error loading leagues file {filepath}: {e}")
            return {}

    def generate_predictions(self, processed_data_list):
        """Generate predictions from processed data list.
        
        Args:
            processed_data_list: List of dicts with 'processed_data' containing DataFrames
            
        Returns:
            List of prediction dictionaries
        """
        try:
            predictions = []
            
            for data_item in processed_data_list:
                if data_item.get('status') != 'success':
                    continue
                    
                league_id = data_item.get('league_id')
                processed_data = data_item.get('processed_data')
                
                if processed_data is None or processed_data.empty:
                    logging.info(f"No processed data for league {league_id}, skipping predictions")
                    continue
                
                league_info = self.leagues_data.get(league_id, {'country': 'Unknown', 'name': 'Unknown'})
                country_name = league_info['country']
                league_name = league_info['name']

                # Create predictions from high-form teams data
                for _, team_row in processed_data.iterrows():
                    prediction = {
                        'league_id': league_id,
                        'league_name': league_name,
                        'country': country_name,
                        'team_id': team_row.get('team_id'),
                        'team_name': team_row.get('team_name'),
                        'win_rate': team_row.get('win_rate'),
                        'prediction_type': 'high_form_team',
                        'confidence': team_row.get('win_rate', 0.0)
                    }
                    predictions.append(prediction)
            
            logging.info(f"Generated {len(predictions)} predictions from {len(processed_data_list)} data items")
            return predictions
            
        except Exception as e:
            logging.error(f"Error generating predictions: {e}")
            return []

    def load_high_form_teams(self, filepath):
        """Load high-form teams data from a CSV file."""
        try:
            # Ensure team_id is read as string first to handle N/A or mixed types, then convert
            df = pd.read_csv(filepath, dtype={'team_id': str})
            df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
            # df.dropna(subset=['team_id'], inplace=True) # Optional: remove rows where team_id became NaN
            # df['team_id'] = df['team_id'].astype(int) # Convert to int if all are valid numbers
            logging.info(f"Loaded {len(df)} high-form teams from {filepath}")
            return df
        except pd.errors.EmptyDataError:
            logging.info(f"High-form teams file is empty: {filepath}")
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate']) # Return empty df with expected columns
        except FileNotFoundError:
            logging.warning(f"High-form teams file not found: {filepath}")
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate'])
        except Exception as e:
            logging.error(f"Failed to load high-form teams from {filepath}: {e}")
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate'])


    def load_fixtures(self, filepath):
        """Load upcoming fixtures data from a CSV file."""
        try:
            # Ensure team_ids are read as strings first, then convert
            df = pd.read_csv(filepath, dtype={'home_team_id': str, 'away_team_id': str})
            df['home_team_id'] = pd.to_numeric(df['home_team_id'], errors='coerce')
            df['away_team_id'] = pd.to_numeric(df['away_team_id'], errors='coerce')
            # df.dropna(subset=['home_team_id', 'away_team_id'], how='any', inplace=True) # Optional
            # df['home_team_id'] = df['home_team_id'].astype(int) # Optional
            # df['away_team_id'] = df['away_team_id'].astype(int) # Optional
            logging.info(f"Loaded {len(df)} fixtures from {filepath}")
            return df
        except pd.errors.EmptyDataError:
            logging.info(f"Fixtures file is empty: {filepath}")
            return pd.DataFrame() # Return empty df
        except FileNotFoundError:
            logging.warning(f"Fixtures file not found: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Failed to load fixtures from {filepath}: {e}")
            return pd.DataFrame()

    # extract_league_id is identical to the one in DataProcessor
    # To avoid duplication, this could be in a shared utility module
    def extract_league_id(self, filename):
        """Extract league ID from a filename: league_{league_id}_...csv"""
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


    def process_league(self, league_id, high_form_file_path, fixtures_file_path):
        """Process a single league to find flagged matches with high win potential."""
        high_form_teams = self.load_high_form_teams(high_form_file_path)
        if high_form_teams.empty: # load_high_form_teams now returns empty df on error/empty file
            logging.info(f"No high-form teams data available for league ID {league_id} from {high_form_file_path}, skipping processing for this league.")
            return pd.DataFrame() # Return empty DataFrame
        
        fixtures = self.load_fixtures(fixtures_file_path)
        if fixtures.empty: # load_fixtures now returns empty df on error/empty file
            logging.info(f"No fixtures data available for league ID {league_id} from {fixtures_file_path}, skipping processing for this league.")
            return pd.DataFrame()
        
        # Ensure team_id columns are numeric for merging/mapping
        # High form teams 'team_id' should already be numeric from load_high_form_teams
        # Fixtures 'home_team_id' and 'away_team_id' should be numeric from load_fixtures

        # Create a dictionary for win rates: team_id -> win_rate
        # Filter out NaN team_ids from high_form_teams before setting index
        high_form_teams_cleaned = high_form_teams.dropna(subset=['team_id'])
        if 'team_id' not in high_form_teams_cleaned.columns or 'win_rate' not in high_form_teams_cleaned.columns:
            logging.error(f"High form teams data for league {league_id} is missing 'team_id' or 'win_rate' columns.")
            return pd.DataFrame()

        win_rate_dict = high_form_teams_cleaned.set_index('team_id')['win_rate'].to_dict()
        
        # Add league information to fixtures (useful for combined output)
        league_info = self.leagues_data.get(league_id, {'country': 'Unknown', 'name': 'Unknown'})
        fixtures['league_id'] = league_id # This is the ID extracted from filename
        fixtures['league_name'] = league_info['name']
        fixtures['country'] = league_info['country']


        # Map win rates to home and away teams in fixtures
        fixtures['home_win_rate'] = fixtures['home_team_id'].map(win_rate_dict)
        fixtures['away_win_rate'] = fixtures['away_team_id'].map(win_rate_dict)
        
        # Filter for matches where at least one team has a win rate (i.e., is in high_form_teams)
        flagged_fixtures = fixtures[
            fixtures['home_win_rate'].notnull() | fixtures['away_win_rate'].notnull()
        ].copy() # Use .copy()
        
        # Select and order output columns
        # Include league_name and country for better readability
        output_columns = [
            'league_id', 'league_name', 'country', 'match_id', 'date', 'time', 
            'home_team_name', 'home_win_rate',
            'away_team_name', 'away_win_rate'
        ]
        # Ensure all output columns exist in flagged_fixtures, add if missing (e.g. with None or N/A)
        for col in output_columns:
            if col not in flagged_fixtures.columns:
                flagged_fixtures[col] = None # or pd.NA

        return flagged_fixtures[output_columns]


    def run_predictor_outputter(self, fixtures_dir='data/fixtures', processed_dir='processed_data', output_dir='data/predictions'):
        """Run the Predictor/Outputter to generate daily predictions."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        if not os.path.exists(fixtures_dir):
            logging.error(f"Fixtures directory {fixtures_dir} does not exist. Cannot run predictor.")
            return
        if not os.path.exists(processed_dir):
            logging.error(f"Processed data directory {processed_dir} does not exist. Cannot run predictor.")
            return

        # Expecting filenames like: league_{id}_upcoming_fixtures.csv
        fixture_filenames = [f for f in os.listdir(fixtures_dir) if f.startswith('league_') and f.endswith('_upcoming_fixtures.csv')]
        logging.info(f"Found {len(fixture_filenames)} fixture files to process in {fixtures_dir}")
        
        all_flagged_matches_list = []
        
        for fixture_filename in fixture_filenames:
            league_id = self.extract_league_id(fixture_filename) # Expects league_{id}_...
            if league_id is None:
                logging.warning(f"Could not extract league_id from {fixture_filename}, skipping.")
                continue
            
            # Construct paths to input files for this league
            # high_form_teams file is named without league_name or season
            high_form_file_path = os.path.join(processed_dir, f"league_{league_id}_high_form_teams.csv")
            # fixture_file path uses the exact filename found in the directory
            fixtures_file_path = os.path.join(fixtures_dir, fixture_filename)
            
            logging.info(f"Processing league ID {league_id}: using high-form file '{high_form_file_path}' and fixtures file '{fixtures_file_path}'")
            
            flagged_league_df = self.process_league(league_id, high_form_file_path, fixtures_file_path)
            if not flagged_league_df.empty:
                all_flagged_matches_list.append(flagged_league_df)
            else:
                logging.info(f"No flagged matches found for league ID {league_id}.")
                
        if all_flagged_matches_list:
            combined_flagged_matches = pd.concat(all_flagged_matches_list, ignore_index=True)
            
            # Convert 'date' to datetime for sorting, handle errors
            if 'date' in combined_flagged_matches.columns:
                try:
                    combined_flagged_matches['date_dt'] = pd.to_datetime(combined_flagged_matches['date'], format='%d/%m/%Y', errors='coerce')
                except ValueError: # If format is mixed or different
                     combined_flagged_matches['date_dt'] = pd.to_datetime(combined_flagged_matches['date'], errors='coerce')


                # Sort by date (and time, if time column is consistently formatted)
                # Ensure 'time' column is suitable for sorting, e.g., HH:MM
                if 'time' in combined_flagged_matches.columns and 'date_dt' in combined_flagged_matches.columns:
                    combined_flagged_matches = combined_flagged_matches.sort_values(by=['date_dt', 'time'])
                elif 'date_dt' in combined_flagged_matches.columns:
                     combined_flagged_matches = combined_flagged_matches.sort_values(by=['date_dt'])
                
                if 'date_dt' in combined_flagged_matches.columns: # remove temporary sort column
                    combined_flagged_matches.drop(columns=['date_dt'], inplace=True)


            # Save to a timestamped CSV
            date_str = datetime.now().strftime("%Y%m%d")
            output_filename = os.path.join(output_dir, f"predictions_{date_str}.csv")
            try:
                combined_flagged_matches.to_csv(output_filename, index=False)
                logging.info(f"Saved {len(combined_flagged_matches)} total flagged matches to {output_filename}")
            except Exception as e:
                logging.error(f"Error saving combined predictions to {output_filename}: {e}")
                
        else:
            logging.info("No flagged matches found across all leagues for today.")

if __name__ == "__main__":
    # Define directories or load from a config
    FIXTURES_INPUT_DIR = 'data/fixtures'
    PROCESSED_INPUT_DIR = 'processed_data'
    PREDICTIONS_OUTPUT_DIR = 'data/predictions' # This should match the LOG_DIR for output if desired
                                                # or a separate 'output' dir within data/predictions
    
    # Create the main output directory for predictions CSVs if it doesn't exist
    os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)

    predictor = PredictorOutputter()
    predictor.run_predictor_outputter(
        fixtures_dir=FIXTURES_INPUT_DIR,
        processed_dir=PROCESSED_INPUT_DIR,
        output_dir=PREDICTIONS_OUTPUT_DIR
    )
