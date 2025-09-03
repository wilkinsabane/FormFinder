#!/usr/bin/env python
"""
Standalone Data Processor for FormFinder

A robust script to process soccer match data, calculate team form, and identify high-performing teams.
This standalone version includes database integration, enhanced error handling, and verbose logging.

Usage:
    python standalone_data_processor.py [--config CONFIG_PATH] [--db-only] [--file-only] [--output-dir DIR]
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import pandas as pd
from sqlalchemy import func, and_, or_
from sqlalchemy.exc import SQLAlchemyError

# Import from formfinder package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formfinder.config import load_config as load_formfinder_config
from formfinder.database import (
    DatabaseManager, League, Team, Fixture, Standing, Prediction,
    DataFetchLog, HighFormTeam
)

# Initialize formfinder configuration
load_formfinder_config()

# Initialize database connection
db_manager = DatabaseManager()

# Create session functions to match the expected interface with retry mechanism
def get_db_session(max_retries=3, retry_delay=1.0):
    """
    Get a database session with retry mechanism for transient errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        SQLAlchemy session
    """
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            return db_manager.get_session()
        except SQLAlchemyError as e:
            last_error = e
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"Database connection error, retrying ({retry_count}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                raise
    
    # This should not be reached, but just in case
    if last_error:
        raise last_error
    return db_manager.get_session()

def init_database():
    try:
        db_manager.create_tables()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        logger.debug(traceback.format_exc())
        raise
    
def close_database():
    try:
        db_manager.close()
        logger.info("Database connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")
        logger.debug(traceback.format_exc())

# Configure logging
LOG_DIR = Path('data/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'standalone_data_processor.log'

# Configure logger
logger = logging.getLogger('standalone_data_processor')
logger.setLevel(logging.DEBUG)

# File handler with rotation
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger.addHandler(console_handler)

# Default configuration
DEFAULT_CONFIG = {
    'recent_period': 10,
    'win_rate_threshold': 0.7,
    'min_matches_required': 5,
    'output_directory': 'processed_data',
    'database': {
        'enabled': True,
        'log_operations': True
    },
    'file_processing': {
        'enabled': True,
        'historical_dir': 'data/historical'
    }
}


class DataProcessorError(Exception):
    """Custom exception for DataProcessor errors."""
    pass


class DataProcessor:
    """Enhanced data processor with database integration and robust error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        # Validate and set configuration parameters with defaults
        self.recent_period = self._validate_int_param(config.get('recent_period'), 10, 'recent_period')
        self.win_rate_threshold = self._validate_float_param(config.get('win_rate_threshold'), 0.7, 'win_rate_threshold')
        self.min_matches_required = self._validate_int_param(config.get('min_matches_required'), 5, 'min_matches_required')
        self.output_dir = config.get('output_directory', 'processed_data')
        self.db_enabled = bool(config.get('database', {}).get('enabled', True))
        self.file_enabled = bool(config.get('file_processing', {}).get('enabled', True))
        self.historical_dir = config.get('file_processing', {}).get('historical_dir', 'data/historical')
        self.log_db_operations = bool(config.get('database', {}).get('log_operations', True))
        
        logger.info(f"Initialized DataProcessor with:")
        logger.info(f"  - recent_period: {self.recent_period}")
        logger.info(f"  - win_rate_threshold: {self.win_rate_threshold}")
        logger.info(f"  - min_matches_required: {self.min_matches_required}")
        logger.info(f"  - database enabled: {self.db_enabled}")
        logger.info(f"  - file processing enabled: {self.file_enabled}")
        logger.info(f"  - output directory: {self.output_dir}")
        logger.info(f"  - historical directory: {self.historical_dir}")
        
        # Create output directory if it doesn't exist
        if self.file_enabled and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        
        # Create historical directory if it doesn't exist and file processing is enabled
        if self.file_enabled and not os.path.exists(self.historical_dir):
            os.makedirs(self.historical_dir)
            logger.info(f"Created historical directory: {self.historical_dir}")
    
    def _validate_int_param(self, value, default, param_name):
        """Validate and convert an integer parameter.
        
        Args:
            value: The parameter value to validate
            default: Default value to use if validation fails
            param_name: Name of the parameter for logging
            
        Returns:
            Validated integer value
        """
        try:
            if value is None:
                return default
            validated = int(value)
            if validated <= 0:
                logger.warning(f"Parameter {param_name} must be positive, using default: {default}")
                return default
            return validated
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for {param_name}: {value}, using default: {default}")
            return default
    
    def _validate_float_param(self, value, default, param_name):
        """Validate and convert a float parameter.
        
        Args:
            value: The parameter value to validate
            default: Default value to use if validation fails
            param_name: Name of the parameter for logging
            
        Returns:
            Validated float value
        """
        try:
            if value is None:
                return default
            validated = float(value)
            if validated < 0 or validated > 1:
                logger.warning(f"Parameter {param_name} must be between 0 and 1, using default: {default}")
                return default
            return validated
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for {param_name}: {value}, using default: {default}")
            return default
    
    def load_matches_from_file(self, filepath: str) -> pd.DataFrame:
        """Load historical match data from a CSV file with robust error handling.
        
        Args:
            filepath: Path to the CSV file containing match data
            
        Returns:
            DataFrame containing match data or empty DataFrame on error
        """
        try:
            logger.debug(f"Loading matches from file: {filepath}")
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
                
            df = pd.read_csv(filepath)
            
            # Ensure 'date' column exists before trying to convert
            if 'date' in df.columns:
                # Try multiple date formats to handle both DataFetcher output and other formats
                original_count = len(df)
                
                # First, try the DataFetcher standard format (YYYY-MM-DD)
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
                
                # If that didn't work for all dates, try other common formats
                mask_invalid = df['date'].isnull()
                if mask_invalid.any():
                    logger.info(f"Trying alternative date formats for {mask_invalid.sum()} dates")
                    
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
                
                logger.info(f"Date parsing results: {valid_dates} valid dates, {invalid_dates} invalid dates")
                
                if invalid_dates > 0:
                    logger.warning(f"Found {invalid_dates} rows with unparseable dates")
                    # Show sample of invalid dates for debugging
                    if 'date' in df.columns:
                        invalid_sample = df[df['date'].isnull()]['date'].head(5).tolist()
                        logger.warning(f"Sample invalid dates: {invalid_sample}")
            else:
                logger.warning(f"'date' column not found in {filepath}")
            
            # Validate required columns
            required_columns = ['home_team_id', 'away_team_id', 'home_score', 'away_score', 'status']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in {filepath}: {missing_columns}")
            
            logger.info(f"Loaded {len(df)} matches from {filepath}")
            return df
        except pd.errors.EmptyDataError:
            logger.warning(f"No data or empty file: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load matches from {filepath}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def load_matches_from_db(self, league_id: int, season: Optional[str] = None) -> pd.DataFrame:
        """Load match data from the database for a specific league.
        
        Args:
            league_id: ID of the league to load matches for
            season: Optional season filter (e.g., '2023-2024')
            
        Returns:
            DataFrame containing match data or empty DataFrame on error
        """
        try:
            logger.debug(f"Loading matches from database for league_id={league_id}, season={season}")
            
            with get_db_session() as session:
                # Build query for fixtures
                query = session.query(Fixture)
                query = query.filter(Fixture.league_id == league_id)
                
                if season:
                    # Join with League to filter by season
                    query = query.join(League).filter(League.season == season)
                
                # Execute query
                fixtures = query.all()
                
                if not fixtures:
                    logger.info(f"No fixtures found in database for league_id={league_id}, season={season}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for fixture in fixtures:
                    # Get team names through relationships
                    home_team_name = fixture.home_team.name if fixture.home_team else f"Team-{fixture.home_team_id}"
                    away_team_name = fixture.away_team.name if fixture.away_team else f"Team-{fixture.away_team_id}"
                    
                    data.append({
                        'id': fixture.id,
                        'league_id': fixture.league_id,
                        'home_team_id': fixture.home_team_id,
                        'away_team_id': fixture.away_team_id,
                        'home_team_name': home_team_name,
                        'away_team_name': away_team_name,
                        'date': fixture.match_date,
                        'status': fixture.status,
                        'home_score': fixture.home_score,
                        'away_score': fixture.away_score,
                        'round_number': fixture.round_number
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} matches from database for league_id={league_id}")
                return df
                
        except SQLAlchemyError as e:
            logger.error(f"Database error loading matches for league_id={league_id}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading matches from database for league_id={league_id}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def calculate_win_rate(self, team_id: int, matches_df: pd.DataFrame) -> Tuple[float, int, int]:
        """Calculate the win rate for a team based on their last N games.
        
        Args:
            team_id: ID of the team to calculate win rate for
            matches_df: DataFrame containing match data
            
        Returns:
            Tuple of (win_rate, wins, valid_games_count)
        """
        try:
            # Filter for matches involving the team
            team_matches = matches_df[
                (matches_df['home_team_id'] == team_id) | (matches_df['away_team_id'] == team_id)
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning
            
            if team_matches.empty:
                logger.debug(f"No matches found for team {team_id}")
                return 0.0, 0, 0
            
            if 'date' not in team_matches.columns:
                logger.warning(f"No 'date' column found for team {team_id}. Cannot calculate win rate.")
                return 0.0, 0, 0
                
            # Better handling of null dates
            valid_date_matches = team_matches.dropna(subset=['date'])
            
            if len(valid_date_matches) == 0:
                logger.warning(f"No matches with valid dates for team {team_id}. Cannot calculate win rate.")
                return 0.0, 0, 0

            # Sort by date and select the most recent N games
            team_matches_sorted = valid_date_matches.sort_values(by='date', ascending=False).head(self.recent_period)
            
            if len(team_matches_sorted) == 0:
                return 0.0, 0, 0
            
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
                return 0.0, 0, 0
            
            win_rate = wins / valid_games_for_win_rate
            logger.debug(f"Team {team_id}: win rate {win_rate:.2f} over {valid_games_for_win_rate} valid recent games.")
            return win_rate, wins, valid_games_for_win_rate
        except Exception as e:
            logger.error(f"Error calculating win rate for team {team_id}: {e}")
            logger.debug(traceback.format_exc())
            return 0.0, 0

    def validate_match_data(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean match data before processing.
        
        Args:
            matches_df: DataFrame containing match data
            
        Returns:
            Cleaned and validated DataFrame
        """
        if matches_df.empty:
            return matches_df
            
        try:
            # Make a copy to avoid modifying the original
            df = matches_df.copy()
            
            # Ensure required columns exist
            required_columns = ['home_team_id', 'away_team_id', 'home_score', 'away_score', 'status']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                # Add missing columns with NaN values
                for col in missing_columns:
                    df[col] = pd.NA
            
            # Convert team IDs to numeric, handling mixed types
            for col in ['home_team_id', 'away_team_id']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
            
            # Convert scores to numeric
            for col in ['home_score', 'away_score']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Filter out rows with invalid team IDs
            valid_mask = df['home_team_id'].notna() & df['away_team_id'].notna()
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                logger.warning(f"Filtered out {invalid_count} rows with invalid team IDs")
                df = df[valid_mask]
            
            # Filter out rows with invalid dates if date column exists
            if 'date' in df.columns:
                date_valid_mask = df['date'].notna()
                date_invalid_count = (~date_valid_mask).sum()
                if date_invalid_count > 0:
                    logger.warning(f"Filtered out {date_invalid_count} rows with invalid dates")
                    df = df[date_valid_mask]
            
            # Ensure status column has valid values
            if 'status' in df.columns:
                # Fill missing status with 'unknown'
                df['status'] = df['status'].fillna('unknown')
                
                # Convert status to lowercase for consistency
                df['status'] = df['status'].str.lower()
                
                # Map various status values to standardized ones
                status_mapping = {
                    'ft': 'finished',
                    'full-time': 'finished',
                    'fulltime': 'finished',
                    'complete': 'finished',
                    'completed': 'finished',
                    'done': 'finished',
                    'ns': 'scheduled',
                    'not started': 'scheduled',
                    'notstarted': 'scheduled',
                    'upcoming': 'scheduled',
                    'pending': 'scheduled',
                    'tbd': 'scheduled',
                    'postponed': 'postponed',
                    'suspended': 'postponed',
                    'canceled': 'cancelled',
                    'cancelled': 'cancelled',
                    'abandoned': 'cancelled'
                }
                
                df['status'] = df['status'].map(lambda x: status_mapping.get(x, x))
            
            logger.info(f"Validated match data: {len(df)} valid rows remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error validating match data: {e}")
            logger.debug(traceback.format_exc())
            return matches_df
    
    def process_league_data(self, league_id: Optional[int] = None, fixtures: Optional[pd.DataFrame] = None, 
                          filepath: Optional[str] = None) -> pd.DataFrame:
        """Process matches for a league and identify high-performing teams.
        
        Args:
            league_id: ID of the league to process
            fixtures: DataFrame containing match data (if already loaded)
            filepath: Path to CSV file containing match data (if not loaded)
            
        Returns:
            DataFrame containing high-form teams data
        """
        try:
            # Load data if not provided
            if fixtures is None and filepath is not None:
                matches = self.load_matches_from_file(filepath)
                source_desc = f"file {filepath}"
            elif fixtures is not None:
                matches = fixtures
                source_desc = f"provided DataFrame for league {league_id}"
            else:
                logger.error("Either fixtures DataFrame or filepath must be provided")
                return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate', 'valid_games'])
            
            if matches.empty:
                logger.info(f"No matches loaded from {source_desc}, cannot process league.")
                return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate', 'valid_games'])
            
            # Ensure required columns exist
            required_columns = ['home_team_id', 'away_team_id', 'home_score', 'away_score']
            for col in required_columns:
                if col not in matches.columns:
                    logger.warning(f"Missing required column {col} in {source_desc}")
                    return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate', 'valid_games'])
            
            # Convert team IDs to string first, then to numeric to handle mixed types
            home_team_ids = pd.to_numeric(matches['home_team_id'].astype(str), errors='coerce').dropna()
            away_team_ids = pd.to_numeric(matches['away_team_id'].astype(str), errors='coerce').dropna()
            
            all_team_ids = pd.concat([home_team_ids, away_team_ids]).astype(int).unique()
            
            high_form_teams_data = []
            
            for team_id in all_team_ids:
                win_rate, wins, valid_games = self.calculate_win_rate(team_id, matches)
                
                # Only include teams with sufficient games and meeting threshold
                if valid_games >= self.min_matches_required and win_rate >= self.win_rate_threshold:
                    # Attempt to get team_name, prioritize home_team_name then away_team_name
                    team_name_series = matches.loc[matches['home_team_id'] == team_id, 'home_team_name']
                    if team_name_series.empty:
                        team_name_series = matches.loc[matches['away_team_id'] == team_id, 'away_team_name']
                    
                    team_name = team_name_series.iloc[0] if not team_name_series.empty else f"TeamId-{team_id}"
                    
                    high_form_teams_data.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'win_rate': win_rate,
                        'wins': wins,
                        'total_matches': valid_games
                    })
            
            high_form_df = pd.DataFrame(high_form_teams_data, 
                                       columns=['team_id', 'team_name', 'win_rate', 'wins', 'total_matches'])
            
            if not high_form_df.empty:
                logger.info(f"Identified {len(high_form_df)} high-form teams from {source_desc}")
                # Sort by win rate descending
                high_form_df = high_form_df.sort_values(by='win_rate', ascending=False)
            else:
                logger.info(f"No high-form teams met threshold from {source_desc}")
                
            return high_form_df
        except Exception as e:
            logger.error(f"Error processing league data: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate', 'valid_games'])

    def save_high_form_teams_to_file(self, high_form_df: pd.DataFrame, league_id: int, 
                                   output_filepath: Optional[str] = None) -> str:
        """Save the high-form teams to a CSV file.
        
        Args:
            high_form_df: DataFrame containing high-form teams data
            league_id: ID of the league
            output_filepath: Optional custom output filepath
            
        Returns:
            Path to the saved file
        """
        try:
            if output_filepath is None:
                output_filepath = os.path.join(self.output_dir, f"league_{league_id}_high_form_teams.csv")
            
            expected_columns = ['team_id', 'team_name', 'win_rate', 'wins', 'total_matches']
            
            if high_form_df.empty:
                logger.info(f"No high-form teams identified. Saving empty file with headers to {output_filepath}")
                pd.DataFrame(columns=expected_columns).to_csv(output_filepath, index=False)
            else:
                # Ensure DataFrame has the expected columns before saving
                df_to_save = high_form_df.reindex(columns=expected_columns)
                df_to_save.to_csv(output_filepath, index=False)
                logger.info(f"Saved {len(df_to_save)} high-form teams to {output_filepath}")
            
            return output_filepath
        except Exception as e:
            logger.error(f"Error saving high-form teams to file: {e}")
            logger.debug(traceback.format_exc())
            return ""

    def save_high_form_teams_to_db(self, high_form_df: pd.DataFrame, league_id: int) -> bool:
        """Save high-form teams data to the dedicated high_form_teams table.
        
        Args:
            high_form_df: DataFrame containing high-form teams data
            league_id: ID of the league
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db_enabled:
            logger.info("Database operations disabled, skipping save to database")
            return False
            
        if high_form_df.empty:
            logger.info(f"No high-form teams to save to database for league {league_id}")
            return True
            
        try:
            with get_db_session() as session:
                now = datetime.now(timezone.utc)
                # Calculate analysis period (typically last 30 days)
                analysis_start_date = now - timedelta(days=30)  # Assuming 30-day analysis window
                analysis_end_date = now
                
                # Save each high-form team to the database
                teams_saved = 0
                for _, row in high_form_df.iterrows():
                    team_id = int(row['team_id'])
                    win_rate = float(row['win_rate'])
                    total_matches = int(row['total_matches'])
                    wins = int(row['wins'])
                    
                    # Check if this team is already in the high_form_teams table
                    existing_entry = session.query(HighFormTeam).filter(
                        HighFormTeam.team_id == team_id,
                        HighFormTeam.league_id == league_id,
                        HighFormTeam.analysis_date >= now.date()
                    ).first()
                    
                    if existing_entry:
                        # Update existing entry
                        existing_entry.win_rate = win_rate
                        existing_entry.total_matches = total_matches
                        existing_entry.wins = wins
                        existing_entry.team_name = str(row['team_name'])
                        existing_entry.analysis_start_date = analysis_start_date
                        existing_entry.analysis_end_date = analysis_end_date
                        existing_entry.updated_at = now
                    else:
                        # Create new entry
                        high_form_team = HighFormTeam(
                            team_id=team_id,
                            league_id=league_id,
                            team_name=str(row['team_name']),
                            win_rate=win_rate,
                            total_matches=total_matches,
                            wins=wins,
                            analysis_start_date=analysis_start_date,
                            analysis_end_date=analysis_end_date,
                            algorithm_version="standalone_data_processor_v1",
                            analysis_date=now
                        )
                        session.add(high_form_team)
                    
                    teams_saved += 1
                
                # Also create predictions for upcoming fixtures using the high-form teams data
                self._create_predictions_from_high_form_teams(session, high_form_df, league_id)
                
                # Log the operation
                if self.log_db_operations:
                    log_entry = DataFetchLog(
                        league_id=league_id,
                        data_type="high_form_teams",
                        status="success",
                        records_fetched=teams_saved,
                        duration_seconds=0.0
                    )
                    session.add(log_entry)
                
                session.commit()
                logger.info(f"Saved/updated {teams_saved} high-form teams to database for league {league_id}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error saving high-form teams for league {league_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"Error saving high-form teams to database for league {league_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
            
    def _create_predictions_from_high_form_teams(self, session, high_form_df: pd.DataFrame, league_id: int) -> int:
        """Create predictions for upcoming fixtures using high-form teams data.
        
        Args:
            session: SQLAlchemy session
            high_form_df: DataFrame containing high-form teams data
            league_id: ID of the league
            
        Returns:
            Number of predictions created/updated
        """
        now = datetime.now(timezone.utc)
        # Get upcoming fixtures for high-form teams
        upcoming_fixtures_query = session.query(Fixture).filter(
            Fixture.league_id == league_id,
            Fixture.match_date > now,
            Fixture.status == 'scheduled',
            or_(
                Fixture.home_team_id.in_(high_form_df['team_id'].tolist()),
                Fixture.away_team_id.in_(high_form_df['team_id'].tolist())
            )
        ).order_by(Fixture.match_date)
        
        upcoming_fixtures = upcoming_fixtures_query.all()
        
        if not upcoming_fixtures:
            logger.info(f"No upcoming fixtures found for high-form teams in league {league_id}")
            return 0
        
        # Create predictions for upcoming fixtures
        predictions_created = 0
        for fixture in upcoming_fixtures:
            # Check if home or away team is high-form
            home_team_row = high_form_df[high_form_df['team_id'] == fixture.home_team_id]
            away_team_row = high_form_df[high_form_df['team_id'] == fixture.away_team_id]
            
            home_win_prob = float(home_team_row['win_rate'].iloc[0]) if not home_team_row.empty else 0.5
            away_win_prob = float(away_team_row['win_rate'].iloc[0]) if not away_team_row.empty else 0.5
            
            # Normalize probabilities
            total_prob = home_win_prob + away_win_prob
            if total_prob > 0:
                home_win_prob = home_win_prob / total_prob * 0.9  # Leave 10% for draw
                away_win_prob = away_win_prob / total_prob * 0.9
            else:
                home_win_prob = 0.45
                away_win_prob = 0.45
                
            draw_prob = 1.0 - home_win_prob - away_win_prob
            
            # Check if prediction already exists
            existing_prediction = session.query(Prediction).filter(
                Prediction.fixture_id == fixture.id
            ).first()
            
            if existing_prediction:
                # Update existing prediction
                existing_prediction.home_win_probability = home_win_prob
                existing_prediction.draw_probability = draw_prob
                existing_prediction.away_win_probability = away_win_prob
                existing_prediction.home_team_form_score = float(home_team_row['win_rate'].iloc[0]) if not home_team_row.empty else None
                existing_prediction.away_team_form_score = float(away_team_row['win_rate'].iloc[0]) if not away_team_row.empty else None
                existing_prediction.confidence_score = max(home_win_prob, away_win_prob, draw_prob)
                existing_prediction.updated_at = now
            else:
                # Create new prediction
                prediction = Prediction(
                    fixture_id=fixture.id,
                    home_win_probability=home_win_prob,
                    draw_probability=draw_prob,
                    away_win_probability=away_win_prob,
                    home_team_form_score=float(home_team_row['win_rate'].iloc[0]) if not home_team_row.empty else None,
                    away_team_form_score=float(away_team_row['win_rate'].iloc[0]) if not away_team_row.empty else None,
                    confidence_score=max(home_win_prob, away_win_prob, draw_prob),
                    algorithm_version="standalone_data_processor_v1",
                    features_used=json.dumps(["recent_form", "win_rate"]),
                    prediction_date=now
                )
                session.add(prediction)
            
            predictions_created += 1
            
        logger.info(f"Created/updated {predictions_created} predictions for upcoming fixtures in league {league_id}")
        return predictions_created

    @staticmethod
    def extract_league_id(filename: str) -> Optional[int]:
        """Extract league ID from a filename: historical_matches_{league_id}_...csv
        
        Args:
            filename: Filename to extract league ID from
            
        Returns:
            League ID as integer or None if not found
        """
        try:
            # Example: historical_matches_123_2023-2024.csv -> 123
            if filename.startswith('historical_matches_') and filename.endswith('.csv'):
                parts = filename.split('_')
                if len(parts) > 2:
                    try:
                        return int(parts[2])
                    except ValueError:
                        logger.warning(f"Could not parse league_id from {parts[2]} in {filename}")
                        return None
            logger.warning(f"Cannot extract league_id from filename: {filename}")
            return None
        except Exception as e:
            logger.error(f"Error extracting league ID from filename {filename}: {e}")
            return None

    def process_file(self, filepath: str) -> Tuple[pd.DataFrame, int]:
        """Process a single historical match file.
        
        Args:
            filepath: Path to the CSV file containing match data
            
        Returns:
            Tuple of (high_form_teams_df, league_id)
        """
        try:
            filename = os.path.basename(filepath)
            logger.info(f"Processing historical file: {filename}")
            
            league_id = self.extract_league_id(filename)
            if league_id is None:
                logger.warning(f"Skipping {filename} as league_id could not be extracted.")
                return pd.DataFrame(), 0
            
            high_form_df = self.process_league_data(filepath=filepath)
            
            if self.file_enabled:
                self.save_high_form_teams_to_file(high_form_df, league_id)
            
            if self.db_enabled:
                self.save_high_form_teams_to_db(high_form_df, league_id)
            
            return high_form_df, league_id
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame(), 0

    def process_db_league(self, league_id: int, season: Optional[str] = None) -> pd.DataFrame:
        """Process a league from the database.
        
        Args:
            league_id: ID of the league to process
            season: Optional season filter
            
        Returns:
            DataFrame containing high-form teams data
        """
        try:
            logger.info(f"Processing league {league_id} from database")
            
            # Load fixtures from database
            fixtures_df = self.load_matches_from_db(league_id, season)
            
            if fixtures_df.empty:
                logger.warning(f"No fixtures found for league {league_id}, season {season}")
                return pd.DataFrame()
            
            # Process league data
            high_form_df = self.process_league_data(league_id=league_id, fixtures=fixtures_df)
            
            if self.file_enabled:
                self.save_high_form_teams_to_file(high_form_df, league_id)
            
            if self.db_enabled:
                self.save_high_form_teams_to_db(high_form_df, league_id)
            
            return high_form_df
        except Exception as e:
            logger.error(f"Error processing league {league_id} from database: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def get_historical_files(self) -> List[str]:
        """Retrieve all historical match CSV files from the specified directory.
        
        Returns:
            List of file paths to historical match CSV files
        """
        try:
            if not os.path.exists(self.historical_dir):
                logger.error(f"Historical data directory {self.historical_dir} does not exist")
                return []
            
            # Expecting filenames like: historical_matches_{id}_{season}.csv
            csv_files = [
                os.path.join(self.historical_dir, f) for f in os.listdir(self.historical_dir)
                if f.startswith('historical_matches_') and f.endswith('.csv') 
                and os.path.isfile(os.path.join(self.historical_dir, f))
            ]
            
            logger.info(f"Found {len(csv_files)} historical CSV files in {self.historical_dir}")
            return csv_files
        except Exception as e:
            logger.error(f"Error getting historical files: {e}")
            logger.debug(traceback.format_exc())
            return []

    def get_leagues_from_db(self) -> List[Tuple[int, str]]:
        """Get all leagues from the database.
        
        Returns:
            List of tuples containing (league_id, season)
        """
        try:
            with get_db_session() as session:
                leagues = session.query(League.id, League.season).all()
                logger.info(f"Found {len(leagues)} leagues in database")
                return leagues
        except SQLAlchemyError as e:
            logger.error(f"Database error getting leagues: {e}")
            logger.debug(traceback.format_exc())
            return []
        except Exception as e:
            logger.error(f"Error getting leagues from database: {e}")
            logger.debug(traceback.format_exc())
            return []

    def run(self, db_only: bool = False, file_only: bool = False) -> Dict[str, Any]:
        """Run the data processor on all available data sources.
        
        Args:
            db_only: Only process data from the database
            file_only: Only process data from files
            
        Returns:
            Dictionary with processing results summary
        """
        start_time = datetime.now()
        logger.info(f"Starting data processing run at {start_time}")
        
        results = {
            'start_time': start_time.isoformat(),
            'file_processing': {'enabled': self.file_enabled and not db_only, 'leagues_processed': 0, 'high_form_teams': 0},
            'db_processing': {'enabled': self.db_enabled and not file_only, 'leagues_processed': 0, 'high_form_teams': 0},
            'errors': [],
            'end_time': None,
            'duration_seconds': None
        }
        
        try:
            # Process files if enabled and not db_only
            if self.file_enabled and not db_only:
                logger.info("Processing historical match files")
                historical_files = self.get_historical_files()
                
                if not historical_files:
                    logger.warning("No historical files found")
                    results['errors'].append("No historical files found")
                else:
                    processed_leagues = set()
                    total_high_form_teams = 0
                    
                    for filepath in historical_files:
                        high_form_df, league_id = self.process_file(filepath)
                        if league_id > 0:
                            processed_leagues.add(league_id)
                            total_high_form_teams += len(high_form_df)
                    
                    results['file_processing']['leagues_processed'] = len(processed_leagues)
                    results['file_processing']['high_form_teams'] = total_high_form_teams
                    logger.info(f"Processed {len(processed_leagues)} leagues from files, found {total_high_form_teams} high-form teams")
            
            # Process database if enabled and not file_only
            if self.db_enabled and not file_only:
                logger.info("Processing leagues from database")
                leagues = self.get_leagues_from_db()
                
                if not leagues:
                    logger.warning("No leagues found in database")
                    results['errors'].append("No leagues found in database")
                else:
                    total_high_form_teams = 0
                    
                    for league_id, season in leagues:
                        high_form_df = self.process_db_league(league_id, season)
                        total_high_form_teams += len(high_form_df)
                    
                    results['db_processing']['leagues_processed'] = len(leagues)
                    results['db_processing']['high_form_teams'] = total_high_form_teams
                    logger.info(f"Processed {len(leagues)} leagues from database, found {total_high_form_teams} high-form teams")
        except Exception as e:
            error_msg = f"Error during data processing run: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            results['errors'].append(error_msg)
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = duration
            
            logger.info(f"Data processing run completed at {end_time}, duration: {duration:.2f} seconds")
            
            # Save results to file
            results_file = os.path.join(LOG_DIR, f"data_processor_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {results_file}")
            except Exception as e:
                logger.error(f"Error saving results to file: {e}")
            
            return results


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    user_config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    user_config = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported config file format: {config_path}")
                    return config
                
            # Update config with user values
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    # Merge nested dictionaries
                    config[key].update(value)
                else:
                    config[key] = value
                    
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.debug(traceback.format_exc())
    else:
        logger.info("Using default configuration")
        
        # Try to load from sdata_init_config.json if it exists
        try:
            if os.path.exists('sdata_init_config.json'):
                with open('sdata_init_config.json', 'r') as f:
                    sdata_config = json.load(f)
                    
                if 'recent_period' in sdata_config:
                    config['recent_period'] = sdata_config['recent_period']
                    
                if 'win_rate_threshold' in sdata_config:
                    config['win_rate_threshold'] = sdata_config['win_rate_threshold']
                    
                logger.info("Loaded values from sdata_init_config.json")
        except Exception as e:
            logger.warning(f"Error loading from sdata_init_config.json: {e}")
    
    return config


def main():
    """Main entry point for the standalone data processor."""
    parser = argparse.ArgumentParser(description="Standalone Data Processor for FormFinder")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--db-only', action='store_true', help='Only process data from the database')
    parser.add_argument('--file-only', action='store_true', help='Only process data from files')
    parser.add_argument('--output-dir', help='Directory to save output files')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config['output_directory'] = args.output_dir
        
        # Initialize database if needed
        if config['database']['enabled'] and not args.file_only:
            try:
                # Database is already initialized in the import
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Error initializing database: {e}")
                logger.debug(traceback.format_exc())
                if not args.file_only:
                    logger.error("Database processing disabled due to initialization error")
                    config['database']['enabled'] = False
        
        # Create and run the processor
        processor = DataProcessor(config)
        results = processor.run(db_only=args.db_only, file_only=args.file_only)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Start time: {results['start_time']}")
        print(f"End time: {results['end_time']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        
        if results['file_processing']['enabled']:
            print(f"\nFile Processing:")
            print(f"  Leagues processed: {results['file_processing']['leagues_processed']}")
            print(f"  High-form teams found: {results['file_processing']['high_form_teams']}")
        
        if results['db_processing']['enabled']:
            print(f"\nDatabase Processing:")
            print(f"  Leagues processed: {results['db_processing']['leagues_processed']}")
            print(f"  High-form teams found: {results['db_processing']['high_form_teams']}")
        
        if results['errors']:
            print(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        # Close database connection
        if config['database']['enabled']:
            close_database()
            logger.info("Database connection closed")
        
        return 0
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        logger.debug(traceback.format_exc())
        print(f"Critical error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())