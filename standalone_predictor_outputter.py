#!/usr/bin/env python
"""
Standalone Predictor Outputter for FormFinder

A robust script to generate predictions from processed high-form team data and upcoming fixtures.
This standalone version includes database integration, enhanced error handling, and verbose logging.

Usage:
    python standalone_predictor_outputter.py [--config CONFIG_PATH] [--db-only] [--file-only] [--output-dir DIR]
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import pandas as pd
from sqlalchemy import func, and_, or_, case
from sqlalchemy.exc import SQLAlchemyError

# Import from formfinder package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formfinder.config import load_config as load_formfinder_config
from formfinder.database import (
    DatabaseManager, League, Team, Fixture, Standing, Prediction,
    DataFetchLog, HighFormTeam
)
from formfinder.sentiment import SentimentAnalyzer

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
    """
    Initialize database tables with error handling.
    """
    try:
        db_manager.create_tables()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        logger.debug(traceback.format_exc())
        raise
    
def close_database():
    """
    Close database connection with error handling.
    """
    try:
        db_manager.close()
        logger.info("Database connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")
        logger.debug(traceback.format_exc())

# Configure logging
LOG_DIR = Path('data/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'standalone_predictor_outputter.log'

# Configure logger
logger = logging.getLogger('standalone_predictor_outputter')
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
    'output_directory': 'data/predictions',
    'database': {
        'enabled': True,
        'log_operations': True
    },
    'file_processing': {
        'enabled': False,
        'fixtures_dir': 'data/fixtures',
        'processed_dir': 'processed_data'
    },
    'sentiment_analysis': {
        'enabled': True,
        'newsapi_key': 'ff008e7b4e9b4041ab44c50a729d7885',  # Add your NewsAPI.org API key here
        'weight_form': 0.7,  # Weight for form score in final prediction
        'weight_sentiment': 0.3,  # Weight for sentiment score in final prediction
        'cache_hours': 24  # Cache sentiment results for this many hours
    }
}


class PredictorOutputterError(Exception):
    """Custom exception for PredictorOutputter errors."""
    pass


class PredictorOutputter:
    """Enhanced predictor outputter with database integration and robust error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PredictorOutputter with configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        # Validate and set configuration parameters with defaults
        self.output_dir = self._validate_path_param(config.get('output_directory', 'data/predictions'), 'output_directory')
        self.db_enabled = bool(config.get('database', {}).get('enabled', True))
        self.file_enabled = bool(config.get('file_processing', {}).get('enabled', True))
        self.fixtures_dir = self._validate_path_param(config.get('file_processing', {}).get('fixtures_dir', 'data/fixtures'), 'fixtures_dir')
        self.processed_dir = self._validate_path_param(config.get('file_processing', {}).get('processed_dir', 'processed_data'), 'processed_dir')
        self.log_db_operations = bool(config.get('database', {}).get('log_operations', True))
        self.leagues_filepath = config.get('leagues_filepath', 'leagues.json')
        
        # Initialize sentiment analyzer
        sentiment_config = config.get('sentiment_analysis', {})
        self.sentiment_enabled = bool(sentiment_config.get('enabled', True))
        if self.sentiment_enabled:
            # Initialize SentimentAnalyzer - it will use all configured providers
            # including NewsData.io and TheNewsAPI, not just NewsAPI
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("Sentiment analysis enabled with multiple providers")
        else:
            self.sentiment_analyzer = None
            logger.info("Sentiment analysis disabled")
        
        logger.info(f"Initialized PredictorOutputter with:")
        logger.info(f"  - database enabled: {self.db_enabled}")
        logger.info(f"  - file processing enabled: {self.file_enabled}")
        logger.info(f"  - sentiment enabled: {self.sentiment_enabled}")
        logger.info(f"  - output directory: {self.output_dir}")
        logger.info(f"  - fixtures directory: {self.fixtures_dir}")
        logger.info(f"  - processed data directory: {self.processed_dir}")
        logger.info(f"  - leagues file: {self.leagues_filepath}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        
        # Load leagues data
        self.leagues_data = self._load_leagues_data(self.leagues_filepath)
        
        # Store configuration
        self.config = config
        
        # Initialize statistics
        self.stats = {
            'file_predictions': 0,
            'db_predictions': 0,
            'leagues_processed': 0,
            'errors': 0
        }
    
    def _validate_path_param(self, value, param_name):
        """
        Validate and normalize a path parameter.
        
        Args:
            value: The path value to validate
            param_name: Name of the parameter for logging
            
        Returns:
            Validated path string
        """
        if not value:
            default_value = DEFAULT_CONFIG.get(param_name, '')
            if not default_value and param_name in ['fixtures_dir', 'processed_dir']:
                default_value = DEFAULT_CONFIG.get('file_processing', {}).get(param_name, '')
            if not default_value and param_name == 'output_directory':
                default_value = 'data/predictions'
                
            logger.warning(f"Empty value for {param_name}, using default: {default_value}")
            return default_value
        
        # Normalize path
        normalized_path = os.path.normpath(value)
        return normalized_path

    def _load_leagues_data(self, filepath):
        """
        Loads league data and creates mappings from league_id to country and league name.
        
        Args:
            filepath: Path to the leagues JSON file
            
        Returns:
            Dictionary mapping league IDs to league information
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Leagues file not found at {filepath}. Will attempt to load from database.")
                return self._load_leagues_from_db()
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            league_data = {}
            for league in data.get('results', []):
                league_data[league['id']] = {
                    'country': league['country']['name'],
                    'name': league['name']
                }
            
            logger.info(f"Loaded league data for {len(league_data)} leagues from file.")
            return league_data
        except FileNotFoundError:
            logger.error(f"Leagues file not found at {filepath}. League data will not be available.")
            return self._load_leagues_from_db()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing leagues file {filepath}: {e}")
            return self._load_leagues_from_db()
        except Exception as e:
            logger.error(f"Error loading leagues file {filepath}: {e}")
            logger.debug(traceback.format_exc())
            return self._load_leagues_from_db()
    
    def _load_leagues_from_db(self):
        """
        Load league data from the database as a fallback.
        
        Returns:
            Dictionary mapping league IDs to league information
        """
        if not self.db_enabled:
            logger.warning("Database is disabled, cannot load leagues from database.")
            return {}
            
        try:
            session = get_db_session()
            leagues = session.query(League).all()
            
            league_data = {}
            for league in leagues:
                league_data[league.id] = {
                    'country': league.country or 'Unknown',
                    'name': league.name
                }
            
            session.close()
            logger.info(f"Loaded league data for {len(league_data)} leagues from database.")
            return league_data
        except Exception as e:
            logger.error(f"Error loading leagues from database: {e}")
            logger.debug(traceback.format_exc())
            return {}
            
    def get_sentiment_for_match(self, home_team_name, away_team_name, match_date=None, config=None):
        """
        Fetch news articles and analyze sentiment for both teams in a match.
        
        Args:
            home_team_name: Name of the home team
            away_team_name: Name of the away team
            match_date: Date of the match (optional, for more targeted article search)
            config: Configuration dictionary with sentiment analysis settings
            
        Returns:
            Tuple of (home_team_sentiment, away_team_sentiment, articles_analyzed)
        """
        if not self.sentiment_enabled or not self.sentiment_analyzer:
            logger.debug("Sentiment analysis is disabled.")
            return None, None, 0
            
        try:
            # Convert match_date to datetime if provided
            if match_date:
                try:
                    match_datetime = pd.to_datetime(match_date)
                except Exception as e:
                    logger.warning(f"Could not parse match date '{match_date}', using today: {e}")
                    match_datetime = datetime.now()
            else:
                match_datetime = datetime.now()
            
            # Use sentiment analyzer to get sentiment
            result = self.sentiment_analyzer.get_sentiment_for_match(
                home_team_name, away_team_name, match_datetime
            )
            
            total_articles = result.home_article_count + result.away_article_count
            
            logger.info(f"Sentiment analysis: {home_team_name}: {result.home_sentiment:.2f} "
                       f"({result.home_article_count} articles), "
                       f"{away_team_name}: {result.away_sentiment:.2f} "
                       f"({result.away_article_count} articles)")
            
            return result.home_sentiment, result.away_sentiment, total_articles
            
        except Exception as e:
            logger.error(f"Error performing sentiment analysis: {e}")
            logger.debug(traceback.format_exc())
            return None, None, 0
    

    


    def load_high_form_teams(self, filepath):
        """
        Load high-form teams data from a CSV file.
        
        Args:
            filepath: Path to the high-form teams CSV file
            
        Returns:
            DataFrame containing high-form teams data
        """
        try:
            # Ensure team_id is read as string first to handle N/A or mixed types, then convert
            df = pd.read_csv(filepath, dtype={'team_id': str})
            
            # Validate required columns
            required_columns = ['team_id', 'team_name', 'win_rate']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in high-form teams file: {missing_columns}")
                return pd.DataFrame(columns=required_columns)
            
            # Convert team_id to numeric
            df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
            
            # Drop rows with invalid team_id
            invalid_rows = df['team_id'].isna().sum()
            if invalid_rows > 0:
                logger.warning(f"Dropped {invalid_rows} rows with invalid team_id from {filepath}")
                df = df.dropna(subset=['team_id'])
            
            # Ensure win_rate is numeric
            if 'win_rate' in df.columns:
                df['win_rate'] = pd.to_numeric(df['win_rate'], errors='coerce')
                df = df.dropna(subset=['win_rate'])
            
            logger.info(f"Loaded {len(df)} high-form teams from {filepath}")
            return df
        except pd.errors.EmptyDataError:
            logger.info(f"High-form teams file is empty: {filepath}")
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate'])
        except FileNotFoundError:
            logger.warning(f"High-form teams file not found: {filepath}")
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate'])
        except Exception as e:
            logger.error(f"Failed to load high-form teams from {filepath}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame(columns=['team_id', 'team_name', 'win_rate'])

    def load_fixtures(self, filepath):
        """
        Load upcoming fixtures data from a CSV file.
        
        Args:
            filepath: Path to the fixtures CSV file
            
        Returns:
            DataFrame containing fixtures data
        """
        try:
            # Ensure team_ids are read as strings first, then convert
            df = pd.read_csv(filepath, dtype={'home_team_id': str, 'away_team_id': str})
            
            # Validate required columns
            required_columns = ['match_id', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in fixtures file: {missing_columns}")
                return pd.DataFrame()
            
            # Convert team_ids to numeric
            df['home_team_id'] = pd.to_numeric(df['home_team_id'], errors='coerce')
            df['away_team_id'] = pd.to_numeric(df['away_team_id'], errors='coerce')
            
            # Drop rows with invalid team_ids
            invalid_rows = (df['home_team_id'].isna() | df['away_team_id'].isna()).sum()
            if invalid_rows > 0:
                logger.warning(f"Dropped {invalid_rows} rows with invalid team_ids from {filepath}")
                df = df.dropna(subset=['home_team_id', 'away_team_id'])
            
            # Convert date column if present
            if 'date' in df.columns:
                try:
                    # Try multiple date formats
                    for date_format in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            df['date'] = pd.to_datetime(df['date'], format=date_format, errors='raise')
                            break
                        except ValueError:
                            continue
                    else:
                        # If none of the formats worked, use the default parser
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                except Exception as e:
                    logger.warning(f"Error converting date column: {e}")
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            logger.info(f"Loaded {len(df)} fixtures from {filepath}")
            return df
        except pd.errors.EmptyDataError:
            logger.info(f"Fixtures file is empty: {filepath}")
            return pd.DataFrame()
        except FileNotFoundError:
            logger.warning(f"Fixtures file not found: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load fixtures from {filepath}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def extract_league_id(self, filename):
        """
        Extract league ID from a filename: league_{league_id}_...csv
        
        Args:
            filename: Filename to extract league ID from
            
        Returns:
            League ID as integer or None if not found
        """
        if filename.startswith('league_') and filename.endswith('.csv'):
            parts = filename.split('_')
            if len(parts) > 1:
                try:
                    return int(parts[1])
                except ValueError:
                    logger.warning(f"Could not parse league_id from {parts[1]} in {filename}")
                    pass
        logger.warning(f"Cannot extract league_id from filename: {filename}")
        return None

    def process_league(self, league_id, high_form_file_path, fixtures_file_path, config=None):
        """
        Process a single league to find flagged matches with high win potential.
        
        Args:
            league_id: League ID
            high_form_file_path: Path to high-form teams CSV file
            fixtures_file_path: Path to fixtures CSV file
            config: Configuration dictionary with sentiment analysis settings
            
        Returns:
            DataFrame containing flagged matches
        """
        high_form_teams = self.load_high_form_teams(high_form_file_path)
        if high_form_teams.empty:
            logger.info(f"No high-form teams data available for league ID {league_id} from {high_form_file_path}, skipping processing for this league.")
            return pd.DataFrame()
        
        fixtures = self.load_fixtures(fixtures_file_path)
        if fixtures.empty:
            logger.info(f"No fixtures data available for league ID {league_id} from {fixtures_file_path}, skipping processing for this league.")
            return pd.DataFrame()
        
        return self.process_league_from_dataframes(league_id, high_form_teams, fixtures, config)
    
    def process_league_from_dataframes(self, league_id, high_form_teams, fixtures, config=None):
        """
        Process a single league to find flagged matches with high win potential using DataFrames directly.
        
        Args:
            league_id: League ID
            high_form_teams: DataFrame containing high-form teams data
            fixtures: DataFrame containing fixtures data
            config: Configuration dictionary with sentiment analysis settings
            
        Returns:
            DataFrame containing flagged matches
        """
        if high_form_teams.empty:
            logger.info(f"No high-form teams data available for league ID {league_id}, skipping processing for this league.")
            return pd.DataFrame()
        
        if fixtures.empty:
            logger.info(f"No fixtures data available for league ID {league_id}, skipping processing for this league.")
            return pd.DataFrame()
        
        # Create a dictionary for win rates: team_id -> win_rate
        # Filter out NaN team_ids from high_form_teams before setting index
        high_form_teams_cleaned = high_form_teams.dropna(subset=['team_id'])
        if 'team_id' not in high_form_teams_cleaned.columns or 'win_rate' not in high_form_teams_cleaned.columns:
            logger.error(f"High form teams data for league {league_id} is missing 'team_id' or 'win_rate' columns.")
            return pd.DataFrame()

        win_rate_dict = high_form_teams_cleaned.set_index('team_id')['win_rate'].to_dict()
        
        # Add league information to fixtures (useful for combined output)
        league_info = self.leagues_data.get(league_id, {'country': 'Unknown', 'name': 'Unknown'})
        fixtures['league_id'] = league_id
        fixtures['league_name'] = league_info['name']
        fixtures['country'] = league_info['country']

        # Map win rates to home and away teams in fixtures
        fixtures['home_win_rate'] = fixtures['home_team_id'].map(win_rate_dict)
        fixtures['away_win_rate'] = fixtures['away_team_id'].map(win_rate_dict)
        
        # Filter for matches where at least one team has a win rate (i.e., is in high_form_teams)
        flagged_fixtures = fixtures[
            fixtures['home_win_rate'].notnull() | fixtures['away_win_rate'].notnull()
        ].copy()
        
        # Add sentiment analysis if enabled in config
        if config and config.get('sentiment_analysis', {}).get('enabled', False):
            logger.info(f"Performing sentiment analysis for {len(flagged_fixtures)} flagged fixtures in league {league_id}")
            
            # Initialize sentiment columns
            flagged_fixtures['home_team_sentiment'] = None
            flagged_fixtures['away_team_sentiment'] = None
            flagged_fixtures['sentiment_articles_analyzed'] = 0
            
            for idx, match in flagged_fixtures.iterrows():
                home_team = match['home_team_name']
                away_team = match['away_team_name']
                match_date = match.get('date')
                
                # Get sentiment scores for this match
                home_sentiment, away_sentiment, articles_count = self.get_sentiment_for_match(
                    home_team, away_team, match_date, config
                )
                
                # Update DataFrame with sentiment data
                flagged_fixtures.at[idx, 'home_team_sentiment'] = home_sentiment
                flagged_fixtures.at[idx, 'away_team_sentiment'] = away_sentiment
                flagged_fixtures.at[idx, 'sentiment_articles_analyzed'] = articles_count
        
        # Deduplicate based on match characteristics to prevent duplicates with different match_ids
        # Create a unique identifier based on league, teams, and date
        # Ensure date is properly formatted as datetime before using .dt accessor
        if 'date' in flagged_fixtures.columns:
            flagged_fixtures['date'] = pd.to_datetime(flagged_fixtures['date'], errors='coerce')
            
        flagged_fixtures['match_identifier'] = (
            flagged_fixtures['league_id'].astype(str) + '_' +
            flagged_fixtures['home_team_name'].str.lower().str.strip() + '_' +
            flagged_fixtures['away_team_name'].str.lower().str.strip() + '_' +
            flagged_fixtures['date'].dt.strftime('%Y-%m-%d')
        )
        
        # Keep the first occurrence of each unique match (based on our identifier)
        flagged_fixtures = flagged_fixtures.drop_duplicates(subset=['match_identifier'], keep='first')
        
        # Drop the temporary identifier column
        flagged_fixtures = flagged_fixtures.drop(columns=['match_identifier'])
        
        # Select and order output columns
        output_columns = [
            'league_id', 'league_name', 'country', 'match_id', 'date', 'time', 
            'home_team_name', 'home_win_rate', 'home_team_sentiment',
            'away_team_name', 'away_win_rate', 'away_team_sentiment',
            'sentiment_articles_analyzed'
        ]
        
        # Ensure all output columns exist in flagged_fixtures, add if missing
        for col in output_columns:
            if col not in flagged_fixtures.columns:
                flagged_fixtures[col] = None

        return flagged_fixtures[output_columns]

    def save_predictions_to_db(self, flagged_matches, config=None):
        """
        Save predictions to the database.
        
        Args:
            flagged_matches: DataFrame containing flagged matches
            config: Configuration dictionary with sentiment analysis settings
            
        Returns:
            Number of predictions saved
        """
        if not self.db_enabled:
            logger.info("Database operations disabled, skipping database save.")
            return 0
            
        if flagged_matches.empty:
            logger.info("No flagged matches to save to database.")
            return 0
            
        # Get sentiment analysis configuration
        sentiment_enabled = False
        weight_form = 0.7
        weight_sentiment = 0.3
        
        if config and config.get('sentiment_analysis', {}).get('enabled', False):
            sentiment_enabled = True
            weight_form = config.get('sentiment_analysis', {}).get('weight_form', 0.7)
            weight_sentiment = config.get('sentiment_analysis', {}).get('weight_sentiment', 0.3)
            logger.info(f"Sentiment analysis enabled for predictions with weights: form={weight_form}, sentiment={weight_sentiment}")
            
        try:
            session = get_db_session()
            predictions_saved = 0
            
            for _, match in flagged_matches.iterrows():
                # Get fixture from database
                fixture = None
                if 'match_id' in match and not pd.isna(match['match_id']):
                    fixture = session.query(Fixture).filter(Fixture.id == match['match_id']).first()
                
                if fixture is None and 'home_team_id' in match and 'away_team_id' in match:
                    # Try to find by team IDs and date
                    query = session.query(Fixture).filter(
                        Fixture.home_team_id == match['home_team_id'],
                        Fixture.away_team_id == match['away_team_id']
                    )
                    
                    if 'date' in match and not pd.isna(match['date']):
                        # If date is a string, convert to datetime
                        match_date = match['date']
                        if isinstance(match_date, str):
                            try:
                                match_date = pd.to_datetime(match_date)
                            except:
                                pass
                                
                        if isinstance(match_date, pd.Timestamp):
                            # Find fixtures on the same day
                            start_date = match_date.replace(hour=0, minute=0, second=0)
                            end_date = match_date.replace(hour=23, minute=59, second=59)
                            query = query.filter(Fixture.match_date.between(start_date, end_date))
                    
                    fixture = query.first()
                
                if fixture is None:
                    logger.warning(f"Could not find fixture for match: {match['home_team_name']} vs {match['away_team_name']}")
                    continue
                
                # Get form scores (win rates)
                home_win_rate = match.get('home_win_rate', 0)
                away_win_rate = match.get('away_win_rate', 0)
                
                if pd.isna(home_win_rate):
                    home_win_rate = 0
                if pd.isna(away_win_rate):
                    away_win_rate = 0
                
                # Get sentiment scores if available
                home_sentiment = match.get('home_team_sentiment', None)
                away_sentiment = match.get('away_team_sentiment', None)
                sentiment_articles = match.get('sentiment_articles_analyzed', 0)
                
                # Normalize sentiment scores to 0-1 range if they exist
                normalized_home_sentiment = None
                normalized_away_sentiment = None
                
                if sentiment_enabled and home_sentiment is not None and away_sentiment is not None:
                    # Convert from -1 to 1 scale to 0 to 1 scale
                    normalized_home_sentiment = (home_sentiment + 1) / 2
                    normalized_away_sentiment = (away_sentiment + 1) / 2
                    
                    logger.debug(f"Normalized sentiment scores: {match['home_team_name']}={normalized_home_sentiment:.2f}, "
                               f"{match['away_team_name']}={normalized_away_sentiment:.2f}")
                
                # Calculate goal predictions based on form and sentiment
                # Base prediction on team form (win rates correlate with goal scoring)
                home_expected_goals = home_win_rate * 2.0  # Scale win rate to expected goals
                away_expected_goals = away_win_rate * 2.0  # Scale win rate to expected goals
                
                # Apply sentiment adjustment if available
                if sentiment_enabled and normalized_home_sentiment is not None and normalized_away_sentiment is not None:
                    # Sentiment affects goal scoring ability (positive sentiment = more goals)
                    home_sentiment_boost = (normalized_home_sentiment - 0.5) * 0.5  # -0.25 to +0.25 goals
                    away_sentiment_boost = (normalized_away_sentiment - 0.5) * 0.5  # -0.25 to +0.25 goals
                    
                    home_expected_goals += home_sentiment_boost
                    away_expected_goals += away_sentiment_boost
                    
                    logger.debug(f"Sentiment-adjusted expected goals: home={home_expected_goals:.2f}, away={away_expected_goals:.2f}")
                
                # Ensure minimum expected goals
                home_expected_goals = max(0.5, home_expected_goals)
                away_expected_goals = max(0.5, away_expected_goals)
                
                # Calculate total expected goals
                predicted_total_goals = home_expected_goals + away_expected_goals
                
                # Calculate over 2.5 probability using Poisson distribution approximation
                # For simplicity, use a sigmoid function based on total goals
                import math
                over_2_5_probability = 1 / (1 + math.exp(-(predicted_total_goals - 2.5) * 2))
                
                logger.debug(f"Goal predictions: total={predicted_total_goals:.2f}, over2.5={over_2_5_probability:.2f}")
                
                # Check if prediction already exists
                existing_prediction = session.query(Prediction).filter(
                    Prediction.fixture_id == fixture.id
                ).first()
                
                # Determine which features were used
                features_used = ["recent_win_rate"]
                algorithm_version = "high_form_team_v1"
                
                if sentiment_enabled and normalized_home_sentiment is not None and normalized_away_sentiment is not None:
                    features_used.append("sentiment_analysis")
                    algorithm_version = "high_form_team_v2_with_sentiment"
                
                if existing_prediction:
                    # Update existing prediction with goal-based model
                    existing_prediction.predicted_total_goals = predicted_total_goals
                    existing_prediction.over_2_5_probability = over_2_5_probability
                    existing_prediction.home_team_form_score = home_win_rate
                    existing_prediction.away_team_form_score = away_win_rate
                    existing_prediction.confidence_score = over_2_5_probability  # Use over 2.5 probability as confidence
                    existing_prediction.algorithm_version = algorithm_version
                    existing_prediction.features_used = json.dumps(features_used)
                    existing_prediction.prediction_date = datetime.now(timezone.utc)
                    existing_prediction.updated_at = datetime.now(timezone.utc)
                    
                    # Add sentiment data if available
                    if sentiment_enabled and home_sentiment is not None and away_sentiment is not None:
                        existing_prediction.home_team_sentiment = home_sentiment
                        existing_prediction.away_team_sentiment = away_sentiment
                        existing_prediction.sentiment_articles_analyzed = sentiment_articles
                    
                    logger.debug(f"Updated goal prediction for fixture {fixture.id}: {fixture.home_team.name} vs {fixture.away_team.name} - Total: {predicted_total_goals:.2f}, Over 2.5: {over_2_5_probability:.2f}")
                else:
                    # Create new prediction with goal-based model
                    new_prediction = Prediction(
                        fixture_id=fixture.id,
                        predicted_total_goals=predicted_total_goals,
                        over_2_5_probability=over_2_5_probability,
                        home_team_form_score=home_win_rate,
                        away_team_form_score=away_win_rate,
                        confidence_score=over_2_5_probability,  # Use over 2.5 probability as confidence
                        algorithm_version=algorithm_version,
                        features_used=json.dumps(features_used),
                        prediction_date=datetime.now(timezone.utc)
                    )
                    
                    # Add sentiment data if available
                    if sentiment_enabled and home_sentiment is not None and away_sentiment is not None:
                        new_prediction.home_team_sentiment = home_sentiment
                        new_prediction.away_team_sentiment = away_sentiment
                        new_prediction.sentiment_articles_analyzed = sentiment_articles
                    
                    session.add(new_prediction)
                    logger.debug(f"Created new goal prediction for fixture {fixture.id}: {fixture.home_team.name} vs {fixture.away_team.name} - Total: {predicted_total_goals:.2f}, Over 2.5: {over_2_5_probability:.2f}")
                
                predictions_saved += 1
            
            # Commit all changes
            if predictions_saved > 0:
                session.commit()
                logger.info(f"Saved {predictions_saved} predictions to database.")
            
            session.close()
            return predictions_saved
        except Exception as e:
            logger.error(f"Error saving predictions to database: {e}")
            logger.debug(traceback.format_exc())
            try:
                session.rollback()
            except:
                pass
            return 0

    def run_predictor_outputter(self):
        """
        Run the Predictor/Outputter to generate daily predictions.
        
        Returns:
            Dictionary with processing statistics
        """
        # Reset statistics
        self.stats = {
            'file_predictions': 0,
            'db_predictions': 0,
            'leagues_processed': 0,
            'errors': 0
        }
        
        # Prioritize database processing over file processing
        if self.db_enabled:
            logger.info("Checking database for fixtures...")
            
            try:
                # Check if we have fixtures in database
                session = get_db_session()
                fixture_count = session.query(Fixture).filter(
                    Fixture.match_date > datetime.now(timezone.utc),
                    Fixture.match_date < datetime.now(timezone.utc) + pd.Timedelta(days=7)
                ).count()
                session.close()
                
                if fixture_count > 0:
                    logger.info(f"Found {fixture_count} upcoming fixtures in database - using database mode")
                    return self._process_database_only()
                else:
                    logger.info("No upcoming fixtures found in database")
            except Exception as e:
                logger.error(f"Error checking database fixtures: {e}")
                logger.info("Falling back to file processing...")
        
        # Process file-based data only if database has no fixtures or database is disabled
        if self.file_enabled:
            # Check if directories exist
            if not os.path.exists(self.fixtures_dir):
                logger.error(f"Fixtures directory {self.fixtures_dir} does not exist. Cannot run predictor.")
                self.stats['errors'] += 1
                return self.stats
                
            if not os.path.exists(self.processed_dir):
                logger.error(f"Processed data directory {self.processed_dir} does not exist. Cannot run predictor.")
                self.stats['errors'] += 1
                return self.stats
            
            # Expecting filenames like: league_{id}_upcoming_fixtures.csv
            fixture_filenames = [f for f in os.listdir(self.fixtures_dir) if f.startswith('league_') and f.endswith('_upcoming_fixtures.csv')]
            logger.info(f"Found {len(fixture_filenames)} fixture files to process in {self.fixtures_dir}")
            
            if not fixture_filenames:
                logger.warning("No fixture files found in fixtures directory")
                logger.info("No data to process")
                return self.stats
            
            all_flagged_matches_list = []
            
            for fixture_filename in fixture_filenames:
                league_id = self.extract_league_id(fixture_filename)
                if league_id is None:
                    logger.warning(f"Could not extract league_id from {fixture_filename}, skipping.")
                    continue
                
                # Construct paths to input files for this league
                high_form_file_path = os.path.join(self.processed_dir, f"league_{league_id}_high_form_teams.csv")
                fixtures_file_path = os.path.join(self.fixtures_dir, fixture_filename)
                
                logger.info(f"Processing league ID {league_id}: using high-form file '{high_form_file_path}' and fixtures file '{fixtures_file_path}'")
                
                try:
                    flagged_league_df = self.process_league(league_id, high_form_file_path, fixtures_file_path, self.config)
                    
                    if not flagged_league_df.empty:
                        all_flagged_matches_list.append(flagged_league_df)
                        
                        # Save to database if enabled
                        if self.db_enabled:
                            db_predictions = self.save_predictions_to_db(flagged_league_df, self.config)
                            self.stats['db_predictions'] += db_predictions
                    else:
                        logger.info(f"No flagged matches found for league ID {league_id}.")
                    
                    self.stats['leagues_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing league ID {league_id}: {e}")
                    logger.debug(traceback.format_exc())
                    self.stats['errors'] += 1
            
            # Combine and save all flagged matches to a single CSV file
            if all_flagged_matches_list:
                combined_flagged_matches = pd.concat(all_flagged_matches_list, ignore_index=True)
                
                # Convert 'date' to datetime for sorting, handle errors
                if 'date' in combined_flagged_matches.columns:
                    try:
                        combined_flagged_matches['date_dt'] = pd.to_datetime(combined_flagged_matches['date'], errors='coerce')
                        
                        # Sort by date (and time, if time column is consistently formatted)
                        if 'time' in combined_flagged_matches.columns and 'date_dt' in combined_flagged_matches.columns:
                            combined_flagged_matches = combined_flagged_matches.sort_values(by=['date_dt', 'time'])
                        elif 'date_dt' in combined_flagged_matches.columns:
                            combined_flagged_matches = combined_flagged_matches.sort_values(by=['date_dt'])
                        
                        # Remove temporary sort column
                        if 'date_dt' in combined_flagged_matches.columns:
                            combined_flagged_matches.drop(columns=['date_dt'], inplace=True)
                    except Exception as e:
                        logger.warning(f"Error sorting by date: {e}")
                
                # Save to a timestamped CSV
                date_str = datetime.now().strftime("%Y%m%d")
                output_filename = os.path.join(self.output_dir, f"predictions_{date_str}.csv")
                try:
                    combined_flagged_matches.to_csv(output_filename, index=False)
                    logger.info(f"Saved {len(combined_flagged_matches)} total flagged matches to {output_filename}")
                    self.stats['file_predictions'] = len(combined_flagged_matches)
                except Exception as e:
                    logger.error(f"Error saving combined predictions to {output_filename}: {e}")
                    logger.debug(traceback.format_exc())
                    self.stats['errors'] += 1
            else:
                logger.info("No flagged matches found across all leagues.")
        
        # Process database-only predictions as final fallback
        elif self.db_enabled:
            logger.info("File processing disabled, using database-only mode.")
            return self._process_database_only()

    def _process_database_only(self):
        """
        Process predictions using database data only.
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            # Get database session
            session = get_db_session()
            
            # Get all leagues from database
            leagues = session.query(League).all()
            logger.info(f"Found {len(leagues)} leagues in database")
            
            all_flagged_matches_list = []
            
            for league in leagues:
                logger.info(f"Processing league ID {league.id}: {league.name}")
                
                try:
                    # Define current time for both queries
                    now = datetime.now(timezone.utc)
                    
                    # Get high-form teams from the high_form_teams table
                    # This data is now pre-calculated by standalone_data_processor.py
                    high_form_teams_query = session.query(
                        HighFormTeam.team_id,
                        Team.name.label('team_name'),
                        HighFormTeam.win_rate,
                        HighFormTeam.total_matches,
                        HighFormTeam.wins
                    ).join(
                        Team, HighFormTeam.team_id == Team.id
                    ).filter(
                        HighFormTeam.league_id == league.id,
                        # Get only recent analyses (last 24 hours to ensure fresh data)
                        HighFormTeam.analysis_date >= now - timedelta(hours=24)
                    )
                    
                    high_form_teams_results = high_form_teams_query.all()
                    
                    # If no high-form teams found in the table, fall back to calculating them on the fly
                    # This ensures backward compatibility during transition
                    if not high_form_teams_results:
                        logger.info(f"No pre-calculated high-form teams found for league ID {league.id}, calculating on the fly.")
                        # Calculate win rates based on recent fixtures (last 30 days)
                        recent_date_cutoff = now - timedelta(days=30)  # Last 30 days
                        teams_query = session.query(
                            Team.id,
                            Team.name,
                            func.count(Fixture.id).label('total_matches'),
                            func.sum(case(
                                (and_(Fixture.home_team_id == Team.id, Fixture.home_score > Fixture.away_score), 1),
                                (and_(Fixture.away_team_id == Team.id, Fixture.away_score > Fixture.home_score), 1),
                                else_=0
                            )).label('wins')
                        ).filter(
                            Team.league_id == league.id
                        ).outerjoin(
                            Fixture, or_(
                                Fixture.home_team_id == Team.id,
                                Fixture.away_team_id == Team.id
                            )
                        ).filter(
                            Fixture.match_date >= recent_date_cutoff,
                            Fixture.match_date <= now,
                            Fixture.status == 'finished'  # Only count completed matches
                        ).group_by(Team.id)
                        
                        teams_with_stats = teams_query.all()
                        
                        # Calculate win rates and filter for high-form teams
                        high_form_teams = []
                        for team in teams_with_stats:
                            if team.total_matches > 0:
                                win_rate = team.wins / team.total_matches
                                if win_rate >= 0.6:  # Consider teams with 60%+ win rate as high-form
                                    high_form_teams.append({
                                        'team_id': team.id,
                                        'team_name': team.name,
                                        'win_rate': win_rate,
                                        'total_matches': team.total_matches,
                                        'wins': team.wins
                                    })
                    else:
                        # Convert query results to the expected dictionary format
                        high_form_teams = []
                        for team in high_form_teams_results:
                            high_form_teams.append({
                                'team_id': team.team_id,
                                'team_name': team.team_name,
                                'win_rate': team.win_rate,
                                'total_matches': team.total_matches,
                                'wins': team.wins
                            })
                        logger.info(f"Found {len(high_form_teams)} pre-calculated high-form teams for league ID {league.id}")
                    
                    # Get upcoming fixtures
                    fixtures_query = session.query(Fixture).filter(
                        Fixture.league_id == league.id,
                        Fixture.match_date > now,
                        Fixture.match_date < now + timedelta(days=7)  # Next 7 days
                    ).all()
                    
                    # Skip if no high-form teams or fixtures
                    if not high_form_teams or not fixtures_query:
                        logger.info(f"No high-form teams or upcoming fixtures for league ID {league.id}, skipping.")
                        continue
                    
                    # Convert to DataFrames
                    high_form_df = pd.DataFrame(high_form_teams)
                    
                    fixtures_data = []
                    for fixture in fixtures_query:
                        fixtures_data.append({
                            'match_id': fixture.id,
                            'date': fixture.match_date.strftime('%Y-%m-%d'),
                            'time': fixture.match_date.strftime('%H:%M'),
                            'home_team_id': fixture.home_team_id,
                            'away_team_id': fixture.away_team_id,
                            'home_team_name': fixture.home_team.name if fixture.home_team else 'Unknown',
                            'away_team_name': fixture.away_team.name if fixture.away_team else 'Unknown'
                        })
                    
                    fixtures_df = pd.DataFrame(fixtures_data)
                    
                    # Process league using the DataFrames
                    flagged_league_df = self.process_league_from_dataframes(
                        league.id, high_form_df, fixtures_df, self.config
                    )
                    
                    if not flagged_league_df.empty:
                        all_flagged_matches_list.append(flagged_league_df)
                        
                        # Save to database
                        db_predictions = self.save_predictions_to_db(flagged_league_df, self.config)
                        self.stats['db_predictions'] += db_predictions
                    else:
                        logger.info(f"No flagged matches found for league ID {league.id}.")
                    
                    self.stats['leagues_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing league ID {league.id}: {e}")
                    logger.debug(traceback.format_exc())
                    self.stats['errors'] += 1
            
            # Combine and save all flagged matches to a single CSV file if output is enabled
            if all_flagged_matches_list and self.output_dir:
                combined_flagged_matches = pd.concat(all_flagged_matches_list, ignore_index=True)
                
                # Remove duplicates based on match characteristics (league, teams, date)
                if not combined_flagged_matches.empty:
                    # Create a unique identifier based on match characteristics
                    combined_flagged_matches['match_identifier'] = (
                        combined_flagged_matches['league_id'].astype(str) + '_' +
                        combined_flagged_matches['home_team_name'].str.lower().str.replace(' ', '_') + '_' +
                        combined_flagged_matches['away_team_name'].str.lower().str.replace(' ', '_') + '_' +
                        combined_flagged_matches['date'].astype(str)
                    )
                    
                    # Keep only the first occurrence of each unique match
                    initial_count = len(combined_flagged_matches)
                    combined_flagged_matches = combined_flagged_matches.drop_duplicates(
                        subset=['match_identifier'], keep='first'
                    )
                    final_count = len(combined_flagged_matches)
                    
                    if initial_count != final_count:
                        logger.info(f"Removed {initial_count - final_count} duplicate matches from combined results")
                    
                    # Clean up the temporary identifier column
                    combined_flagged_matches = combined_flagged_matches.drop(columns=['match_identifier'])
                
                # Convert 'date' to datetime for sorting, handle errors
                if 'date' in combined_flagged_matches.columns:
                    try:
                        combined_flagged_matches['date_dt'] = pd.to_datetime(combined_flagged_matches['date'], errors='coerce')
                        
                        # Sort by date (and time, if time column is consistently formatted)
                        if 'time' in combined_flagged_matches.columns and 'date_dt' in combined_flagged_matches.columns:
                            combined_flagged_matches = combined_flagged_matches.sort_values(by=['date_dt', 'time'])
                        elif 'date_dt' in combined_flagged_matches.columns:
                            combined_flagged_matches = combined_flagged_matches.sort_values(by=['date_dt'])
                        
                        # Remove temporary sort column
                        if 'date_dt' in combined_flagged_matches.columns:
                            combined_flagged_matches.drop(columns=['date_dt'], inplace=True)
                    except Exception as e:
                        logger.warning(f"Error sorting by date: {e}")
                
                # Save to a timestamped CSV
                date_str = datetime.now().strftime("%Y%m%d")
                output_filename = os.path.join(self.output_dir, f"predictions_{date_str}.csv")
                try:
                    combined_flagged_matches.to_csv(output_filename, index=False)
                    logger.info(f"Saved {len(combined_flagged_matches)} total flagged matches to {output_filename}")
                    self.stats['file_predictions'] = len(combined_flagged_matches)
                except Exception as e:
                    logger.error(f"Error saving combined predictions to {output_filename}: {e}")
                    logger.debug(traceback.format_exc())
                    self.stats['errors'] += 1
            else:
                logger.info("No flagged matches found across all leagues.")
            
            session.close()
            return self.stats
        except Exception as e:
            logger.error(f"Error in database-only mode: {e}")
            logger.debug(traceback.format_exc())
            self.stats['errors'] += 1
            session.close()
            return self.stats


def load_config(config_path=None):
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
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
                    logger.warning(f"Unsupported config file format: {config_path}. Using default configuration.")
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
        # Try to load from default location
        try:
            default_config_path = 'sdata_init_config.json'
            if os.path.exists(default_config_path):
                with open(default_config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Update config with user values
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                        # Merge nested dictionaries
                        config[key].update(value)
                    else:
                        config[key] = value
                        
                logger.info(f"Loaded configuration from default location: {default_config_path}")
        except Exception as e:
            logger.warning(f"Could not load configuration from default location: {e}")
    
    return config


def main():
    """
    Main entry point for the standalone predictor outputter.
    """
    parser = argparse.ArgumentParser(description="Standalone Predictor Outputter for FormFinder")
    parser.add_argument('--config', help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--db-only', action='store_true', help='Process database data only')
    parser.add_argument('--file-only', action='store_true', help='Process file data only')
    parser.add_argument('--output-dir', help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.db_only:
        config['file_processing']['enabled'] = False
    if args.file_only:
        config['database']['enabled'] = False
    if args.output_dir:
        config['output_directory'] = args.output_dir
    
    # Initialize database
    try:
        init_database()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1
    
    try:
        # Create and run predictor outputter
        predictor = PredictorOutputter(config)
        stats = predictor.run_predictor_outputter()
        
        # Print summary
        print("\nPrediction Generation Summary:")
        print(f"Leagues processed: {stats['leagues_processed']}")
        print(f"File predictions generated: {stats['file_predictions']}")
        print(f"Database predictions saved: {stats['db_predictions']}")
        print(f"Errors encountered: {stats['errors']}")
        
        # Save summary to file
        summary_path = os.path.join(predictor.output_dir, f"prediction_summary_{datetime.now().strftime('%Y%m%d')}.json")
        try:
            with open(summary_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'stats': stats
                }, f, indent=2)
            logger.info(f"Saved prediction summary to {summary_path}")
        except Exception as e:
            logger.error(f"Error saving prediction summary: {e}")
        
        return 0 if stats['errors'] == 0 else 1
    except Exception as e:
        logger.error(f"Error running predictor outputter: {e}")
        logger.debug(traceback.format_exc())
        return 1
    finally:
        # Close database connection
        close_database()


if __name__ == "__main__":
    sys.exit(main())