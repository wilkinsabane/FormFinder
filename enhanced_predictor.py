"""Enhanced goal prediction with quantile regression and Poisson baseline.

This module provides advanced football goal prediction capabilities using:
- Quantile regression for uncertainty estimation
- Poisson baseline models for goal distribution modeling
- Comprehensive feature engineering
- Model calibration and monitoring
- Rich logging and error handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
import warnings
import sys
import os
import time
import traceback
import json
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any, Union

# Scientific computing and ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.exceptions import ConvergenceWarning
import joblib
from scipy import stats

# Database
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, DatabaseError

# Rich imports for beautiful console output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Configure rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path to import formfinder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from formfinder.config import load_config, get_config
    from scripts.calibration_monitor import CalibrationMonitor
    from formfinder.markov_feature_generator import MarkovFeatureGenerator
    from formfinder.sentiment import SentimentAnalyzer
    logger.info("📦 Enhanced predictor modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import required modules: {e}")
    sys.exit(1)

@contextmanager
def timer(description: str):
    """Context manager for timing operations with rich output."""
    start_time = time.time()
    logger.info(f"⏱️ {description}...")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"✅ {description} completed in {elapsed:.2f}s")

class EnhancedGoalPredictor:
    """Enhanced predictor with quantile regression and calibration.
    
    This class provides advanced goal prediction capabilities including:
    - Multiple model types (Poisson, Gradient Boosting)
    - Quantile regression for uncertainty estimation
    - Comprehensive feature engineering
    - Model calibration and monitoring
    - Rich logging and error handling
    
    Attributes:
        config: Configuration object
        engine: SQLAlchemy database engine
        db_session: Database session
        models: Dictionary of trained models
        scalers: Dictionary of feature scalers
        feature_importance: Feature importance scores
        calibration_monitor: Calibration monitoring instance
        quantiles: List of quantiles for regression
        model_configs: Model configuration parameters
    """
    
    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the Enhanced Goal Predictor.
        
        Args:
            config: Configuration object. If None, loads from default config.
            
        Raises:
            SystemExit: If critical initialization errors occur
        """
        start_time = time.time()
        
        try:
            # Display initialization banner
            rprint(Panel.fit(
                "[bold blue]🎯 Enhanced Goal Predictor[/bold blue]\n"
                "[dim]Advanced ML for football goal prediction[/dim]",
                border_style="blue"
            ))
            
            # Load configuration
            with timer("⚙️ Loading configuration"):
                if config is None:
                    load_config()
                    config = get_config()
                    logger.info("📋 Configuration loaded from default source")
                else:
                    logger.info("📋 Using provided configuration")
                
                self.config = config
                logger.info(f"🔧 Database URL: {config.get_database_url()[:50]}...")
            
            # Initialize database connection
            with timer("🗄️ Establishing database connection"):
                try:
                    self.engine = create_engine(
                        config.get_database_url(),
                        pool_pre_ping=True,  # Verify connections before use
                        pool_recycle=3600,   # Recycle connections every hour
                        echo=False           # Set to True for SQL debugging
                    )
                    Session = sessionmaker(bind=self.engine)
                    self.db_session = Session()
                    
                    # Test connection
                    self.db_session.execute(text("SELECT 1"))
                    logger.info("✅ Database connection established successfully")
                    
                except SQLAlchemyError as e:
                    logger.error(f"❌ Database connection failed: {e}")
                    raise SystemExit(1)
                except Exception as e:
                    logger.error(f"❌ Unexpected database error: {e}")
                    raise SystemExit(1)
            
            # Initialize model storage
            with timer("🧠 Initializing model storage"):
                self.models: Dict[str, Any] = {}
                self.scalers: Dict[str, StandardScaler] = {}
                self.feature_importance: Dict[str, Dict[str, float]] = {}
                logger.info("📊 Model storage initialized")
            
            # Initialize calibration monitor
            with timer("📈 Setting up calibration monitoring"):
                try:
                    self.calibration_monitor = CalibrationMonitor(config)
                    logger.info("✅ Calibration monitor initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Calibration monitor initialization failed: {e}")
                    self.calibration_monitor = None
            
            # Initialize Markov feature generator
            with timer("🔗 Setting up Markov feature generator"):
                try:
                    self.markov_generator = MarkovFeatureGenerator(db_session=self.db_session, lookback_window=10)
                    logger.info("✅ Markov feature generator initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Markov feature generator initialization failed: {e}")
                    self.markov_generator = None
            
            # Initialize sentiment analyzer
            with timer("💭 Setting up sentiment analyzer"):
                try:
                    # Initialize SentimentAnalyzer - it will use all configured providers
                    # including NewsData.io and TheNewsAPI, not just NewsAPI
                    self.sentiment_analyzer = SentimentAnalyzer()
                    logger.info("✅ Sentiment analyzer initialized with multiple providers")
                except Exception as e:
                    logger.warning(f"⚠️ Sentiment analyzer initialization failed: {e}")
                    self.sentiment_analyzer = None
            
            # Define quantiles for uncertainty estimation
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            logger.info(f"📊 Using quantiles: {self.quantiles}")
            
            # Model configurations with enhanced parameters
            self.model_configs = {
                'poisson_baseline': {
                    'model_class': PoissonRegressor,
                    'params': {
                        'alpha': 0.1,
                        'max_iter': 1000,
                        'tol': 1e-4,
                        'fit_intercept': True
                    },
                    'description': 'Poisson regression for goal count modeling'
                },
                'quantile_regression': {
                    'model_class': GradientBoostingRegressor,
                    'params': {
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'random_state': 42,
                        'min_samples_split': 10,
                        'min_samples_leaf': 5,
                        'max_features': 'sqrt'
                    },
                    'description': 'Gradient boosting for quantile regression'
                }
            }
            
            # Log model configurations
            config_table = Table(title="🎯 Model Configurations", border_style="cyan")
            config_table.add_column("Model Type", style="cyan")
            config_table.add_column("Description", style="white")
            config_table.add_column("Key Parameters", style="green")
            
            for model_name, config_data in self.model_configs.items():
                key_params = ', '.join([f"{k}={v}" for k, v in list(config_data['params'].items())[:3]])
                config_table.add_row(
                    model_name,
                    config_data['description'],
                    key_params + '...'
                )
            
            console.print(config_table)
            
            # Initialization complete
            init_time = time.time() - start_time
            logger.info(f"🎉 Enhanced Goal Predictor initialized successfully in {init_time:.2f}s")
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Initialization interrupted by user")
            raise SystemExit(130)
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"❌ Critical error during initialization: {e}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            raise SystemExit(1)
    
    def _get_team_name(self, team_id: int) -> str:
        """Get team name from team ID for sentiment analysis."""
        try:
            result = self.db_session.execute(
                text("SELECT name FROM teams WHERE id = :team_id"),
                {'team_id': team_id}
            ).fetchone()
            return result[0] if result else f"Team_{team_id}"
        except Exception as e:
            logger.warning(f"Failed to get team name for ID {team_id}: {e}")
            return f"Team_{team_id}"
    
    def extract_enhanced_features(self, fixture_id: int) -> Optional[Dict]:
        """Extract comprehensive features for a fixture with enhanced error handling and logging."""
        with timer(f"Extracting enhanced features for fixture {fixture_id}"):
            try:
                # Input validation
                if not isinstance(fixture_id, int) or fixture_id <= 0:
                    logger.error(f"Invalid fixture_id: {fixture_id}")
                    return None
                
                logger.info(f"Starting enhanced feature extraction for fixture {fixture_id}")
                
                # Complex SQL query for comprehensive feature extraction
                query = """
                WITH team_stats AS (
                    SELECT 
                        team_id,
                        AVG(goals_for) as avg_goals_for,
                        AVG(goals_against) as avg_goals_against,
                        COUNT(*) as games_played,
                        -- Form metrics (last 5 games)
                        AVG(CASE WHEN row_num <= 5 THEN goals_for END) as form_goals_for,
                        AVG(CASE WHEN row_num <= 5 THEN goals_against END) as form_goals_against,
                        -- Home/Away splits
                        AVG(CASE WHEN venue = 'home' THEN goals_for END) as home_goals_for,
                        AVG(CASE WHEN venue = 'away' THEN goals_for END) as away_goals_for,
                        AVG(CASE WHEN venue = 'home' THEN goals_against END) as home_goals_against,
                        AVG(CASE WHEN venue = 'away' THEN goals_against END) as away_goals_against
                    FROM (
                        SELECT 
                            team_id,
                            goals_for,
                            goals_against,
                            venue,
                            row_num
                        FROM (
                            SELECT 
                                team_id,
                                CASE WHEN team_id = home_team_id THEN home_score ELSE away_score END as goals_for,
                                CASE WHEN team_id = home_team_id THEN away_score ELSE home_score END as goals_against,
                                CASE WHEN team_id = home_team_id THEN 'home' ELSE 'away' END as venue,
                                ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY match_date DESC) as row_num
                            FROM (
                                SELECT f.id, f.home_team_id, f.away_team_id, f.home_score, f.away_score, f.match_date, f.league_id,
                                       f.home_team_id as team_id
                                FROM fixtures f 
                                WHERE f.home_score IS NOT NULL 
                                    AND f.match_date < (SELECT match_date FROM fixtures WHERE id = :fixture_id_1)
                                    AND f.league_id = (SELECT league_id FROM fixtures WHERE id = :fixture_id_2)
                                UNION ALL
                                SELECT f.id, f.home_team_id, f.away_team_id, f.home_score, f.away_score, f.match_date, f.league_id,
                                       f.away_team_id as team_id
                                FROM fixtures f 
                                WHERE f.home_score IS NOT NULL 
                                    AND f.match_date < (SELECT match_date FROM fixtures WHERE id = :fixture_id_3)
                                    AND f.league_id = (SELECT league_id FROM fixtures WHERE id = :fixture_id_4)
                            ) team_matches
                        ) numbered_matches
                        WHERE row_num <= 20  -- Last 20 games
                    ) recent_matches
                    GROUP BY team_id
                ),
                head_to_head AS (
                    SELECT 
                        AVG(home_score + away_score) as h2h_avg_goals,
                        AVG(ABS(home_score - away_score)) as h2h_avg_margin,
                        COUNT(*) as h2h_games
                    FROM fixtures f2
                    WHERE ((f2.home_team_id = (SELECT home_team_id FROM fixtures WHERE id = :fixture_id_5) 
                            AND f2.away_team_id = (SELECT away_team_id FROM fixtures WHERE id = :fixture_id_6))
                        OR (f2.home_team_id = (SELECT away_team_id FROM fixtures WHERE id = :fixture_id_7) 
                            AND f2.away_team_id = (SELECT home_team_id FROM fixtures WHERE id = :fixture_id_8)))
                        AND f2.home_score IS NOT NULL
                        AND f2.match_date < (SELECT match_date FROM fixtures WHERE id = :fixture_id_9)
                )
                SELECT 
                    f.id as fixture_id,
                    f.home_team_id,
                    f.away_team_id,
                    f.match_date,
                    f.league_id,
                    -- Home team stats
                    hs.avg_goals_for as home_avg_goals_for,
                    hs.avg_goals_against as home_avg_goals_against,
                    hs.form_goals_for as home_form_goals_for,
                    hs.form_goals_against as home_form_goals_against,
                    hs.home_goals_for as home_home_goals,
                    hs.home_goals_against as home_home_conceded,
                    -- Away team stats
            as_.avg_goals_for as away_avg_goals_for,
            as_.avg_goals_against as away_avg_goals_against,
            as_.form_goals_for as away_form_goals_for,
            as_.form_goals_against as away_form_goals_against,
            as_.away_goals_for as away_away_goals,
            as_.away_goals_against as away_away_conceded,
            -- Head to head
            h2h.h2h_avg_goals,
            h2h.h2h_avg_margin,
            h2h.h2h_games,
            -- League averages
            league.avg_goals_per_game as league_avg_goals
        FROM fixtures f
        LEFT JOIN team_stats hs ON f.home_team_id = hs.team_id
        LEFT JOIN team_stats as_ ON f.away_team_id = as_.team_id
        LEFT JOIN head_to_head h2h ON 1=1
        LEFT JOIN (
            SELECT 
                league_id,
                AVG(home_score + away_score) as avg_goals_per_game
            FROM fixtures 
            WHERE home_score IS NOT NULL
            GROUP BY league_id
        ) league ON f.league_id = league.league_id
        WHERE f.id = :fixture_id_10
        """
        
                logger.debug("Executing enhanced feature extraction query")
                
                # Execute query with comprehensive error handling
                try:
                    # Use the session connection instead of engine to maintain transaction consistency
                    result = self.db_session.execute(text(query), {
                        'fixture_id_1': fixture_id, 'fixture_id_2': fixture_id, 'fixture_id_3': fixture_id, 'fixture_id_4': fixture_id,
                        'fixture_id_5': fixture_id, 'fixture_id_6': fixture_id, 'fixture_id_7': fixture_id, 'fixture_id_8': fixture_id, 
                        'fixture_id_9': fixture_id, 'fixture_id_10': fixture_id
                    })
                    
                    # Convert result to DataFrame
                    rows = result.fetchall()
                    if not rows:
                        logger.warning(f"No data found for fixture {fixture_id}")
                        return None
                    
                    # Create DataFrame from result
                    columns = result.keys()
                    df = pd.DataFrame(rows, columns=columns)
                    logger.debug(f"Query executed successfully, returned {len(df)} rows")
                    
                except SQLAlchemyError as e:
                    logger.error(f"Database error during feature extraction for fixture {fixture_id}: {e}")
                    self.db_session.rollback()
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error during query execution for fixture {fixture_id}: {e}")
                    self.db_session.rollback()
                    return None
                
                # Validate query results
                if len(df) == 0:
                    logger.warning(f"No data found for fixture {fixture_id}")
                    return None
                    
                row = df.iloc[0]
                logger.debug(f"Processing feature calculation for fixture {fixture_id}")
                
                # Calculate derived features with error handling
                try:
                    # Get league average goals with fallback
                    league_avg = row['league_avg_goals'] or 2.5  # Default to 2.5 goals per game
                    league_half = max(league_avg / 2, 0.1)
                    
                    # Get team stats with defaults for teams with no match history
                    home_goals_for = row['home_avg_goals_for'] or league_half
                    home_goals_against = row['home_avg_goals_against'] or league_half
                    away_goals_for = row['away_avg_goals_for'] or league_half
                    away_goals_against = row['away_avg_goals_against'] or league_half
                    
                    features = {
                        # Basic attacking/defensive strength
                        'home_attack_strength': home_goals_for / league_half,
                        'home_defense_strength': home_goals_against / league_half,
                        'away_attack_strength': away_goals_for / league_half,
                        'away_defense_strength': away_goals_against / league_half,
                        
                        # Team strength features (expected by tests)
                        'home_team_strength': home_goals_for / league_half,
                        'away_team_strength': away_goals_for / league_half,
                        
                        # Form indicators
                        'home_form_diff': (row['home_form_goals_for'] or 0) - (row['home_form_goals_against'] or 0),
                        'away_form_diff': (row['away_form_goals_for'] or 0) - (row['away_form_goals_against'] or 0),
                        
                        # Team form scores (expected by tests)
                        'home_team_form_score': (row['home_form_goals_for'] or 0) - (row['home_form_goals_against'] or 0),
                        'away_team_form_score': (row['away_form_goals_for'] or 0) - (row['away_form_goals_against'] or 0),
                        
                        # Team positions - get actual positions from standings with confidence
                        'home_team_position': self._get_team_position_simple(int(row['home_team_id']), int(row['league_id']), row['match_date']),
                        'away_team_position': self._get_team_position_simple(int(row['away_team_id']), int(row['league_id']), row['match_date']),
                        
                        # Position confidence features
                        'home_position_confidence': self._get_team_position(int(row['home_team_id']), int(row['league_id']), row['match_date'])[1],
                        'away_position_confidence': self._get_team_position(int(row['away_team_id']), int(row['league_id']), row['match_date'])[1],
                        
                        # Home advantage
                        'home_advantage': (row['home_home_goals'] or 0) - (row['away_away_goals'] or 0),
                        'defensive_home_advantage': (row['away_away_conceded'] or 0) - (row['home_home_conceded'] or 0),
                        
                        # Note: Shot efficiency and possession features removed due to missing columns in fixtures table
                        
                        # Head-to-head features
                        'h2h_total_goals': row['h2h_avg_goals'] or league_avg,
                        'h2h_competitiveness': 1 / (1 + (row['h2h_avg_margin'] or 1)),
                        
                        # Expected goals (simple) - using calculated features
                        'home_xg': (home_goals_for / league_half) * (away_goals_against / league_half) * league_half,
                        'away_xg': (away_goals_for / league_half) * (home_goals_against / league_half) * league_half,
                    }
                    
                    # Add raw features for model with validation
                    raw_features = [
                        'home_avg_goals_for', 'home_avg_goals_against', 'away_avg_goals_for', 'away_avg_goals_against',
                        'league_avg_goals'
                    ]
                    
                    for feat in raw_features:
                        if row[feat] is not None and not pd.isna(row[feat]):
                            features[feat] = float(row[feat])
                        else:
                            features[feat] = 0.0
                            logger.debug(f"Missing value for feature {feat}, using default 0.0")
                    
                    features['fixture_id'] = fixture_id
                    features['league_id'] = int(row['league_id'])
                    
                    # Add proper form features using feature_precomputer
                    try:
                        from formfinder.feature_precomputer import FeaturePrecomputer
                        feature_precomputer = FeaturePrecomputer(self.db_session)
                        
                        home_team_id = int(row.get('home_team_id', 0))
                        away_team_id = int(row.get('away_team_id', 0))
                        match_date = row.get('match_date')
                        league_id = int(row['league_id'])
                        
                        if home_team_id and away_team_id and match_date:
                            # Get home team form features
                            home_form = feature_precomputer._get_team_form_features(
                                team_id=home_team_id,
                                match_date=match_date,
                                league_id=league_id,
                                venue='all'
                            )
                            
                            # Get away team form features
                            away_form = feature_precomputer._get_team_form_features(
                                team_id=away_team_id,
                                match_date=match_date,
                                league_id=league_id,
                                venue='all'
                            )
                            
                            # Get home team home-specific form
                            home_form_home = feature_precomputer._get_team_form_features(
                                team_id=home_team_id,
                                match_date=match_date,
                                league_id=league_id,
                                venue='home'
                            )
                            
                            # Get away team away-specific form
                            away_form_away = feature_precomputer._get_team_form_features(
                                team_id=away_team_id,
                                match_date=match_date,
                                league_id=league_id,
                                venue='away'
                            )
                            
                            # Map form features to expected database column names
                            features.update({
                                # Home team form features
                                'home_avg_goals_scored': home_form.get('avg_goals_scored', 0.0),
                                'home_avg_goals_conceded': home_form.get('avg_goals_conceded', 0.0),
                                'home_avg_goals_scored_home': home_form_home.get('avg_goals_scored', 0.0),
                                'home_avg_goals_conceded_home': home_form_home.get('avg_goals_conceded', 0.0),
                                'home_form_last_5_games': home_form.get('form_last_5_games', '[]'),
                                'home_wins_last_5': home_form.get('wins_last_5', 0),
                                'home_draws_last_5': home_form.get('draws_last_5', 0),
                                'home_losses_last_5': home_form.get('losses_last_5', 0),
                                'home_goals_for_last_5': home_form.get('goals_for_last_5', 0),
                                'home_goals_against_last_5': home_form.get('goals_against_last_5', 0),
                                
                                # Away team form features
                                'away_avg_goals_scored': away_form.get('avg_goals_scored', 0.0),
                                'away_avg_goals_conceded': away_form.get('avg_goals_conceded', 0.0),
                                'away_avg_goals_scored_away': away_form_away.get('avg_goals_scored', 0.0),
                                'away_avg_goals_conceded_away': away_form_away.get('avg_goals_conceded', 0.0),
                                'away_form_last_5_games': away_form.get('form_last_5_games', '[]'),
                                'away_wins_last_5': away_form.get('wins_last_5', 0),
                                'away_draws_last_5': away_form.get('draws_last_5', 0),
                                'away_losses_last_5': away_form.get('losses_last_5', 0),
                                'away_goals_for_last_5': away_form.get('goals_for_last_5', 0),
                                'away_goals_against_last_5': away_form.get('goals_against_last_5', 0),
                            })
                            
                            # Add H2H features using feature_precomputer
                            try:
                                h2h_result = feature_precomputer.compute_h2h_features(fixture_id)
                                if h2h_result.get('success', False):
                                    h2h_features = h2h_result.get('h2h_features', {})
                                    
                                    # Map H2H features to expected database column names
                                    features.update({
                                        'h2h_total_matches': h2h_features.get('h2h_overall_games', 0),
                                        'h2h_avg_goals': h2h_features.get('h2h_avg_total_goals', 0.0),
                                        'h2h_overall_home_goals': h2h_features.get('h2h_overall_home_goals', 0.0),
                                        'h2h_overall_away_goals': h2h_features.get('h2h_overall_away_goals', 0.0),
                                        'h2h_home_advantage': h2h_features.get('h2h_home_advantage', 0.0),
                                        'h2h_home_wins': h2h_features.get('h2h_team1_wins', 0),
                                        'h2h_away_wins': h2h_features.get('h2h_team2_wins', 0),
                                        'h2h_draws': h2h_features.get('h2h_draws', 0),
                                    })
                                    
                                    logger.debug(f"Added H2H features for fixture {fixture_id}")
                                else:
                                    logger.debug(f"H2H computation failed for fixture {fixture_id}")
                            except Exception as e:
                                logger.warning(f"Error generating H2H features for fixture {fixture_id}: {e}")
                            
                            logger.debug(f"Added form features for fixture {fixture_id}")
                        else:
                            logger.debug(f"Missing team IDs or match date for form features in fixture {fixture_id}")
                    except Exception as e:
                        logger.warning(f"Error generating form features for fixture {fixture_id}: {e}")
                    
                    # Add Markov chain features if available
                    if self.markov_generator:
                        try:
                            home_team_id = int(row.get('home_team_id', 0))
                            away_team_id = int(row.get('away_team_id', 0))
                            match_date = row.get('match_date')
                            
                            if home_team_id and away_team_id and match_date:
                                # Generate Markov features for both teams
                                markov_features = self.markov_generator.generate_features(
                                    home_team_id=home_team_id,
                                    away_team_id=away_team_id,
                                    match_date=match_date,
                                    league_id=int(row['league_id'])
                                )
                                
                                if markov_features:
                                    # Add Markov features with proper mapping for database columns
                                    for key, value in markov_features.items():
                                        if key == 'outcome_probabilities' and isinstance(value, dict):
                                            # Handle outcome probabilities dictionary
                                            features[f'markov_{key}'] = str(value)  # Store as JSON string
                                        elif isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                                            features[f'markov_{key}'] = float(value)
                                        elif isinstance(value, str):
                                            features[f'markov_{key}'] = value
                                        else:
                                            # Handle None or invalid values
                                            if key in ['home_current_state', 'away_current_state', 'home_expected_next_state', 'away_expected_next_state']:
                                                features[f'markov_{key}'] = 'unknown'
                                            else:
                                                features[f'markov_{key}'] = 0.0
                                    
                                    # Map specific Markov features to expected database columns
                                    # Momentum features
                                    if 'home_momentum_score' in markov_features:
                                        features['home_team_momentum'] = float(markov_features['home_momentum_score'])
                                    if 'away_momentum_score' in markov_features:
                                        features['away_team_momentum'] = float(markov_features['away_momentum_score'])
                                    
                                    logger.debug(f"Added {len(markov_features)} Markov features for fixture {fixture_id}")
                                else:
                                    logger.debug(f"No Markov features available for fixture {fixture_id}")
                            else:
                                logger.debug(f"Missing team IDs or match date for Markov features in fixture {fixture_id}")
                        except Exception as e:
                            logger.warning(f"Error generating Markov features for fixture {fixture_id}: {e}")
                    else:
                        logger.debug("Markov generator not available, skipping Markov features")
                    
                    # Add sentiment features if available
                    if self.sentiment_analyzer:
                        try:
                            home_team_id = int(row.get('home_team_id', 0))
                            away_team_id = int(row.get('away_team_id', 0))
                            match_date = row.get('match_date')
                            
                            if home_team_id and away_team_id and match_date:
                                # Get team names for sentiment analysis
                                home_team_name = self._get_team_name(home_team_id)
                                away_team_name = self._get_team_name(away_team_id)
                                
                                # Get sentiment scores for both teams
                                sentiment_result = self.sentiment_analyzer.get_sentiment_for_match(
                                    home_team=home_team_name,
                                    away_team=away_team_name,
                                    match_date=match_date
                                )
                                
                                home_sentiment = sentiment_result.home_sentiment
                                away_sentiment = sentiment_result.away_sentiment
                                
                                # Add sentiment features
                                features['home_team_sentiment'] = float(home_sentiment) if home_sentiment is not None else 0.0
                                features['away_team_sentiment'] = float(away_sentiment) if away_sentiment is not None else 0.0
                                
                                logger.debug(f"Added sentiment features for fixture {fixture_id}: home={home_sentiment:.3f}, away={away_sentiment:.3f}")
                            else:
                                logger.debug(f"Missing team IDs or match date for sentiment features in fixture {fixture_id}")
                                features['home_team_sentiment'] = 0.0
                                features['away_team_sentiment'] = 0.0
                        except Exception as e:
                            logger.warning(f"Error generating sentiment features for fixture {fixture_id}: {e}")
                            features['home_team_sentiment'] = 0.0
                            features['away_team_sentiment'] = 0.0
                    else:
                        logger.debug("Sentiment analyzer not available, skipping sentiment features")
                        features['home_team_sentiment'] = 0.0
                        features['away_team_sentiment'] = 0.0
                    
                    # Validate feature values
                    for key, value in features.items():
                        if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                            logger.warning(f"Invalid value for feature {key}: {value}, setting to 0")
                            features[key] = 0.0
                    
                    logger.info(f"Successfully extracted {len(features)} features for fixture {fixture_id}")
                    logger.debug(f"Feature summary: home_xg={features.get('home_xg', 0):.2f}, away_xg={features.get('away_xg', 0):.2f}")
                    
                    return features
                    
                except Exception as e:
                    logger.error(f"Error calculating features for fixture {fixture_id}: {e}")
                    logger.debug(f"Feature calculation error traceback: {traceback.format_exc()}")
                    return None
                    
            except SQLAlchemyError as e:
                logger.error(f"Database error in extract_enhanced_features for fixture {fixture_id}: {e}")
                self.db_session.rollback()
                return None
            except Exception as e:
                logger.error(f"Unexpected error in extract_enhanced_features for fixture {fixture_id}: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return None
    
    def prepare_training_data(self, league_id: int = None, days_back: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with enhanced features and comprehensive error handling."""
        with timer(f"Preparing training data for league {league_id or 'all'}"):
            try:
                # Input validation
                if days_back <= 0:
                    logger.error(f"Invalid days_back parameter: {days_back}")
                    return np.array([]), np.array([])
                
                # Use a fixed cutoff date that covers the actual data in the database
                # Database contains fixtures from 2023-08-26 to 2025-08-12
                # Calculate dynamic cutoff date based on available data and config
                end_date = datetime.now()
                months_back = 24  # Use 24 months of data for better training
                cutoff_date = end_date - timedelta(days=months_back * 30.44)
                
                # Ensure we don't go before earliest available data
                earliest_data_date = datetime(2023, 8, 1)  # Based on actual data range
                cutoff_date = max(cutoff_date, earliest_data_date)
                logger.info(f"Preparing training data from {cutoff_date.strftime('%Y-%m-%d')} for league {league_id or 'all'}")
                
                # Get completed fixtures with comprehensive query
                query = """
                SELECT id, home_score + away_score as total_goals, match_date, league_id
                FROM fixtures 
                WHERE home_score IS NOT NULL 
                    AND away_score IS NOT NULL
                    AND match_date >= :cutoff_date
                """
                
                params = {'cutoff_date': cutoff_date.isoformat()}
                if league_id:
                    query += " AND league_id = :league_id"
                    params['league_id'] = league_id
                    
                query += " ORDER BY match_date DESC"
                    
                logger.debug("Executing fixtures query for training data")
                
                try:
                    fixtures_df = pd.read_sql_query(text(query), self.engine, params=params)
                    logger.info(f"Found {len(fixtures_df)} fixtures for training")
                except SQLAlchemyError as e:
                    logger.error(f"Database error while fetching fixtures: {e}")
                    self.db_session.rollback()
                    return np.array([]), np.array([])
                except Exception as e:
                    logger.error(f"Unexpected error while fetching fixtures: {e}")
                    return np.array([]), np.array([])
                
                if len(fixtures_df) == 0:
                    logger.warning(f"No fixtures found for league {league_id or 'all'} since {cutoff_date.strftime('%Y-%m-%d')}")
                    return np.array([]), np.array([])
                
                features_list = []
                targets = []
                failed_extractions = 0
                
                # Process fixtures
                logger.info(f"Extracting features from {len(fixtures_df)} fixtures")
                
                for idx, row in fixtures_df.iterrows():
                    try:
                        if idx % 50 == 0:  # Log progress every 50 fixtures
                            logger.debug(f"Processing fixture {idx + 1}/{len(fixtures_df)}")
                        
                        features = self.extract_enhanced_features(row['id'])
                        if features:
                            # Create feature vector from sorted keys (excluding metadata and string features)
                            feature_keys = sorted([k for k in features.keys() 
                                                 if k not in ['fixture_id', 'league_id'] 
                                                 and not isinstance(features[k], str)])
                            feature_vector = [features[key] for key in feature_keys]
                            
                            # Validate feature vector - ensure all values are numeric
                            try:
                                feature_vector = [float(val) for val in feature_vector]
                                if any(np.isnan(val) or np.isinf(val) for val in feature_vector):
                                    logger.warning(f"Invalid values in feature vector for fixture {row['id']}, skipping")
                                    failed_extractions += 1
                                else:
                                    features_list.append(feature_vector)
                                    targets.append(float(row['total_goals']))
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Non-numeric values in feature vector for fixture {row['id']}: {e}, skipping")
                                failed_extractions += 1
                        else:
                            failed_extractions += 1
                            logger.debug(f"Failed to extract features for fixture {row['id']}")
                                
                    except Exception as e:
                        failed_extractions += 1
                        logger.warning(f"Error processing fixture {row['id']}: {e}")
                
                # Log extraction statistics
                successful_extractions = len(features_list)
                total_fixtures = len(fixtures_df)
                success_rate = (successful_extractions / total_fixtures * 100) if total_fixtures > 0 else 0
                
                logger.info(f"Feature extraction completed: {successful_extractions}/{total_fixtures} successful ({success_rate:.1f}%)")
                if failed_extractions > 0:
                    logger.warning(f"Failed to extract features for {failed_extractions} fixtures")
                
                if len(features_list) == 0:
                    logger.error("No valid features extracted from any fixture")
                    return np.array([]), np.array([])
                
                # Convert to numpy arrays with validation
                try:
                    X = np.array(features_list, dtype=np.float64)
                    y = np.array(targets, dtype=np.float64)
                    
                    # Final validation
                    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                        logger.error("NaN or Inf values detected in feature matrix")
                        return np.array([]), np.array([])
                    
                    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        logger.error("NaN or Inf values detected in target vector")
                        return np.array([]), np.array([])
                    
                    logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
                    logger.debug(f"Target statistics: mean={np.mean(y):.2f}, std={np.std(y):.2f}, min={np.min(y):.1f}, max={np.max(y):.1f}")
                    
                    return X, y
                    
                except Exception as e:
                    logger.error(f"Error converting to numpy arrays: {e}")
                    return np.array([]), np.array([])
                    
            except Exception as e:
                logger.error(f"Unexpected error in prepare_training_data: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return np.array([]), np.array([])
    
    def train_models(self, league_id: int = None) -> bool:
        """Train quantile regression and Poisson models with comprehensive error handling."""
        with timer(f"Training models for league {league_id or 'global'}"):
            try:
                logger.info(f"Starting model training for league {league_id or 'global'}")
                
                # Prepare training data
                X, y = self.prepare_training_data(league_id)
                
                if len(X) == 0:
                    logger.warning(f"No training data available for league {league_id}")
                    return False
                
                logger.info(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
                logger.debug(f"Target distribution: mean={np.mean(y):.2f}, std={np.std(y):.2f}")
                
                # Scale features with error handling
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    logger.debug("Feature scaling completed successfully")
                    
                    # Validate scaled features
                    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                        logger.error("NaN or Inf values detected in scaled features")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error during feature scaling: {e}")
                    return False
                
                models = {}
                training_errors = []
                
                # Train Poisson baseline with error handling
                try:
                    logger.debug("Training Poisson baseline model")
                    poisson_model = PoissonRegressor(**self.model_configs['poisson_baseline']['params'])
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        poisson_model.fit(X_scaled, y)
                    
                    models['poisson'] = poisson_model
                    logger.debug("Poisson model training completed")
                    
                except Exception as e:
                    logger.error(f"Error training Poisson model: {e}")
                    training_errors.append(f"Poisson: {e}")
                
                # Train quantile regression models
                logger.debug(f"Training {len(self.quantiles)} quantile regression models")
                
                for i, quantile in enumerate(self.quantiles, 1):
                    try:
                        logger.debug(f"Training quantile {quantile} model ({i}/{len(self.quantiles)})")
                        qr_params = self.model_configs['quantile_regression']['params'].copy()
                        qr_params['loss'] = 'quantile'
                        qr_params['alpha'] = quantile
                        
                        qr_model = GradientBoostingRegressor(**qr_params)
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            qr_model.fit(X_scaled, y)
                        
                        models[f'quantile_{quantile}'] = qr_model
                        logger.debug(f"Quantile {quantile} model training completed")
                        
                    except Exception as e:
                        logger.error(f"Error training quantile {quantile} model: {e}")
                        training_errors.append(f"Quantile {quantile}: {e}")
                
                # Check if we have enough models to proceed
                if len(models) == 0:
                    logger.error("No models were successfully trained")
                    return False
                
                if 'quantile_0.5' not in models:
                    logger.error("Median quantile model failed to train")
                    return False
                
                # Evaluate models with error handling
                try:
                    logger.debug("Evaluating model performance")
                    
                    evaluation_results = {}
                    
                    if 'poisson' in models:
                        poisson_pred = models['poisson'].predict(X_scaled)
                        poisson_mae = mean_absolute_error(y, poisson_pred)
                        poisson_rmse = np.sqrt(mean_squared_error(y, poisson_pred))
                        evaluation_results['poisson'] = {'mae': poisson_mae, 'rmse': poisson_rmse}
                    
                    median_pred = models['quantile_0.5'].predict(X_scaled)
                    quantile_mae = mean_absolute_error(y, median_pred)
                    quantile_rmse = np.sqrt(mean_squared_error(y, median_pred))
                    evaluation_results['quantile_median'] = {'mae': quantile_mae, 'rmse': quantile_rmse}
                    
                    # Log evaluation results
                    logger.info(f"Model evaluation for league {league_id or 'global'}:")
                    for model_name, metrics in evaluation_results.items():
                        logger.info(f"  {model_name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error during model evaluation: {e}")
                    # Continue despite evaluation errors
                
                # Store models and scaler
                model_key = f'league_{league_id}' if league_id else 'global'
                self.models[model_key] = models
                self.scalers[model_key] = scaler
                
                # Calculate and store feature importance
                try:
                    logger.debug("Calculating feature importance")
                    
                    # Get feature names from a sample extraction
                    sample_features = self.extract_enhanced_features(1)
                    if sample_features:
                        feature_names = [key for key in sorted(sample_features.keys()) 
                                       if key not in ['fixture_id', 'league_id']]
                        
                        if len(feature_names) == len(models['quantile_0.5'].feature_importances_):
                            importance = models['quantile_0.5'].feature_importances_
                            self.feature_importance[model_key] = dict(zip(feature_names, importance))
                            
                            # Log top features
                            top_features = sorted(self.feature_importance[model_key].items(), 
                                                key=lambda x: x[1], reverse=True)[:5]
                            logger.debug(f"Top 5 features: {[f'{name}={imp:.3f}' for name, imp in top_features]}")
                        else:
                            logger.warning("Feature name count mismatch with importance array")
                    
                except Exception as e:
                    logger.error(f"Error calculating feature importance: {e}")
                
                # Log training summary
                if training_errors:
                    logger.warning(f"Training completed with {len(training_errors)} errors: {'; '.join(training_errors)}")
                else:
                    logger.info(f"All models trained successfully for league {league_id or 'global'}")
                
                logger.info(f"Model training completed for league {league_id or 'global'}: {len(models)} models stored")
                return True
                
            except Exception as e:
                logger.error(f"Unexpected error in train_models for league {league_id}: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return False
    
    def predict_with_uncertainty(self, fixture_id: int) -> Dict:
        """Generate predictions with confidence intervals and comprehensive error handling."""
        with timer(f"Generating predictions for fixture {fixture_id}"):
            try:
                logger.debug(f"Starting prediction for fixture {fixture_id}")
                
                # Input validation
                if not isinstance(fixture_id, int) or fixture_id <= 0:
                    logger.error(f"Invalid fixture_id: {fixture_id}")
                    return None
                
                # Extract features with error handling
                try:
                    features = self.extract_enhanced_features(fixture_id)
                    if not features:
                        logger.warning(f"No features extracted for fixture {fixture_id}")
                        return None
                    
                    logger.debug(f"Extracted {len(features)} features for fixture {fixture_id}")
                    
                except Exception as e:
                    logger.error(f"Error extracting features for fixture {fixture_id}: {e}")
                    return None
                
                league_id = features.get('league_id')
                if league_id is None:
                    logger.error(f"No league_id found in features for fixture {fixture_id}")
                    return None
                
                # Determine model to use
                model_key = f'league_{league_id}' if f'league_{league_id}' in self.models else 'global'
                
                if model_key not in self.models:
                    logger.warning(f"No trained model available for prediction (tried {model_key})")
                    return None
                
                logger.debug(f"Using model: {model_key}")
                
                try:
                    models = self.models[model_key]
                    scaler = self.scalers[model_key]
                    
                    # Validate models
                    if not models or not scaler:
                        logger.error(f"Invalid models or scaler for {model_key}")
                        return None
                    
                except Exception as e:
                    logger.error(f"Error accessing models for {model_key}: {e}")
                    return None
                
                # Prepare feature vector with error handling
                try:
                    feature_keys = [key for key in sorted(features.keys()) 
                                  if key not in ['fixture_id', 'league_id']]
                    
                    if not feature_keys:
                        logger.error(f"No valid feature keys found for fixture {fixture_id}")
                        return None
                    
                    feature_values = [features[key] for key in feature_keys]
                    
                    # Validate feature values
                    if any(pd.isna(val) or np.isinf(val) for val in feature_values):
                        logger.warning(f"Invalid feature values detected for fixture {fixture_id}")
                        # Replace invalid values with 0
                        feature_values = [0.0 if pd.isna(val) or np.isinf(val) else val for val in feature_values]
                    
                    feature_vector = np.array([feature_values])
                    feature_vector_scaled = scaler.transform(feature_vector)
                    
                    logger.debug(f"Prepared feature vector with {len(feature_values)} features")
                    
                except Exception as e:
                    logger.error(f"Error preparing feature vector for fixture {fixture_id}: {e}")
                    return None
                
                # Generate predictions with error handling
                predictions = {}
                prediction_errors = []
                
                # Poisson baseline prediction
                try:
                    if 'poisson' in models:
                        poisson_pred = models['poisson'].predict(feature_vector_scaled)[0]
                        
                        # Validate prediction
                        if pd.isna(poisson_pred) or np.isinf(poisson_pred) or poisson_pred < 0:
                            logger.warning(f"Invalid Poisson prediction: {poisson_pred}")
                            poisson_pred = 2.5  # Default reasonable value
                        
                        predictions['poisson_total_goals'] = float(poisson_pred)
                        logger.debug(f"Poisson prediction: {poisson_pred:.3f}")
                    else:
                        logger.warning("Poisson model not available")
                        
                except Exception as e:
                    logger.error(f"Error generating Poisson prediction: {e}")
                    prediction_errors.append(f"Poisson: {e}")
                
                # Quantile predictions
                quantile_preds = {}
                successful_quantiles = 0
                
                for quantile in self.quantiles:
                    try:
                        model_name = f'quantile_{quantile}'
                        if model_name in models:
                            pred = models[model_name].predict(feature_vector_scaled)[0]
                            
                            # Validate prediction
                            if pd.isna(pred) or np.isinf(pred) or pred < 0:
                                logger.warning(f"Invalid quantile {quantile} prediction: {pred}")
                                pred = 2.5  # Default reasonable value
                            
                            quantile_preds[quantile] = float(pred)
                            predictions[f'quantile_{quantile}'] = float(pred)
                            successful_quantiles += 1
                        else:
                            logger.warning(f"Quantile {quantile} model not available")
                            
                    except Exception as e:
                        logger.error(f"Error generating quantile {quantile} prediction: {e}")
                        prediction_errors.append(f"Quantile {quantile}: {e}")
                
                # Check if we have enough quantile predictions
                if successful_quantiles == 0:
                    logger.error("No successful quantile predictions generated")
                    return None
                
                if 0.5 not in quantile_preds:
                    logger.error("Median quantile prediction missing")
                    return None
                
                logger.debug(f"Generated {successful_quantiles} quantile predictions")
                
                # Calculate confidence intervals with error handling
                try:
                    if 0.1 in quantile_preds and 0.9 in quantile_preds:
                        predictions['prediction_interval_80'] = (quantile_preds[0.1], quantile_preds[0.9])
                    
                    if 0.25 in quantile_preds and 0.75 in quantile_preds:
                        predictions['prediction_interval_50'] = (quantile_preds[0.25], quantile_preds[0.75])
                    
                    predictions['median_total_goals'] = quantile_preds[0.5]
                    
                except Exception as e:
                    logger.error(f"Error calculating confidence intervals: {e}")
                
                # Calculate over/under probabilities with error handling
                try:
                    over_thresholds = [1.5, 2.5, 3.5]
                    for threshold in over_thresholds:
                        try:
                            prob = self._calculate_over_probability(quantile_preds, threshold)
                            
                            # Validate probability
                            if pd.isna(prob) or np.isinf(prob) or prob < 0 or prob > 1:
                                logger.warning(f"Invalid over {threshold} probability: {prob}")
                                prob = 0.5  # Default neutral probability
                            
                            predictions[f'over_{threshold}_probability'] = float(prob)
                            
                        except Exception as e:
                            logger.error(f"Error calculating over {threshold} probability: {e}")
                            predictions[f'over_{threshold}_probability'] = 0.5
                    
                    logger.debug("Over/under probabilities calculated")
                    
                except Exception as e:
                    logger.error(f"Error in over/under probability calculations: {e}")
                
                # Apply calibration if available
                try:
                    if (hasattr(self, 'calibration_monitor') and 
                        self.calibration_monitor and 
                        league_id in self.calibration_monitor.calibrators and
                        'over_2.5_probability' in predictions):
                        
                        calibrated_prob = self.calibration_monitor.calibrate_prediction(
                            league_id, predictions['over_2.5_probability']
                        )
                        
                        if not (pd.isna(calibrated_prob) or np.isinf(calibrated_prob)):
                            predictions['over_2_5_probability_calibrated'] = float(calibrated_prob)
                            logger.debug(f"Applied calibration: {predictions['over_2.5_probability']:.3f} -> {calibrated_prob:.3f}")
                        else:
                            logger.warning("Invalid calibrated probability")
                    
                except Exception as e:
                    logger.error(f"Error applying calibration: {e}")
                
                # Calculate confidence score with error handling
                try:
                    if 0.1 in quantile_preds and 0.9 in quantile_preds:
                        interval_width = quantile_preds[0.9] - quantile_preds[0.1]
                        confidence_score = max(0, 1 - (interval_width / 6))  # Normalize by reasonable range
                        
                        # Validate confidence score
                        if pd.isna(confidence_score) or np.isinf(confidence_score):
                            confidence_score = 0.5
                        
                        predictions['confidence_score'] = float(confidence_score)
                        logger.debug(f"Confidence score: {confidence_score:.3f}")
                    else:
                        predictions['confidence_score'] = 0.5
                        logger.warning("Cannot calculate confidence score - missing quantiles")
                    
                except Exception as e:
                    logger.error(f"Error calculating confidence score: {e}")
                    predictions['confidence_score'] = 0.5
                
                # Add metadata
                predictions['fixture_id'] = fixture_id
                predictions['league_id'] = league_id
                predictions['model_used'] = model_key
                predictions['prediction_timestamp'] = time.time()
                
                # Log prediction summary
                if prediction_errors:
                    logger.warning(f"Prediction completed with {len(prediction_errors)} errors: {'; '.join(prediction_errors)}")
                else:
                    logger.debug("All predictions generated successfully")
                
                logger.info(f"Prediction completed for fixture {fixture_id}: median={predictions.get('median_total_goals', 'N/A'):.3f}, confidence={predictions.get('confidence_score', 'N/A'):.3f}")
                
                return predictions
                
            except Exception as e:
                logger.error(f"Unexpected error in predict_with_uncertainty for fixture {fixture_id}: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return None
    
    def _calculate_over_probability(self, quantile_preds: Dict, threshold: float) -> float:
        """Estimate probability of exceeding threshold using quantile predictions with error handling."""
        try:
            # Input validation
            if not quantile_preds:
                logger.warning("Empty quantile predictions provided")
                return 0.5
            
            if not isinstance(threshold, (int, float)) or pd.isna(threshold) or np.isinf(threshold):
                logger.warning(f"Invalid threshold: {threshold}")
                return 0.5
            
            # Validate quantile predictions
            valid_preds = {}
            for q, pred in quantile_preds.items():
                if isinstance(q, (int, float)) and isinstance(pred, (int, float)):
                    if not (pd.isna(q) or pd.isna(pred) or np.isinf(q) or np.isinf(pred)):
                        if 0 <= q <= 1 and pred >= 0:
                            valid_preds[q] = pred
            
            if not valid_preds:
                logger.warning("No valid quantile predictions found")
                return 0.5
            
            # Find quantiles around the threshold
            quantiles_sorted = sorted(valid_preds.keys())
            
            for i, q in enumerate(quantiles_sorted):
                try:
                    if valid_preds[q] >= threshold:
                        if i == 0:
                            prob = 1 - q
                        else:
                            # Linear interpolation
                            q_prev = quantiles_sorted[i-1]
                            val_prev = valid_preds[q_prev]
                            val_curr = valid_preds[q]
                            
                            if val_curr == val_prev:
                                interp_q = q
                            else:
                                try:
                                    interp_q = q_prev + (q - q_prev) * (threshold - val_prev) / (val_curr - val_prev)
                                    
                                    # Validate interpolated quantile
                                    if pd.isna(interp_q) or np.isinf(interp_q) or interp_q < 0 or interp_q > 1:
                                        interp_q = q
                                        
                                except (ZeroDivisionError, ValueError):
                                    interp_q = q
                            
                            prob = 1 - interp_q
                        
                        # Validate final probability
                        if pd.isna(prob) or np.isinf(prob) or prob < 0 or prob > 1:
                            logger.warning(f"Invalid calculated probability: {prob}")
                            return 0.5
                        
                        return float(prob)
                        
                except Exception as e:
                    logger.error(f"Error in probability calculation for quantile {q}: {e}")
                    continue
            
            # If threshold is above all quantiles
            return 0.0
            
        except Exception as e:
            logger.error(f"Unexpected error in _calculate_over_probability: {e}")
            return 0.5
    
    def save_models(self, filepath: str = "models/enhanced_predictor.joblib") -> bool:
        """Save all models and scalers with comprehensive error handling."""
        with timer(f"Saving models to {filepath}"):
            try:
                logger.info(f"Starting model save to {filepath}")
                
                # Input validation
                if not filepath or not isinstance(filepath, str):
                    logger.error(f"Invalid filepath: {filepath}")
                    return False
                
                # Create directory if it doesn't exist
                try:
                    filepath_obj = Path(filepath)
                    filepath_obj.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {filepath_obj.parent}")
                except Exception as e:
                    logger.error(f"Error creating directory for {filepath}: {e}")
                    return False
                
                # Validate data to save
                if not self.models:
                    logger.warning("No models to save")
                    return False
                
                # Prepare model data
                try:
                    model_data = {
                        'models': self.models,
                        'scalers': self.scalers,
                        'feature_importance': self.feature_importance,
                        'quantiles': self.quantiles,
                        'model_configs': self.model_configs,
                        'save_timestamp': time.time(),
                        'version': '1.0'
                    }
                    
                    # Validate model data
                    total_models = sum(len(models) for models in self.models.values())
                    total_scalers = len(self.scalers)
                    
                    logger.debug(f"Preparing to save {total_models} models and {total_scalers} scalers")
                    
                except Exception as e:
                    logger.error(f"Error preparing model data: {e}")
                    return False
                
                # Save models
                try:
                    joblib.dump(model_data, filepath)
                    logger.info(f"Successfully saved enhanced predictor to {filepath}")
                    
                    # Verify save by checking file size
                    try:
                        file_size = filepath_obj.stat().st_size
                        logger.debug(f"Saved file size: {file_size / (1024*1024):.2f} MB")
                        
                        if file_size == 0:
                            logger.error("Saved file is empty")
                            return False
                            
                    except Exception as e:
                        logger.warning(f"Could not verify file size: {e}")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Error saving models to {filepath}: {e}")
                    return False
                
            except Exception as e:
                logger.error(f"Unexpected error in save_models: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return False
    
    def load_leagues_from_database(self, min_fixtures: int = 100) -> Dict[int, str]:
        """Load leagues from database with sufficient fixture data.
        
        Args:
            min_fixtures: Minimum number of fixtures required for a league
            
        Returns:
            Dict[int, str]: Dictionary mapping league_id to league_name
        """
        try:
            logger.info(f"Loading leagues from database with minimum {min_fixtures} fixtures")
            
            query = text("""
                SELECT 
                    l.league_pk as league_id,
                    l.name,
                    COUNT(f.id) as fixture_count
                FROM leagues l
                JOIN fixtures f ON l.league_pk = f.league_id
                WHERE f.home_score IS NOT NULL
                    AND f.away_score IS NOT NULL
                GROUP BY l.league_pk, l.name
                HAVING COUNT(f.id) >= :min_fixtures
                ORDER BY COUNT(f.id) DESC
            """)
            
            result = self.db_session.execute(query, {"min_fixtures": min_fixtures})
            leagues_data = result.mappings().all()
            
            leagues = {}
            for row in leagues_data:
                leagues[row['league_id']] = f"{row['name']} ({row['fixture_count']} fixtures)"
            
            logger.info(f"Found {len(leagues)} leagues with sufficient data")
            for league_id, league_name in leagues.items():
                logger.info(f"  • League {league_id}: {league_name}")
            
            return leagues
            
        except Exception as e:
            logger.error(f"Error loading leagues from database: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {}

    def load_models(self, filepath: str = "models/enhanced_predictor.joblib") -> bool:
        """Load models and scalers with comprehensive error handling."""
        with timer(f"Loading models from {filepath}"):
            try:
                logger.info(f"Starting model load from {filepath}")
                
                # Input validation
                if not filepath or not isinstance(filepath, str):
                    logger.error(f"Invalid filepath: {filepath}")
                    return False
                
                # Check if file exists
                filepath_obj = Path(filepath)
                if not filepath_obj.exists():
                    logger.warning(f"No saved models found at {filepath}")
                    return False
                
                if not filepath_obj.is_file():
                    logger.error(f"Path is not a file: {filepath}")
                    return False
                
                # Check file size
                try:
                    file_size = filepath_obj.stat().st_size
                    if file_size == 0:
                        logger.error(f"Model file is empty: {filepath}")
                        return False
                    
                    logger.debug(f"Loading file size: {file_size / (1024*1024):.2f} MB")
                    
                except Exception as e:
                    logger.warning(f"Could not check file size: {e}")
                
                # Load model data
                try:
                    model_data = joblib.load(filepath)
                    logger.debug("Model data loaded successfully")
                    
                except Exception as e:
                    logger.error(f"Error loading model data from {filepath}: {e}")
                    return False
                
                # Validate loaded data
                try:
                    if not isinstance(model_data, dict):
                        logger.error("Invalid model data format - not a dictionary")
                        return False
                    
                    required_keys = ['models', 'scalers']
                    missing_keys = [key for key in required_keys if key not in model_data]
                    
                    if missing_keys:
                        logger.error(f"Missing required keys in model data: {missing_keys}")
                        return False
                    
                    # Validate models
                    if not isinstance(model_data['models'], dict):
                        logger.error("Invalid models format")
                        return False
                    
                    if not isinstance(model_data['scalers'], dict):
                        logger.error("Invalid scalers format")
                        return False
                    
                    # Count loaded components
                    total_models = sum(len(models) for models in model_data['models'].values())
                    total_scalers = len(model_data['scalers'])
                    
                    logger.debug(f"Loaded {total_models} models and {total_scalers} scalers")
                    
                except Exception as e:
                    logger.error(f"Error validating model data: {e}")
                    return False
                
                # Apply loaded data
                try:
                    self.models = model_data['models']
                    self.scalers = model_data['scalers']
                    self.feature_importance = model_data.get('feature_importance', {})
                    
                    # Load additional metadata if available
                    if 'quantiles' in model_data:
                        self.quantiles = model_data['quantiles']
                        logger.debug(f"Loaded quantiles: {self.quantiles}")
                    
                    if 'model_configs' in model_data:
                        logger.debug("Loaded model configurations")
                    
                    if 'save_timestamp' in model_data:
                        save_time = model_data['save_timestamp']
                        logger.debug(f"Models saved at: {time.ctime(save_time)}")
                    
                    logger.info(f"Successfully loaded enhanced predictor from {filepath}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error applying loaded model data: {e}")
                    return False
                
            except Exception as e:
                logger.error(f"Unexpected error in load_models: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return False
    
    def _get_team_position(self, team_id, league_id, match_date):
        """Get team's league position with improved fallback logic.
        
        Returns tuple: (position, confidence_score)
        - position: team's league position
        - confidence_score: 1.0 (recent data), 0.5 (median), 0.0 (no data)
        """
        try:
            # 1. Try to get most recent position data (regardless of match date)
            recent_query = """
                SELECT position, updated_at
                FROM standings 
                WHERE team_id = :team_id 
                AND league_id = :league_id 
                ORDER BY updated_at DESC 
                LIMIT 1
            """
            
            result = self.db_session.execute(
                text(recent_query), 
                {"team_id": team_id, "league_id": league_id}
            ).fetchone()
            
            if result:
                position, updated_at = result[0], result[1]
                # High confidence if data is recent (within 30 days of match)
                days_diff = abs((match_date - updated_at).days) if updated_at else 999
                confidence = 1.0 if days_diff <= 30 else 0.8
                return position, confidence
            
            # 2. Fallback to league-specific median position
            median_query = """
                SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY position) as median_pos
                FROM standings 
                WHERE league_id = :league_id
                AND position IS NOT NULL
            """
            
            median_result = self.db_session.execute(
                text(median_query), 
                {"league_id": league_id}
            ).fetchone()
            
            if median_result and median_result[0] is not None:
                median_position = int(round(median_result[0]))
                logger.info(f"Using league median position {median_position} for team {team_id}")
                return median_position, 0.5
            
            # 3. Final fallback to default mid-table position
            logger.warning(f"No position data available for team {team_id}, using default")
            return 10, 0.0
                
        except Exception as e:
            logger.warning(f"Error getting team position for team {team_id}: {e}")
            return 10, 0.0
    
    def _get_team_position_simple(self, team_id, league_id, match_date):
        """Simplified version that returns only position for backward compatibility."""
        position, _ = self._get_team_position(team_id, league_id, match_date)
        return position

def main():
    """Train enhanced predictor with comprehensive error handling and progress tracking."""
    with timer("Enhanced predictor training pipeline"):
        try:
            console.print("\n[bold blue]🚀 Starting Enhanced Predictor Training Pipeline[/bold blue]")
            logger.info("Starting enhanced predictor training pipeline")
            
            # Load configuration
            try:
                with timer("Configuration loading"):
                    load_config()
                    config = get_config()
                    logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                console.print(f"[bold red]❌ Configuration loading failed: {e}[/bold red]")
                return False
            
            # Initialize predictor
            try:
                with timer("Predictor initialization"):
                    predictor = EnhancedGoalPredictor(config)
                    logger.info("Enhanced predictor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize predictor: {e}")
                console.print(f"[bold red]❌ Predictor initialization failed: {e}[/bold red]")
                return False
            
            # Load calibration monitor
            try:
                with timer("Calibration monitor loading"):
                    predictor.calibration_monitor.load_calibrators()
                    logger.info("Calibration monitor loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load calibration monitor: {e}")
                console.print(f"[yellow]⚠️ Calibration monitor loading failed: {e}[/yellow]")
            
            # Train global model
            try:
                with timer("Global model training"):
                    console.print("\n[bold green]🎯 Training global model...[/bold green]")
                    logger.info("Starting global model training")
                    
                    success = predictor.train_models()
                    if success:
                        logger.info("Global model training completed successfully")
                        console.print("[green]✅ Global model training completed[/green]")
                    else:
                        logger.error("Global model training failed")
                        console.print("[red]❌ Global model training failed[/red]")
                        
            except Exception as e:
                logger.error(f"Error during global model training: {e}")
                console.print(f"[bold red]❌ Global model training error: {e}[/bold red]")
            
            # Load leagues dynamically from database
            major_leagues = predictor.load_leagues_from_database(min_fixtures=100)
            
            if not major_leagues:
                logger.error("No leagues found with sufficient fixture data")
                console.print("[red]❌ No leagues found with sufficient fixture data[/red]")
                return False
            
            console.print(f"\n[bold cyan]🏆 Training {len(major_leagues)} league-specific models...[/bold cyan]")
            
            successful_leagues = []
            failed_leagues = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Training league models", total=len(major_leagues))
                
                for league_id, league_name in major_leagues.items():
                    try:
                        progress.update(task, description=f"Training {league_name} (ID: {league_id})")
                        
                        with timer(f"League {league_id} model training"):
                            logger.info(f"Starting training for league {league_id} ({league_name})")
                            
                            success = predictor.train_models(league_id)
                            
                            if success:
                                successful_leagues.append((league_id, league_name))
                                logger.info(f"Successfully trained model for league {league_id} ({league_name})")
                            else:
                                failed_leagues.append((league_id, league_name))
                                logger.error(f"Failed to train model for league {league_id} ({league_name})")
                        
                        progress.advance(task)
                        
                    except Exception as e:
                        failed_leagues.append((league_id, league_name))
                        logger.error(f"Error training model for league {league_id} ({league_name}): {e}")
                        progress.advance(task)
            
            # Report league training results
            if successful_leagues:
                console.print(f"\n[green]✅ Successfully trained {len(successful_leagues)} league models:[/green]")
                for league_id, league_name in successful_leagues:
                    console.print(f"  • {league_name} (ID: {league_id})")
            
            if failed_leagues:
                console.print(f"\n[red]❌ Failed to train {len(failed_leagues)} league models:[/red]")
                for league_id, league_name in failed_leagues:
                    console.print(f"  • {league_name} (ID: {league_id})")
            
            # Save models
            try:
                with timer("Model saving"):
                    console.print("\n[bold blue]💾 Saving trained models...[/bold blue]")
                    logger.info("Starting model save process")
                    
                    save_success = predictor.save_models()
                    
                    if save_success:
                        logger.info("Models saved successfully")
                        console.print("[green]✅ Models saved successfully[/green]")
                    else:
                        logger.error("Failed to save models")
                        console.print("[red]❌ Failed to save models[/red]")
                        
            except Exception as e:
                logger.error(f"Error during model saving: {e}")
                console.print(f"[bold red]❌ Model saving error: {e}[/bold red]")
            
            # Final summary
            console.print("\n[bold green]🎉 Enhanced predictor training pipeline completed![/bold green]")
            logger.info("Enhanced predictor training pipeline completed successfully")
            
            # Training summary
            total_leagues = len(major_leagues)
            success_rate = len(successful_leagues) / total_leagues * 100 if total_leagues > 0 else 0
            
            summary_table = Table(title="Training Summary", show_header=True, header_style="bold magenta")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Total Leagues", str(total_leagues))
            summary_table.add_row("Successful", str(len(successful_leagues)))
            summary_table.add_row("Failed", str(len(failed_leagues)))
            summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
            
            console.print(summary_table)
            
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in main training pipeline: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            console.print(f"[bold red]💥 Unexpected error in training pipeline: {e}[/bold red]")
            return False

if __name__ == "__main__":
    main()