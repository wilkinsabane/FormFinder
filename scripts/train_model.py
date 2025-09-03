from __future__ import annotations
import os
import glob
import json
import sys
import time
import traceback
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, DatabaseError
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
import argparse
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
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

# Add parent directory to path to import formfinder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Load configuration first
    from formfinder.config import load_config, get_config
    logger.info("🔧 Loading FormFinder configuration...")
    load_config()
    logger.info("✅ Configuration loaded successfully")
    
    # Now import modules that depend on configuration
    from formfinder.clients.api_client import SoccerDataAPIClient
    from formfinder.features import get_rolling_form_features, get_h2h_feature, get_preview_metrics
    from formfinder.logger import log
    
    # Import enhanced prediction capabilities
    from enhanced_predictor import EnhancedGoalPredictor
    from scripts.calibration_monitor import CalibrationMonitor
    from scripts.database_feature_engine import DatabaseFeatureEngine, load_features_from_database
    from formfinder.training_engine import TrainingEngine
    
    logger.info("📦 All modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import required modules: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Unexpected error during initialization: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

@contextmanager
def timer(description: str):
    """Context manager for timing operations with rich output."""
    start_time = time.time()
    logger.info(f"⏱️  Starting: {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"✅ Completed: {description} (took {elapsed:.2f}s)")

def load_training_data(db_session, leagues: List[int]) -> pd.DataFrame:
    """Load finished fixtures for training with comprehensive error handling and logging.
    
    Args:
        db_session: SQLAlchemy database session
        leagues: List of league IDs to load data for
        
    Returns:
        DataFrame containing training fixtures
        
    Raises:
        DatabaseError: If database query fails
        ValueError: If no data is found
    """
    logger.info(f"🔍 Loading training data for {len(leagues)} leagues: {leagues}")
    
    with timer("Database query for training fixtures"):
        try:
            query = text("""
                SELECT 
                    f.id as fixture_id,
                    f.match_date,
                    f.home_team_id,
                    f.away_team_id,
                    f.home_score,
                    f.away_score,
                    f.league_id
                FROM fixtures f
                WHERE f.status = 'finished'
                AND f.home_score IS NOT NULL
                AND f.away_score IS NOT NULL
                AND f.league_id = ANY(:leagues)
                ORDER BY f.match_date ASC
            """)
            
            logger.debug(f"📊 Executing query with parameters: leagues={leagues}")
            rows = db_session.execute(query, {"leagues": leagues}).mappings().all()
            
            if not rows:
                logger.warning(f"⚠️  No training data found for leagues {leagues}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            logger.info(f"📈 Successfully loaded {len(df)} training fixtures")
            
            # Log data quality metrics
            date_range = df['match_date'].agg(['min', 'max'])
            logger.info(f"📅 Date range: {date_range['min']} to {date_range['max']}")
            
            league_counts = df['league_id'].value_counts().to_dict()
            logger.info(f"🏆 Fixtures per league: {league_counts}")
            
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database error while loading training data: {e}")
            raise DatabaseError(f"Failed to load training data: {e}") from e
        except Exception as e:
            logger.error(f"❌ Unexpected error loading training data: {e}")
            logger.error(traceback.format_exc())
            raise

def build_features(df: pd.DataFrame, db_session, api_client) -> pd.DataFrame:
    """Build comprehensive features for training with enhanced error handling and progress tracking.
    
    Args:
        df: DataFrame containing fixture data
        db_session: Database session for queries
        api_client: API client for external data
        
    Returns:
        DataFrame with engineered features
        
    Raises:
        ValueError: If input DataFrame is empty or missing required columns
        DatabaseError: If database operations fail critically
    """
    with timer("🔧 Building features for training data"):
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        required_columns = ['fixture_id', 'home_team_id', 'away_team_id', 'match_date', 'league_id', 'home_score', 'away_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        logger.info(f"📊 Processing {len(df)} fixtures for feature engineering")
        
        features = []
        failed_fixtures = []
        feature_stats = {
            'home_form_failures': 0,
            'away_form_failures': 0,
            'h2h_failures': 0,
            'preview_failures': 0,
            'total_processed': 0
        }
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Building features...", total=len(df))
            
            for idx, row in df.iterrows():
                try:
                    fixture_id = row['fixture_id']
                    progress.update(task, description=f"Processing fixture {fixture_id}")
                    
                    # Default feature values for fallback
                    default_form_features = {
                        'avg_goals_scored_last_5': 0.0,
                        'avg_goals_conceded_last_5': 0.0,
                        'avg_goals_scored_home_last_5': 0.0,
                        'avg_goals_conceded_home_last_5': 0.0,
                        'avg_goals_scored_away_last_5': 0.0,
                        'avg_goals_conceded_away_last_5': 0.0
                    }
                    
                    default_h2h_features = {
                        'h2h_overall_games': 0,
                        'h2h_avg_total_goals': 0.0,
                        'h2h_overall_team1_goals': 0.0,
                        'h2h_overall_team2_goals': 0.0,
                        'h2h_team1_games_played_at_home': 0,
                        'h2h_team1_wins_at_home': 0,
                        'h2h_team1_losses_at_home': 0,
                        'h2h_team1_draws_at_home': 0,
                        'h2h_team1_scored_at_home': 0.0,
                        'h2h_team1_conceded_at_home': 0.0,
                        'h2h_team2_games_played_at_home': 0,
                        'h2h_team2_wins_at_home': 0,
                        'h2h_team2_losses_at_home': 0,
                        'h2h_team2_draws_at_home': 0,
                        'h2h_team2_scored_at_home': 0.0,
                        'h2h_team2_conceded_at_home': 0.0
                    }
                    
                    # Rolling form features for home team
                    try:
                        logger.debug(f"🏠 Fetching home team form for team {row['home_team_id']}")
                        home_form = get_rolling_form_features(
                            row['home_team_id'], 
                            row['match_date'], 
                            db_session, 
                            last_n_games=5
                        )
                        logger.debug(f"✅ Home form retrieved: avg goals {home_form.get('avg_goals_scored_last_5', 0):.2f}")
                    except SQLAlchemyError as e:
                        logger.warning(f"🔴 Database error getting home form for team {row['home_team_id']}: {e}")
                        feature_stats['home_form_failures'] += 1
                        try:
                            db_session.rollback()
                            logger.debug("🔄 Database session rolled back successfully")
                        except Exception as rollback_error:
                            logger.error(f"❌ Failed to rollback session: {rollback_error}")
                        home_form = default_form_features.copy()
                    except Exception as e:
                        logger.warning(f"⚠️ Unexpected error getting home form for team {row['home_team_id']}: {e}")
                        feature_stats['home_form_failures'] += 1
                        home_form = default_form_features.copy()
                    
                    # Rolling form features for away team
                    try:
                        logger.debug(f"✈️ Fetching away team form for team {row['away_team_id']}")
                        away_form = get_rolling_form_features(
                            row['away_team_id'], 
                            row['match_date'], 
                            db_session, 
                            last_n_games=5
                        )
                        logger.debug(f"✅ Away form retrieved: avg goals {away_form.get('avg_goals_scored_last_5', 0):.2f}")
                    except SQLAlchemyError as e:
                        logger.warning(f"🔴 Database error getting away form for team {row['away_team_id']}: {e}")
                        feature_stats['away_form_failures'] += 1
                        try:
                            db_session.rollback()
                            logger.debug("🔄 Database session rolled back successfully")
                        except Exception as rollback_error:
                            logger.error(f"❌ Failed to rollback session: {rollback_error}")
                        away_form = default_form_features.copy()
                    except Exception as e:
                        logger.warning(f"⚠️ Unexpected error getting away form for team {row['away_team_id']}: {e}")
                        feature_stats['away_form_failures'] += 1
                        away_form = default_form_features.copy()
                    
                    # H2H features
                    try:
                        logger.debug(f"🤝 Fetching H2H data for {row['home_team_id']} vs {row['away_team_id']}")
                        h2h = get_h2h_feature(
                            row['home_team_id'], 
                            row['away_team_id'], 
                            api_client,
                            competition_id=row['league_id']
                        )
                        logger.debug(f"✅ H2H retrieved: {h2h.get('h2h_overall_games', 0)} historical games")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get H2H features for teams {row['home_team_id']} vs {row['away_team_id']}: {e}")
                        feature_stats['h2h_failures'] += 1
                        h2h = default_h2h_features.copy()
                    
                    # Match preview features (if available)
                    preview = {'preview_excitement_rating': 0.0}
                    if 'match_api_id' in row and row['match_api_id']:
                        try:
                            logger.debug(f"📋 Fetching preview metrics for match {row['match_api_id']}")
                            preview = get_preview_metrics(row['match_api_id'], api_client, compute_sentiment=False)
                            logger.debug(f"✅ Preview retrieved: excitement rating {preview.get('preview_excitement_rating', 0):.2f}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get preview for match {row['match_api_id']}: {e}")
                            feature_stats['preview_failures'] += 1
                            preview = {'preview_excitement_rating': 0.0}
                    else:
                        logger.debug(f"ℹ️ No match API ID available for fixture {fixture_id}")
            
                    # Combine all features with comprehensive validation
                    try:
                        total_goals = int(row['home_score']) + int(row['away_score'])
                        over_2_5 = int(total_goals > 2.5)
                        
                        feature_row = {
                            'fixture_id': row['fixture_id'],
                            'match_date': row['match_date'],
                            'league_id': row['league_id'],
                            'home_team_id': row['home_team_id'],
                            'away_team_id': row['away_team_id'],
                            # Home team form features
                            'home_avg_goals_scored': float(home_form.get('avg_goals_scored_last_5', 0.0)),
                            'home_avg_goals_conceded': float(home_form.get('avg_goals_conceded_last_5', 0.0)),
                            'home_avg_goals_scored_home': float(home_form.get('avg_goals_scored_home_last_5', 0.0)),
                            'home_avg_goals_conceded_home': float(home_form.get('avg_goals_conceded_home_last_5', 0.0)),
                            # Away team form features
                            'away_avg_goals_scored': float(away_form.get('avg_goals_scored_last_5', 0.0)),
                            'away_avg_goals_conceded': float(away_form.get('avg_goals_conceded_last_5', 0.0)),
                            'away_avg_goals_scored_away': float(away_form.get('avg_goals_scored_away_last_5', 0.0)),
                            'away_avg_goals_conceded_away': float(away_form.get('avg_goals_conceded_away_last_5', 0.0)),
                            # H2H features
                            'h2h_total_matches': int(h2h.get('h2h_total_matches', 0)),
                            'h2h_avg_goals': float(h2h.get('h2h_avg_goals', 0.0)),
                            'h2h_home_wins': int(h2h.get('h2h_home_wins', 0)),
                            'h2h_away_wins': int(h2h.get('h2h_away_wins', 0)),
                            # Preview features
                            'excitement_rating': float(preview.get('preview_excitement_rating', 0.0)),
                            # Weather features with safe defaults
                            'weather_temp_c': float(preview.get('weather_temp_c', 21.0)),
                            'weather_temp_f': float(preview.get('weather_temp_f', 69.8)),
                            'weather_humidity': float(preview.get('weather_humidity', 50.0)),
                            'weather_wind_speed': float(preview.get('weather_wind_speed', 5.0)),
                            'weather_precipitation': float(preview.get('weather_precipitation', 0.0)),
                            # Target variables
                            'total_goals': total_goals,
                            'over_2_5': over_2_5
                        }
                        
                        # Validate feature values
                        for key, value in feature_row.items():
                            if key not in ['fixture_id', 'match_date'] and pd.isna(value):
                                logger.warning(f"⚠️ NaN value detected for feature '{key}' in fixture {fixture_id}, setting to 0")
                                feature_row[key] = 0.0 if isinstance(value, float) else 0
                        
                        features.append(feature_row)
                        feature_stats['total_processed'] += 1
                        
                        logger.debug(f"✅ Features built for fixture {fixture_id}: {total_goals} goals, over 2.5: {over_2_5}")
                        
                    except (ValueError, TypeError) as e:
                        logger.error(f"❌ Data validation error for fixture {fixture_id}: {e}")
                        failed_fixtures.append(fixture_id)
                        continue
                        
                except Exception as e:
                    logger.error(f"❌ Critical error building features for fixture {row.get('fixture_id', 'unknown')}: {e}")
                    logger.error(f"🔍 Error details: {traceback.format_exc()}")
                    failed_fixtures.append(row.get('fixture_id', 'unknown'))
                    continue
                finally:
                    progress.advance(task)
        
        # Log comprehensive statistics
        logger.info(f"📈 Feature engineering completed:")
        logger.info(f"   ✅ Successfully processed: {feature_stats['total_processed']} fixtures")
        logger.info(f"   ❌ Failed fixtures: {len(failed_fixtures)}")
        logger.info(f"   🏠 Home form failures: {feature_stats['home_form_failures']}")
        logger.info(f"   ✈️ Away form failures: {feature_stats['away_form_failures']}")
        logger.info(f"   🤝 H2H failures: {feature_stats['h2h_failures']}")
        logger.info(f"   📋 Preview failures: {feature_stats['preview_failures']}")
        
        if failed_fixtures:
            logger.warning(f"⚠️ Failed to process fixtures: {failed_fixtures[:10]}{'...' if len(failed_fixtures) > 10 else ''}")
        
        if not features:
            raise ValueError("No features were successfully generated from the input data")
            
        result_df = pd.DataFrame(features)
        logger.info(f"🎯 Generated feature matrix: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
        
        # Log feature statistics
        numeric_features = result_df.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            logger.info(f"📊 Feature statistics:")
            for feature in ['total_goals', 'home_avg_goals_scored', 'away_avg_goals_scored', 'h2h_overall_games']:
                if feature in result_df.columns:
                    mean_val = result_df[feature].mean()
                    std_val = result_df[feature].std()
                    logger.info(f"   {feature}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        return result_df

def train_models(X_train: np.ndarray, y_total: np.ndarray, y_over25: np.ndarray, sample_weight: Optional[np.ndarray] = None, X_df: Optional[pd.DataFrame] = None) -> Tuple[GradientBoostingRegressor, GradientBoostingClassifier, StandardScaler]:
    """Train models for total goals and over/under 2.5 with comprehensive monitoring.
    
    Args:
        X_train: Training feature matrix
        y_total: Total goals target values
        y_over25: Over/under 2.5 goals binary target
        sample_weight: Optional sample weights for recency weighting
        
    Returns:
        Tuple of (regressor, classifier, scaler)
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If model training fails
    """
    with timer("🤖 Training machine learning models"):
        # Input validation
        if X_train.shape[0] == 0:
            raise ValueError("Training data is empty")
        if len(y_total) != X_train.shape[0] or len(y_over25) != X_train.shape[0]:
            raise ValueError("Feature matrix and target arrays have mismatched lengths")
            
        logger.info(f"🎯 Training data shape: {X_train.shape}")
        logger.info(f"📊 Target statistics:")
        logger.info(f"   Total goals - mean: {np.mean(y_total):.2f}, std: {np.std(y_total):.2f}, range: [{np.min(y_total)}, {np.max(y_total)}]")
        logger.info(f"   Over 2.5 goals - positive rate: {np.mean(y_over25):.1%} ({np.sum(y_over25)}/{len(y_over25)})")
        
        # Feature scaling with monitoring
        try:
            logger.info("⚖️ Applying feature scaling (StandardScaler)...")
            scaler = StandardScaler()
            
            with timer("Feature scaling"):
                # Use DataFrame if available to preserve feature names, otherwise use numpy array
                if X_df is not None:
                    X_scaled = scaler.fit_transform(X_df)
                else:
                    X_scaled = scaler.fit_transform(X_train)
            
            logger.info(f"✅ Feature scaling completed")
            logger.info(f"📈 Scaled features - mean: {np.mean(X_scaled):.6f}, std: {np.std(X_scaled):.6f}")
            
            # Check for potential scaling issues
            if np.any(np.isnan(X_scaled)):
                nan_features = np.any(np.isnan(X_scaled), axis=0)
                logger.error(f"❌ NaN values detected in scaled features at indices: {np.where(nan_features)[0]}")
                raise ValueError("NaN values in scaled features")
                
            if np.any(np.isinf(X_scaled)):
                inf_features = np.any(np.isinf(X_scaled), axis=0)
                logger.error(f"❌ Infinite values detected in scaled features at indices: {np.where(inf_features)[0]}")
                raise ValueError("Infinite values in scaled features")
                
        except Exception as e:
            logger.error(f"❌ Feature scaling failed: {e}")
            raise RuntimeError(f"Feature scaling failed: {e}")
        
        # Total goals model (regression)
        try:
            logger.info("🎯 Training total goals regression model...")
            regressor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                verbose=0  # Suppress sklearn verbose output
            )
            
            with timer("Regression model training"):
                if sample_weight is not None:
                    regressor.fit(X_scaled, y_total, sample_weight=sample_weight)
                    logger.info(f"⚖️ Applied recency weights to regression model")
                else:
                    regressor.fit(X_scaled, y_total)
            
            # Model validation
            train_predictions = regressor.predict(X_scaled)
            train_mae = mean_absolute_error(y_total, train_predictions)
            train_r2 = regressor.score(X_scaled, y_total)
            
            logger.info(f"✅ Regression model trained successfully")
            logger.info(f"📊 Training performance:")
            logger.info(f"   MAE: {train_mae:.3f}")
            logger.info(f"   R² Score: {train_r2:.3f}")
            logger.info(f"   Feature importance (top 5): {sorted(enumerate(regressor.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}")
            
        except Exception as e:
            logger.error(f"❌ Regression model training failed: {e}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            raise RuntimeError(f"Regression model training failed: {e}")
        
        # Over/under 2.5 model (classification)
        try:
            logger.info("🎲 Training over/under 2.5 classification model...")
            classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                verbose=0  # Suppress sklearn verbose output
            )
            
            with timer("Classification model training"):
                if sample_weight is not None:
                    classifier.fit(X_scaled, y_over25, sample_weight=sample_weight)
                    logger.info(f"⚖️ Applied recency weights to classification model")
                else:
                    classifier.fit(X_scaled, y_over25)
            
            # Model validation
            train_proba = classifier.predict_proba(X_scaled)[:, 1]
            train_predictions_class = classifier.predict(X_scaled)
            train_accuracy = classifier.score(X_scaled, y_over25)
            train_brier = brier_score_loss(y_over25, train_proba)
            
            logger.info(f"✅ Classification model trained successfully")
            logger.info(f"📊 Training performance:")
            logger.info(f"   Accuracy: {train_accuracy:.3f}")
            logger.info(f"   Brier Score: {train_brier:.3f}")
            logger.info(f"   Feature importance (top 5): {sorted(enumerate(classifier.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}")
            
            # Class distribution check
            unique, counts = np.unique(train_predictions_class, return_counts=True)
            logger.info(f"   Predicted class distribution: {dict(zip(unique, counts))}")
            
        except Exception as e:
            logger.error(f"❌ Classification model training failed: {e}")
            logger.error(f"🔍 Error details: {traceback.format_exc()}")
            raise RuntimeError(f"Classification model training failed: {e}")
        
        logger.info(f"🎉 Model training completed successfully")
        return regressor, classifier, scaler

def main() -> None:
    """Main training pipeline with enhanced prediction capabilities and comprehensive monitoring.
    
    This function orchestrates the complete model training workflow including:
    - Configuration loading and validation
    - Data loading and preprocessing
    - Feature engineering
    - Model training and validation
    - Model persistence
    - Enhanced predictor training
    - Calibration monitoring
    
    Raises:
        SystemExit: If critical errors occur during training
    """
    start_time = time.time()
    
    # Initialize database variables at function start to ensure they're available in finally
    db_session = None
    engine = None
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="FormFinder model training pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python train_model.py                           # Standard training
  python train_model.py --api-timeout 30         # Custom API timeout
  python train_model.py --api-max-retries 5      # Custom retry count
            """
        )
        parser.add_argument("--api-timeout", type=float, help="Override API timeout (seconds)")
        parser.add_argument("--api-max-retries", type=int, help="Override API max retries")
        parser.add_argument("--min-samples", type=int, default=100, help="Minimum training samples required")
        parser.add_argument("--cv-splits", type=int, default=5, help="Number of cross-validation splits")
        parser.add_argument("--use-enhanced-features", action="store_true", help="Use enhanced predictor for feature extraction instead of database features")
        args = parser.parse_args()
        
        # Display startup banner
        rprint(Panel.fit(
            "[bold blue]🚀 FormFinder Model Training Pipeline[/bold blue]\n"
            "[dim]Advanced machine learning for football predictions[/dim]",
            border_style="blue"
        ))
        
        # Load and validate configuration
        with timer("⚙️ Loading configuration"):
            load_config()
            config = get_config()
            dynamic_config = config.dynamic_training
            
            # Apply CLI overrides
            if args.api_timeout is not None:
                config.api.timeout = args.api_timeout
                logger.info(f"🔧 API timeout overridden to {args.api_timeout} seconds")
            if args.api_max_retries is not None:
                config.api.max_retries = args.api_max_retries
                logger.info(f"🔧 API max retries overridden to {args.api_max_retries}")
        
        # Database connection with retry logic
        try:
            with timer("🗄️ Establishing database connection"):
                engine = create_engine(
                    config.get_database_url(),
                    pool_pre_ping=True,  # Verify connections before use
                    pool_recycle=3600,   # Recycle connections every hour
                    echo=False           # Set to True for SQL debugging
                )
                Session = sessionmaker(bind=engine)
                db_session = Session()
                
                # Test connection
                db_session.execute(text("SELECT 1"))
                logger.info("✅ Database connection established successfully")
                
        except SQLAlchemyError as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise SystemExit(1)
        except Exception as e:
            logger.error(f"❌ Unexpected database error: {e}")
            raise SystemExit(1)
        
        try:
            # Load leagues configuration
            leagues_file = Path('free_leagues.txt')
            if not leagues_file.exists():
                logger.error(f"❌ Leagues file not found: {leagues_file}")
                raise SystemExit(1)
                
            with timer("📋 Loading leagues configuration"):
                try:
                    with open(leagues_file, 'r') as f:
                        leagues = [int(line.strip()) for line in f if line.strip().isdigit()]
                    
                    if not leagues:
                        logger.error("❌ No valid leagues found in free_leagues.txt")
                        raise SystemExit(1)
                        
                    logger.info(f"🏆 Training on {len(leagues)} leagues: {leagues}")
                    
                except (ValueError, IOError) as e:
                    logger.error(f"❌ Error reading leagues file: {e}")
                    raise SystemExit(1)
            
            # Load and validate training data
            with timer("📊 Loading training data"):
                try:
                    df = load_training_data(db_session, leagues)
                    
                    if df.empty:
                        logger.error("❌ No training data loaded")
                        raise SystemExit(1)
                        
                    logger.info(f"📈 Loaded {len(df)} training samples")
                    
                    if len(df) < args.min_samples:
                        logger.error(f"❌ Insufficient training data: {len(df)} < {args.min_samples} required")
                        raise SystemExit(1)
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load training data: {e}")
                    raise SystemExit(1)
            
            # Initialize API client with error handling
            try:
                with timer("🌐 Initializing API client"):
                    api_client = SoccerDataAPIClient(db_session)
                    logger.info("✅ API client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize API client: {e}")
                raise SystemExit(1)
            
            # Build features using either enhanced predictor or database approach
            if args.use_enhanced_features:
                # Use enhanced predictor for feature extraction
                try:
                    with timer("🚀 Initializing TrainingEngine with enhanced predictor"):
                        training_engine = TrainingEngine()
                        training_engine.initialize_enhanced_predictor()
                    
                    with timer("📊 Loading enhanced training data"):
                        features_df = training_engine.load_enhanced_training_data(leagues=leagues)
                        
                        if features_df.empty:
                            logger.error("❌ No enhanced training data loaded")
                            raise SystemExit(1)
                            
                        logger.info(f"📈 Loaded {len(features_df)} samples with enhanced features")
                        
                        if len(features_df) < args.min_samples:
                            logger.error(f"❌ Insufficient enhanced training data: {len(features_df)} < {args.min_samples} required")
                            raise SystemExit(1)
                            
                except Exception as e:
                    logger.error(f"❌ Enhanced feature extraction failed: {e}")
                    logger.error(f"🔍 Error details: {traceback.format_exc()}")
                    raise SystemExit(1)
            else:
                # Use traditional database feature approach
                try:
                    with timer("🔧 Initializing DatabaseFeatureEngine"):
                        feature_engine = DatabaseFeatureEngine(db_session)
                    
                    with timer("📊 Loading features from database"):
                        # Calculate dynamic training date range
                        config = get_config()
                        dynamic_config = config.dynamic_training
                        
                        # Calculate optimal date range with league-specific adjustments
                        start_date, end_date = dynamic_config.calculate_training_dates(league_ids=leagues)
                        
                        logger.info(f"📅 Dynamic training period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                        logger.info(f"📊 Training period: {(end_date - start_date).days} days ({(end_date - start_date).days / 30.44:.1f} months)")
                        logger.info(f"🏆 League-specific adjustments applied for leagues: {leagues}")
                        
                        features_df = load_features_from_database(db_session, leagues, start_date, end_date)
                        
                        # Validate minimum sample requirements with fallback
                        sample_count = len(features_df)
                        logger.info(f"📈 Initial sample count: {sample_count}")
                        
                        if sample_count < dynamic_config.min_training_samples:
                            logger.warning(f"⚠️ Insufficient samples ({sample_count} < {dynamic_config.min_training_samples})")
                            logger.info("🔄 Extending training period to meet minimum requirements...")
                            
                            # Extend to maximum period
                            extended_months = min(dynamic_config.max_months_back, dynamic_config.default_months_back + 12)
                            extended_start = end_date - timedelta(days=extended_months * 30.44)
                            
                            logger.info(f"📅 Extended training period: {extended_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                            
                            features_df = load_features_from_database(db_session, leagues, extended_start, end_date)
                            sample_count = len(features_df)
                            logger.info(f"📈 Extended sample count: {sample_count}")
                            
                            if sample_count < dynamic_config.min_training_samples:
                                logger.error(f"❌ Still insufficient samples after extension ({sample_count} < {dynamic_config.min_training_samples})")
                                logger.error("💡 Consider: reducing min_training_samples, adding more leagues, or checking data availability")
                                raise SystemExit(1)
                    
                    if features_df.empty:
                        logger.error("❌ No features were successfully built")
                        raise SystemExit(1)
                        
                except Exception as e:
                    logger.error(f"❌ Database feature building failed: {e}")
                    logger.error(f"🔍 Error details: {traceback.format_exc()}")
                    raise SystemExit(1)
            
            # Prepare training data with validation
            with timer("🎯 Preparing training matrices"):
                try:
                    if args.use_enhanced_features:
                        # Use enhanced feature preparation
                        X_df, y_total_series = training_engine.prepare_enhanced_features(features_df)
                        X = X_df.values
                        y_total = y_total_series.values
                        y_over25 = (y_total > 2.5).astype(int)  # Calculate over 2.5 from total goals
                        feature_cols = training_engine.feature_columns
                        
                        logger.info(f"🚀 Using {len(feature_cols)} enhanced features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
                    else:
                        # Use traditional feature preparation
                        excluded_cols = {
                            'fixture_id', 'match_date', 'league_id', 'home_team_id', 'away_team_id', 
                            'total_goals', 'over_2_5', 'home_score', 'away_score', 'match_result',
                            'weather_condition', 'computation_source', 'features_computed_at',
                            'data_quality_score', 'home_form_last_5_games', 'away_form_last_5_games',
                            'h2h_last_updated', 'id'
                        }
                        feature_cols = [col for col in features_df.columns if col not in excluded_cols]
                        
                        logger.info(f"🔢 Using {len(feature_cols)} database features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
                        
                        # Convert all feature columns to numeric, replacing non-numeric values with 0
                        for col in feature_cols:
                            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
                        
                        logger.info(f"✅ Converted all features to numeric types")
                        
                        # Keep DataFrame format for StandardScaler to preserve feature names
                        X_df = features_df[feature_cols]
                        X = X_df.values  # Convert to numpy for compatibility with existing code
                        y_total = features_df['total_goals'].values
                        y_over25 = features_df['over_2_5'].values
                    
                    # Calculate recency weights for training samples
                    if args.use_enhanced_features:
                        # For enhanced features, use uniform weights for now
                        sample_weights = np.ones(len(y_total))
                        logger.info("⚖️ Using uniform sample weights for enhanced features")
                    else:
                        # Apply recency weighting for database features
                        config = get_config()
                        dynamic_config = config.dynamic_training
                        if dynamic_config.enable_recency_weighting and 'match_date' in features_df.columns:
                            match_dates = pd.to_datetime(features_df['match_date']).tolist()
                            sample_weights = dynamic_config.get_recency_weights(match_dates)
                            sample_weights = np.array(sample_weights)
                            logger.info(f"⚖️ Recency weighting enabled: weights range {sample_weights.min():.3f} to {sample_weights.max():.3f}")
                            logger.info(f"📊 Recent matches (last 3 months) have {sample_weights[-int(len(sample_weights)*0.1):].mean():.2f}x average weight")
                        else:
                            sample_weights = np.ones(len(y_total))
                            logger.info("⚖️ Using uniform sample weights")
                    
                    # Data validation
                    if np.any(np.isnan(X)):
                        logger.warning("⚠️ NaN values detected in feature matrix, filling with zeros")
                        X = np.nan_to_num(X, nan=0.0)
                        
                    logger.info(f"✅ Training matrices prepared: X{X.shape}, y_total{y_total.shape}, y_over25{y_over25.shape}, weights{sample_weights.shape}")
                    
                except Exception as e:
                    logger.error(f"❌ Data preparation failed: {e}")
                    raise SystemExit(1)
            
            # Cross-validation with detailed reporting
            with timer(f"🔄 Performing {args.cv_splits}-fold time series cross-validation"):
                try:
                    tscv = TimeSeriesSplit(n_splits=args.cv_splits)
                    cv_scores_reg = []
                    cv_scores_clf = []
                    
                    logger.info(f"📊 Starting cross-validation with {args.cv_splits} splits...")
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        console=console
                    ) as progress:
                        cv_task = progress.add_task("Cross-validation", total=args.cv_splits)
                        
                        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                            progress.update(cv_task, description=f"Fold {fold + 1}/{args.cv_splits}")
                            
                            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                            y_total_train, y_total_val = y_total[train_idx], y_total[val_idx]
                            y_over25_train, y_over25_val = y_over25[train_idx], y_over25[val_idx]
                            weights_train = sample_weights[train_idx] if sample_weights is not None else None
                            
                            # Train fold models
                            # Create DataFrame for training fold to preserve feature names
                            X_train_df = pd.DataFrame(X_train_fold, columns=feature_cols)
                            reg_fold, clf_fold, scaler_fold = train_models(X_train_fold, y_total_train, y_over25_train, weights_train, X_train_df)
                            
                            # Validate
                            X_val_scaled = scaler_fold.transform(X_val_fold)
                            
                            # Regression metrics
                            reg_pred = reg_fold.predict(X_val_scaled)
                            reg_mae = mean_absolute_error(y_total_val, reg_pred)
                            cv_scores_reg.append(reg_mae)
                            
                            # Classification metrics
                            clf_proba = clf_fold.predict_proba(X_val_scaled)[:, 1]
                            clf_brier = brier_score_loss(y_over25_val, clf_proba)
                            cv_scores_clf.append(clf_brier)
                            
                            logger.debug(f"Fold {fold + 1}: Regression MAE={reg_mae:.3f}, Classification Brier={clf_brier:.3f}")
                            progress.advance(cv_task)
                    
                    # Report cross-validation results
                    logger.info(f"📊 Cross-validation results:")
                    logger.info(f"   Regression MAE: {np.mean(cv_scores_reg):.3f} ± {np.std(cv_scores_reg):.3f}")
                    logger.info(f"   Classification Brier: {np.mean(cv_scores_clf):.3f} ± {np.std(cv_scores_clf):.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Cross-validation failed: {e}")
                    logger.error(f"🔍 Error details: {traceback.format_exc()}")
                    # Continue with training even if CV fails
                    logger.warning("⚠️ Continuing with final model training despite CV failure")
        
            # Train final models on all data
            with timer("🎯 Training final models on complete dataset"):
                try:
                    regressor, classifier, scaler = train_models(X, y_total, y_over25, sample_weights, X_df)
                    logger.info("✅ Final models trained successfully")
                except Exception as e:
                    logger.error(f"❌ Final model training failed: {e}")
                    logger.error(f"🔍 Error details: {traceback.format_exc()}")
                    raise SystemExit(1)
            
            # Save models with comprehensive metadata
            with timer("💾 Saving trained models"):
                try:
                    import pickle
                    models_dir = Path('models')
                    models_dir.mkdir(exist_ok=True)
                    
                    # Generate timestamp for model files
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Prepare model metadata
                    training_metadata = {
                        'training_date': datetime.now().isoformat(),
                        'training_samples': len(features_df),
                        'feature_count': len(feature_cols),
                        'feature_columns': feature_cols,
                        'leagues': leagues,
                        'cv_splits': args.cv_splits,
                        'min_samples': dynamic_config.min_training_samples,
                        'data_date_range': {
                            'start': str(features_df['match_date'].min()) if 'match_date' in features_df.columns else 'unknown',
                            'end': str(features_df['match_date'].max()) if 'match_date' in features_df.columns else 'unknown'
                        },
                        'target_statistics': {
                            'total_goals_mean': float(np.mean(y_total)),
                            'total_goals_std': float(np.std(y_total)),
                            'over_2_5_rate': float(np.mean(y_over25))
                        }
                    }
                    
                    # Save individual model components with timestamped filenames
                    regressor_path = models_dir / f'goal_regressor_{timestamp}.joblib'
                    classifier_path = models_dir / f'over25_classifier_{timestamp}.joblib'
                    scaler_path = models_dir / f'feature_scaler_{timestamp}.joblib'
                    metadata_path = models_dir / f'metadata_{timestamp}.json'
                    
                    # Save models using joblib for compatibility
                    joblib.dump(regressor, regressor_path)
                    joblib.dump(classifier, classifier_path)
                    joblib.dump(scaler, scaler_path)
                    
                    # Save metadata as JSON
                    with open(metadata_path, 'w') as f:
                        json.dump(training_metadata, f, indent=2)
                    
                    # Calculate total file size
                    total_size = sum([
                        regressor_path.stat().st_size,
                        classifier_path.stat().st_size,
                        scaler_path.stat().st_size,
                        metadata_path.stat().st_size
                    ]) / (1024 * 1024)  # MB
                    
                    logger.info(f"💾 Models saved successfully with timestamp {timestamp} ({total_size:.2f} MB total)")
                    logger.info(f"   📊 Regressor: {regressor_path.name}")
                    logger.info(f"   🎯 Classifier: {classifier_path.name}")
                    logger.info(f"   ⚖️ Scaler: {scaler_path.name}")
                    logger.info(f"   📋 Metadata: {metadata_path.name}")
                    
                    # Also save legacy format for backward compatibility
                    legacy_model_data = {
                        'regressor': regressor,
                        'classifier': classifier,
                        'scaler': scaler,
                        'feature_columns': feature_cols,
                        'metadata': training_metadata
                    }
                    
                    legacy_model_path = models_dir / 'models.pkl'
                    with open(legacy_model_path, 'wb') as f:
                        pickle.dump(legacy_model_data, f)
                    
                    legacy_metadata_path = models_dir / 'training_metadata.json'
                    with open(legacy_metadata_path, 'w') as f:
                        json.dump(training_metadata, f, indent=2)
                    
                    logger.info(f"📋 Legacy format also saved for backward compatibility")
                    
                except Exception as e:
                    logger.error(f"❌ Model saving failed: {e}")
                    logger.error(f"🔍 Error details: {traceback.format_exc()}")
                    raise SystemExit(1)
            
            # Train enhanced predictors for each league
            with timer(f"🚀 Training enhanced predictors for {len(leagues)} leagues"):
                successful_leagues = []
                failed_leagues = []
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    predictor_task = progress.add_task("Enhanced predictors", total=len(leagues))
                    
                    for league_id in leagues:
                        progress.update(predictor_task, description=f"League {league_id}")
                        
                        try:
                            logger.info(f"🏆 Training enhanced predictor for league {league_id}...")
                            predictor = EnhancedGoalPredictor(config)
                            predictor.train_models(league_id)
                            successful_leagues.append(league_id)
                            logger.info(f"✅ Enhanced predictor trained for league {league_id}")
                            
                        except Exception as e:
                            failed_leagues.append(league_id)
                            logger.error(f"❌ Failed to train enhanced predictor for league {league_id}: {e}")
                            logger.debug(f"🔍 League {league_id} error details: {traceback.format_exc()}")
                        
                        progress.advance(predictor_task)
                
                # Report enhanced predictor results
                logger.info(f"🎯 Enhanced predictor training completed:")
                logger.info(f"   ✅ Successful: {len(successful_leagues)} leagues {successful_leagues}")
                if failed_leagues:
                    logger.warning(f"   ❌ Failed: {len(failed_leagues)} leagues {failed_leagues}")
            
            # Initialize and run calibration monitoring
            with timer("📊 Running calibration monitoring"):
                try:
                    logger.info("🔧 Initializing calibration monitor...")
                    calibration_monitor = CalibrationMonitor(config)
                    
                    logger.info("📈 Running monthly calibration analysis...")
                    calibration_monitor.run_monthly_calibration()
                    
                    logger.info("✅ Calibration monitoring completed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Calibration monitoring failed: {e}")
                    logger.error(f"🔍 Calibration error details: {traceback.format_exc()}")
                    logger.warning("⚠️ Training completed despite calibration failure")
            
            # Final success summary
            total_time = time.time() - start_time
            
            # Create success summary table
            summary_table = Table(title="🎉 Training Pipeline Summary", border_style="green")
            summary_table.add_column("Component", style="cyan")
            summary_table.add_column("Status", style="green")
            summary_table.add_column("Details", style="white")
            
            summary_table.add_row("📊 Training Data", "✅ Success", f"{len(features_df)} samples")
            summary_table.add_row("🔢 Features", "✅ Success", f"{len(feature_cols)} features")
            summary_table.add_row("🎯 Models", "✅ Success", "Regression + Classification")
            summary_table.add_row("🏆 Enhanced Predictors", "✅ Success", f"{len(successful_leagues)}/{len(leagues)} leagues")
            summary_table.add_row("⏱️ Total Time", "✅ Complete", f"{total_time:.1f} seconds")
            
            console.print(summary_table)
            
            rprint(Panel.fit(
                "[bold green]🎉 FormFinder Training Pipeline Completed Successfully![/bold green]\n"
                f"[dim]Models ready for prediction • Total time: {total_time:.1f}s[/dim]",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Training interrupted by user (Ctrl+C)")
            raise SystemExit(130)  # Standard exit code for Ctrl+C
            
        except SystemExit:
            raise  # Re-raise SystemExit without modification
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in training pipeline: {e}")
            logger.error(f"🔍 Full error details: {traceback.format_exc()}")
            raise SystemExit(1)
            
    except Exception as e:
        logger.error(f"❌ Critical error during initialization: {e}")
        logger.error(f"🔍 Error details: {traceback.format_exc()}")
        raise SystemExit(1)
        
    finally:
        # Cleanup resources
        if db_session:
            try:
                db_session.close()
                logger.info("🔒 Database session closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing database session: {e}")
        
        if engine:
            try:
                engine.dispose()
                logger.info("🔒 Database engine disposed")
            except Exception as e:
                logger.warning(f"⚠️ Error disposing database engine: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user")
        sys.exit(130)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        logger.error(f"❌ Unhandled error: {e}")
        logger.error(f"🔍 Error details: {traceback.format_exc()}")
        sys.exit(1)