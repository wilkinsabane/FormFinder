"""Database Feature Engine for FormFinder2.

This module implements the database-only feature loading approach as specified in the
Data Collection and Training Separation PRD. It replaces API-dependent feature building
with fast database queries for pre-computed features.

Author: FormFinder2 Team
Created: 2025-01-01
Purpose: Eliminate API dependency during model training
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()
logger = logging.getLogger(__name__)


class DatabaseFeatureEngine:
    """Retrieves pre-computed features from database only.
    
    This class implements the database-centric approach for feature loading,
    eliminating API dependencies during training as specified in the PRD.
    """
    
    def __init__(self, db_session: Session):
        """Initialize the database feature engine.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        
        # Feature quality thresholds
        self.min_quality_score = 0.7
        self.max_missing_features_pct = 0.15
        
        # Feature column mappings
        self.feature_columns = {
            # Home team form features
            'home_avg_goals_scored': 'home_avg_goals_scored',
            'home_avg_goals_conceded': 'home_avg_goals_conceded', 
            'home_avg_goals_scored_home': 'home_avg_goals_scored_home',
            'home_avg_goals_conceded_home': 'home_avg_goals_conceded_home',
            'home_wins_last_5': 'home_wins_last_5',
            'home_draws_last_5': 'home_draws_last_5',
            'home_losses_last_5': 'home_losses_last_5',
            'home_goals_for_last_5': 'home_goals_for_last_5',
            'home_goals_against_last_5': 'home_goals_against_last_5',
            
            # Away team form features
            'away_avg_goals_scored': 'away_avg_goals_scored',
            'away_avg_goals_conceded': 'away_avg_goals_conceded',
            'away_avg_goals_scored_away': 'away_avg_goals_scored_away', 
            'away_avg_goals_conceded_away': 'away_avg_goals_conceded_away',
            'away_wins_last_5': 'away_wins_last_5',
            'away_draws_last_5': 'away_draws_last_5',
            'away_losses_last_5': 'away_losses_last_5',
            'away_goals_for_last_5': 'away_goals_for_last_5',
            'away_goals_against_last_5': 'away_goals_against_last_5',
            
            # H2H features - using actual columns that exist in database
            'h2h_total_matches': 'h2h_total_matches',
            'h2h_avg_goals': 'h2h_avg_goals',
            'h2h_home_wins': 'h2h_home_wins',
            'h2h_away_wins': 'h2h_away_wins',
            'h2h_total_goals': 'h2h_total_goals',
            'h2h_competitiveness': 'h2h_competitiveness',
            
            # Preview and weather features
            'excitement_rating': 'excitement_rating',
            'weather_temp_c': 'weather_temp_c',
            'weather_humidity': 'weather_humidity',
            'weather_wind_speed': 'weather_wind_speed',
            'weather_precipitation': 'weather_precipitation',
        }
        
    def load_training_features(self, 
                             leagues: List[int], 
                             start_date: datetime, 
                             end_date: datetime,
                             min_quality_score: Optional[float] = None) -> pd.DataFrame:
        """Load pre-computed features for training.
        
        Args:
            leagues: List of league IDs to include
            start_date: Start date for training data
            end_date: End date for training data
            min_quality_score: Minimum data quality score (default: 0.7)
            
        Returns:
            DataFrame with pre-computed features and targets
            
        Raises:
            SQLAlchemyError: If database query fails
            ValueError: If no data found or data quality is poor
        """
        if min_quality_score is None:
            min_quality_score = self.min_quality_score
            
        self.logger.info(f"🔍 Loading training features for leagues {leagues}")
        self.logger.info(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"📊 Minimum quality score: {min_quality_score}")
        
        start_time = time.time()
        
        try:
            # Query pre-computed features
            query = text("""
                SELECT 
                    pcf.fixture_id,
                    pcf.home_team_id,
                    pcf.away_team_id,
                    pcf.match_date,
                    pcf.league_id,
                    
                    -- Home team form features
                    pcf.home_avg_goals_scored,
                    pcf.home_avg_goals_conceded,
                    pcf.home_avg_goals_scored_home,
                    pcf.home_avg_goals_conceded_home,
                    pcf.home_wins_last_5,
                    pcf.home_draws_last_5,
                    pcf.home_losses_last_5,
                    pcf.home_goals_for_last_5,
                    pcf.home_goals_against_last_5,
                    
                    -- Away team form features
                    pcf.away_avg_goals_scored,
                    pcf.away_avg_goals_conceded,
                    pcf.away_avg_goals_scored_away,
                    pcf.away_avg_goals_conceded_away,
                    pcf.away_wins_last_5,
                    pcf.away_draws_last_5,
                    pcf.away_losses_last_5,
                    pcf.away_goals_for_last_5,
                    pcf.away_goals_against_last_5,
                    
                    -- H2H features (using actual columns that exist)
                    pcf.h2h_total_matches,
                    pcf.h2h_avg_goals,
                    pcf.h2h_home_wins,
                    pcf.h2h_away_wins,
                    pcf.h2h_total_goals,
                    pcf.h2h_competitiveness,
                    
                    -- Preview and weather features
                    pcf.excitement_rating,
                    pcf.weather_temp_c,
                    pcf.weather_humidity,
                    pcf.weather_wind_speed,
                    pcf.weather_precipitation,
                    pcf.weather_condition,
                    
                    -- Target variables
                    pcf.total_goals,
                    pcf.over_2_5,
                    pcf.home_score,
                    pcf.away_score,
                    pcf.match_result,
                    
                    -- Metadata
                    pcf.data_quality_score,
                    pcf.features_computed_at,
                    pcf.computation_source
                    
                FROM pre_computed_features pcf
                WHERE pcf.league_id = ANY(:leagues)
                    AND pcf.match_date >= :start_date
                    AND pcf.match_date <= :end_date
                    AND pcf.data_quality_score >= :min_quality_score
                    AND pcf.total_goals IS NOT NULL
                    AND pcf.home_score IS NOT NULL
                    AND pcf.away_score IS NOT NULL
                ORDER BY pcf.match_date ASC
            """)
            
            result = self.db_session.execute(query, {
                'leagues': leagues,
                'start_date': start_date,
                'end_date': end_date,
                'min_quality_score': min_quality_score
            })
            
            df = pd.DataFrame(result.mappings().all())
            
            if df.empty:
                raise ValueError(f"No training data found for leagues {leagues} between {start_date.date()} and {end_date.date()}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ Loaded {len(df)} training samples in {elapsed_time:.2f}s")
            
            # Validate feature completeness
            quality_report = self.validate_feature_completeness(df)
            self._log_quality_report(quality_report)
            
            # Check if data quality meets requirements
            if quality_report['missing_features_pct'] > self.max_missing_features_pct:
                self.logger.warning(f"⚠️ High missing feature rate: {quality_report['missing_features_pct']:.2%}")
                
            return df
            
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Database error loading training features: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error loading training features: {e}")
            raise
    
    def validate_feature_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that all required features are present.
        
        Args:
            df: DataFrame with features to validate
            
        Returns:
            Dictionary with validation results
        """
        if df.empty:
            return {
                'total_samples': 0,
                'missing_features_pct': 1.0,
                'quality_score_avg': 0.0,
                'missing_columns': [],
                'validation_passed': False
            }
        
        # Check for missing feature columns
        expected_columns = list(self.feature_columns.values())
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        # Calculate missing data percentage
        total_cells = len(df) * len(expected_columns)
        missing_cells = df[expected_columns].isnull().sum().sum() if not missing_columns else total_cells
        missing_pct = missing_cells / total_cells if total_cells > 0 else 1.0
        
        # Calculate average quality score
        avg_quality = df['data_quality_score'].mean() if 'data_quality_score' in df.columns else 0.0
        
        # Check validation status
        validation_passed = (
            len(missing_columns) == 0 and 
            missing_pct <= self.max_missing_features_pct and
            avg_quality >= self.min_quality_score
        )
        
        return {
            'total_samples': len(df),
            'missing_features_pct': missing_pct,
            'quality_score_avg': avg_quality,
            'missing_columns': missing_columns,
            'validation_passed': validation_passed,
            'date_range': {
                'start': df['match_date'].min() if 'match_date' in df.columns else None,
                'end': df['match_date'].max() if 'match_date' in df.columns else None
            },
            'league_distribution': df['league_id'].value_counts().to_dict() if 'league_id' in df.columns else {}
        }
    
    def get_feature_quality_report(self, fixture_ids: List[int]) -> Dict[str, Any]:
        """Generate data quality report for specific fixtures.
        
        Args:
            fixture_ids: List of fixture IDs to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            query = text("""
                SELECT 
                    fixture_id,
                    data_quality_score,
                    features_computed_at,
                    computation_source,
                    CASE 
                        WHEN total_goals IS NULL THEN 'missing_targets'
                        WHEN data_quality_score < 0.7 THEN 'low_quality'
                        WHEN data_quality_score < 0.9 THEN 'medium_quality'
                        ELSE 'high_quality'
                    END as quality_category
                FROM pre_computed_features
                WHERE fixture_id = ANY(:fixture_ids)
            """)
            
            result = self.db_session.execute(query, {'fixture_ids': fixture_ids})
            df = pd.DataFrame(result.mappings().all())
            
            if df.empty:
                return {
                    'total_fixtures': len(fixture_ids),
                    'found_fixtures': 0,
                    'missing_fixtures': len(fixture_ids),
                    'quality_distribution': {},
                    'avg_quality_score': 0.0
                }
            
            quality_dist = df['quality_category'].value_counts().to_dict()
            
            return {
                'total_fixtures': len(fixture_ids),
                'found_fixtures': len(df),
                'missing_fixtures': len(fixture_ids) - len(df),
                'quality_distribution': quality_dist,
                'avg_quality_score': df['data_quality_score'].mean(),
                'computation_sources': df['computation_source'].value_counts().to_dict(),
                'oldest_computation': df['features_computed_at'].min(),
                'newest_computation': df['features_computed_at'].max()
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Database error generating quality report: {e}")
            raise
    
    def load_prediction_features(self, fixture_ids: List[int]) -> pd.DataFrame:
        """Load pre-computed features for prediction (upcoming matches).
        
        Args:
            fixture_ids: List of fixture IDs to load features for
            
        Returns:
            DataFrame with features for prediction
        """
        self.logger.info(f"🔮 Loading prediction features for {len(fixture_ids)} fixtures")
        
        try:
            query = text("""
                SELECT 
                    pcf.fixture_id,
                    pcf.home_team_id,
                    pcf.away_team_id,
                    pcf.match_date,
                    pcf.league_id,
                    
                    -- All feature columns (same as training)
                    pcf.home_avg_goals_scored,
                    pcf.home_avg_goals_conceded,
                    pcf.home_avg_goals_scored_home,
                    pcf.home_avg_goals_conceded_home,
                    pcf.home_wins_last_5,
                    pcf.home_draws_last_5,
                    pcf.home_losses_last_5,
                    pcf.home_goals_for_last_5,
                    pcf.home_goals_against_last_5,
                    
                    pcf.away_avg_goals_scored,
                    pcf.away_avg_goals_conceded,
                    pcf.away_avg_goals_scored_away,
                    pcf.away_avg_goals_conceded_away,
                    pcf.away_wins_last_5,
                    pcf.away_draws_last_5,
                    pcf.away_losses_last_5,
                    pcf.away_goals_for_last_5,
                    pcf.away_goals_against_last_5,
                    
                    pcf.h2h_total_matches,
                    pcf.h2h_avg_goals,
                    pcf.h2h_home_wins,
                    pcf.h2h_away_wins,
                    
                    pcf.excitement_rating,
                    pcf.weather_temp_c,
                    pcf.weather_humidity,
                    pcf.weather_wind_speed,
                    pcf.weather_precipitation,
                    pcf.weather_condition,
                    
                    -- Metadata
                    pcf.data_quality_score,
                    pcf.features_computed_at
                    
                FROM pre_computed_features pcf
                WHERE pcf.fixture_id = ANY(:fixture_ids)
                ORDER BY pcf.match_date ASC
            """)
            
            result = self.db_session.execute(query, {'fixture_ids': fixture_ids})
            df = pd.DataFrame(result.mappings().all())
            
            self.logger.info(f"✅ Loaded features for {len(df)} fixtures")
            
            # Check for missing fixtures
            found_ids = set(df['fixture_id'].tolist()) if not df.empty else set()
            missing_ids = set(fixture_ids) - found_ids
            
            if missing_ids:
                self.logger.warning(f"⚠️ Missing features for {len(missing_ids)} fixtures: {list(missing_ids)[:10]}...")
            
            return df
            
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Database error loading prediction features: {e}")
            raise
    
    def _log_quality_report(self, quality_report: Dict[str, Any]) -> None:
        """Log quality report in a formatted way."""
        table = Table(title="📊 Feature Quality Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Samples", str(quality_report['total_samples']))
        table.add_row("Missing Features %", f"{quality_report['missing_features_pct']:.2%}")
        table.add_row("Avg Quality Score", f"{quality_report['quality_score_avg']:.3f}")
        table.add_row("Validation Passed", "✅" if quality_report['validation_passed'] else "❌")
        
        if quality_report['missing_columns']:
            table.add_row("Missing Columns", ", ".join(quality_report['missing_columns'][:5]))
        
        console.print(table)
        
        # Log league distribution
        if quality_report['league_distribution']:
            league_table = Table(title="🏆 League Distribution")
            league_table.add_column("League ID", style="cyan")
            league_table.add_column("Samples", style="green")
            
            for league_id, count in quality_report['league_distribution'].items():
                league_table.add_row(str(league_id), str(count))
            
            console.print(league_table)


def load_features_from_database(db_session: Session, 
                               leagues: List[int], 
                               start_date: datetime, 
                               end_date: datetime) -> pd.DataFrame:
    """Load all pre-computed features from database.
    
    This replaces the previous build_features() function that made API calls.
    
    Args:
        db_session: Database session
        leagues: List of league IDs
        start_date: Start date for training data
        end_date: End date for training data
        
    Returns:
        DataFrame with pre-computed features
    """
    feature_engine = DatabaseFeatureEngine(db_session)
    
    # Load pre-computed features
    features_df = feature_engine.load_training_features(leagues, start_date, end_date)
    
    # Validate feature completeness
    quality_report = feature_engine.validate_feature_completeness(features_df)
    
    if quality_report['missing_features_pct'] > 0.1:  # More than 10% missing
        logger.warning(f"⚠️ High missing feature rate: {quality_report['missing_features_pct']:.2%}")
    
    return features_df