"""Enhanced Training Engine for FormFinder2

This module implements the pure training layer that reads pre-computed features
from the database without making any API calls, as outlined in the PRD.

Author: FormFinder2 Team
Created: 2025-01-01
Purpose: Database-only training engine for improved performance
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import psycopg2
from psycopg2.extras import RealDictCursor

from .config import load_config, get_config
from .exceptions import TrainingError, DataQualityError, ModelValidationError
from enhanced_predictor import EnhancedGoalPredictor


class TrainingEngine:
    """Enhanced training engine that reads only from database."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the training engine.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            self.config = get_config()
        else:
            self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.db_connection = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = []
        self.model_metadata = {}
        
        # Training configuration
        self.training_config = self.config.training
        
        # Initialize enhanced predictor for feature extraction
        self.enhanced_predictor = None
        
    def connect_to_database(self) -> None:
        """Establish database connection."""
        try:
            db_config = self.config.database.postgresql
            self.db_connection = psycopg2.connect(
                host=db_config.host,
                port=db_config.port,
                database=db_config.database,
                user=db_config.username,
                password=db_config.password
            )
            self.logger.info("Database connection established")
        except Exception as e:
            raise TrainingError(f"Failed to connect to database: {str(e)}")
    
    def close_database_connection(self) -> None:
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.logger.info("Database connection closed")
    
    def initialize_enhanced_predictor(self) -> None:
        """Initialize the enhanced predictor for feature extraction."""
        try:
            self.enhanced_predictor = EnhancedGoalPredictor()
            self.logger.info("Enhanced predictor initialized successfully")
        except Exception as e:
            raise TrainingError(f"Failed to initialize enhanced predictor: {str(e)}")
    
    def load_training_data(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          leagues: Optional[List[int]] = None,
                          min_quality_score: float = 0.8) -> pd.DataFrame:
        """Load pre-computed features from database for training.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            leagues: List of league IDs to include
            min_quality_score: Minimum quality score for features
            
        Returns:
            DataFrame with training features and targets
        """
        if not self.db_connection:
            self.connect_to_database()
        
        # Build query conditions
        conditions = ["pcf.computation_status = 'completed'"]
        params = []
        
        if start_date:
            conditions.append("f.match_date >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("f.match_date <= %s")
            params.append(end_date)
        
        if leagues:
            conditions.append("f.league_id = ANY(%s)")
            params.append(leagues)
        
        conditions.append("pcf.quality_score >= %s")
        params.append(min_quality_score)
        
        # Only include completed matches with actual goals
        conditions.append("f.status = 'finished'")
        conditions.append("pcf.actual_total_goals IS NOT NULL")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT 
            pcf.fixture_id,
            pcf.home_team_id,
            pcf.away_team_id,
            pcf.league_id,
            f.match_date,
            
            -- Form features
            pcf.home_form_goals_scored,
            pcf.home_form_goals_conceded,
            pcf.home_form_points,
            pcf.home_form_wins,
            pcf.home_form_draws,
            pcf.home_form_losses,
            pcf.home_form_clean_sheets,
            pcf.home_form_btts,
            pcf.home_form_over_2_5,
            pcf.home_form_avg_goals,
            
            pcf.away_form_goals_scored,
            pcf.away_form_goals_conceded,
            pcf.away_form_points,
            pcf.away_form_wins,
            pcf.away_form_draws,
            pcf.away_form_losses,
            pcf.away_form_clean_sheets,
            pcf.away_form_btts,
            pcf.away_form_over_2_5,
            pcf.away_form_avg_goals,
            
            -- H2H features
            pcf.h2h_total_matches,
            pcf.h2h_home_wins,
            pcf.h2h_away_wins,
            pcf.h2h_draws,
            pcf.h2h_avg_total_goals,
            pcf.h2h_over_2_5_percentage,
            pcf.h2h_btts_percentage,
            pcf.h2h_home_avg_goals,
            pcf.h2h_away_avg_goals,
            
            -- Preview features
            pcf.preview_predicted_goals,
            pcf.preview_confidence,
            pcf.preview_over_2_5_prob,
            pcf.preview_btts_prob,
            
            -- Weather features
            pcf.weather_temperature,
            pcf.weather_humidity,
            pcf.weather_wind_speed,
            pcf.weather_precipitation,
            
            -- Additional derived features
            pcf.home_attack_strength,
            pcf.home_defense_strength,
            pcf.away_attack_strength,
            pcf.away_defense_strength,
            pcf.goal_expectancy,
            pcf.form_momentum_home,
            pcf.form_momentum_away,
            
            -- Quality metrics
            pcf.quality_score,
            pcf.feature_completeness,
            
            -- Target variable
            pcf.actual_total_goals,
            
            -- Additional targets for multi-task learning
            CASE WHEN pcf.actual_total_goals > 2.5 THEN TRUE ELSE FALSE END as over_2_5,
            CASE WHEN pcf.actual_home_goals > 0 AND pcf.actual_away_goals > 0 THEN TRUE ELSE FALSE END as btts
            
        FROM pre_computed_features pcf
        JOIN fixtures f ON pcf.fixture_id = f.id
        WHERE {where_clause}
        ORDER BY f.match_date DESC
        """
        
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
            
            if not results:
                raise DataQualityError("No training data found matching criteria")
            
            df = pd.DataFrame(results)
            self.logger.info(f"Loaded {len(df)} training samples from database")
            
            # Validate data quality
            self._validate_training_data(df)
            
            return df
            
        except Exception as e:
            raise TrainingError(f"Failed to load training data: {str(e)}")
    
    def _validate_training_data(self, df: pd.DataFrame) -> None:
        """Validate training data quality.
        
        Args:
            df: Training DataFrame
        """
        # Check minimum samples
        min_samples = self.training_config.data.min_training_samples
        if len(df) < min_samples:
            raise DataQualityError(
                f"Insufficient training samples: {len(df)} < {min_samples}"
            )
        
        # Check for missing values in critical features
        critical_features = [
            'actual_total_goals', 'home_form_avg_goals', 'away_form_avg_goals',
            'h2h_avg_total_goals', 'quality_score'
        ]
        
        for feature in critical_features:
            if feature in df.columns:
                missing_pct = df[feature].isnull().sum() / len(df)
                if missing_pct > self.training_config.features.max_missing_percentage:
                    raise DataQualityError(
                        f"Too many missing values in {feature}: {missing_pct:.2%}"
                    )
        
        # Check target variable distribution
        target_col = 'actual_total_goals'
        if target_col in df.columns:
            target_stats = df[target_col].describe()
            if target_stats['std'] < 0.1:  # Very low variance
                raise DataQualityError("Target variable has very low variance")
        
        self.logger.info("Training data validation passed")
    
    def load_enhanced_training_data(self, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   leagues: Optional[List[int]] = None,
                                   min_quality_score: float = 0.8) -> pd.DataFrame:
        """Load training data using enhanced predictor features.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            leagues: List of league IDs to include
            min_quality_score: Minimum quality score for features
            
        Returns:
            DataFrame with enhanced features and targets
        """
        if not self.enhanced_predictor:
            self.initialize_enhanced_predictor()
        
        if not self.db_connection:
            self.connect_to_database()
        
        # Build query to get fixture IDs and targets
        conditions = ["f.status = 'finished'", "f.home_score IS NOT NULL"]
        params = []
        
        if start_date:
            conditions.append("f.match_date >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("f.match_date <= %s")
            params.append(end_date)
        
        if leagues:
            conditions.append("f.league_id = ANY(%s)")
            params.append(leagues)
        
        where_clause = " AND ".join(conditions)
        
        # Query to get fixture metadata and targets
        query = f"""
        SELECT 
            f.id as fixture_id,
            f.home_team_id,
            f.away_team_id,
            f.league_id,
            f.match_date,
            f.home_score + f.away_score as actual_total_goals,
            f.home_score as actual_home_goals,
            f.away_score as actual_away_goals,
            CASE WHEN f.home_score + f.away_score > 2.5 THEN TRUE ELSE FALSE END as over_2_5,
            CASE WHEN f.home_score > 0 AND f.away_score > 0 THEN TRUE ELSE FALSE END as btts
        FROM fixtures f
        WHERE {where_clause}
        ORDER BY f.match_date DESC
        """
        
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                fixture_results = cursor.fetchall()
            
            if not fixture_results:
                raise DataQualityError("No fixtures found matching criteria")
            
            self.logger.info(f"Found {len(fixture_results)} fixtures for feature extraction")
            
            # Extract enhanced features for each fixture
            enhanced_data = []
            successful_extractions = 0
            
            for i, fixture_row in enumerate(fixture_results):
                fixture_id = fixture_row['fixture_id']
                
                if i % 100 == 0:
                    self.logger.info(f"Processing fixture {i+1}/{len(fixture_results)}")
                
                # Extract enhanced features
                features = self.enhanced_predictor.extract_enhanced_features(fixture_id)
                
                if features:
                    # Add target variables and metadata
                    features.update({
                        'actual_total_goals': fixture_row['actual_total_goals'],
                        'actual_home_goals': fixture_row['actual_home_goals'],
                        'actual_away_goals': fixture_row['actual_away_goals'],
                        'over_2_5': fixture_row['over_2_5'],
                        'btts': fixture_row['btts'],
                        'match_date': fixture_row['match_date']
                    })
                    enhanced_data.append(features)
                    successful_extractions += 1
                else:
                    self.logger.warning(f"Failed to extract features for fixture {fixture_id}")
            
            if not enhanced_data:
                raise DataQualityError("No enhanced features could be extracted")
            
            df = pd.DataFrame(enhanced_data)
            self.logger.info(f"Successfully extracted enhanced features for {successful_extractions}/{len(fixture_results)} fixtures")
            
            # Validate enhanced training data
            self._validate_enhanced_training_data(df)
            
            return df
            
        except Exception as e:
            raise TrainingError(f"Failed to load enhanced training data: {str(e)}")
    
    def _validate_enhanced_training_data(self, df: pd.DataFrame) -> None:
        """Validate enhanced training data quality.
        
        Args:
            df: Enhanced training DataFrame
        """
        # Check minimum samples
        min_samples = self.training_config.min_training_samples
        if len(df) < min_samples:
            raise DataQualityError(
                f"Insufficient enhanced training samples: {len(df)} < {min_samples}"
            )
        
        # Check for missing values in critical enhanced features
        critical_features = [
            'actual_total_goals', 'home_attack_strength', 'away_attack_strength',
            'home_xg', 'away_xg', 'fixture_id'
        ]
        
        for feature in critical_features:
            if feature in df.columns:
                missing_pct = df[feature].isnull().sum() / len(df)
                if missing_pct > 0.1:  # Allow 10% missing for enhanced features
                    self.logger.warning(
                        f"High missing values in enhanced feature {feature}: {missing_pct:.2%}"
                    )
        
        # Check target variable distribution
        target_col = 'actual_total_goals'
        if target_col in df.columns:
            target_stats = df[target_col].describe()
            if target_stats['std'] < 0.1:
                raise DataQualityError("Target variable has very low variance")
        
        self.logger.info(f"Enhanced training data validation passed for {len(df)} samples")
    
    def prepare_enhanced_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare enhanced features for training.
        
        Args:
            df: Enhanced training DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Define enhanced feature exclusions (metadata and target columns)
        exclude_columns = {
            'fixture_id', 'home_team_id', 'away_team_id', 'league_id', 'utc_date',
            'actual_total_goals', 'actual_home_goals', 'actual_away_goals',
            'over_2_5', 'btts'
        }
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle missing values for enhanced features
        features_df = df[feature_columns].copy()
        
        # Convert all columns to numeric, coercing errors to NaN
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Fill missing values based on enhanced feature patterns
        for col in features_df.columns:
            if 'strength' in col.lower():
                # For strength features, use 1.0 as neutral
                features_df[col] = features_df[col].fillna(1.0)
            elif 'xg' in col.lower():
                # For expected goals, use median
                features_df[col] = features_df[col].fillna(features_df[col].median())
            elif 'advantage' in col.lower() or 'diff' in col.lower():
                # For advantage/difference features, use 0 as neutral
                features_df[col] = features_df[col].fillna(0.0)
            elif 'markov' in col.lower():
                # For Markov features, use 0 as default
                features_df[col] = features_df[col].fillna(0.0)
            else:
                # For other features, use median or 0 if all NaN
                median_val = features_df[col].median()
                fill_val = median_val if not pd.isna(median_val) else 0.0
                features_df[col] = features_df[col].fillna(fill_val)
        
        # Store feature columns for later use
        self.feature_columns = features_df.columns.tolist()
        
        # Get target variable
        target = df['actual_total_goals']
        
        self.logger.info(f"Prepared {len(self.feature_columns)} enhanced features for training")
        self.logger.debug(f"Enhanced feature columns: {self.feature_columns[:10]}...")  # Log first 10 features
        
        return features_df, target
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training.
        
        Args:
            df: Raw training DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Define feature columns (exclude metadata and target columns)
        exclude_columns = {
            'fixture_id', 'home_team_id', 'away_team_id', 'league_id', 'utc_date',
            'actual_total_goals', 'actual_home_goals', 'actual_away_goals',
            'quality_score', 'feature_completeness', 'over_2_5', 'btts'
        }
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle missing values
        features_df = df[feature_columns].copy()
        
        # Fill missing values based on feature type
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                if 'percentage' in col.lower() or 'prob' in col.lower():
                    # For percentages/probabilities, use 0.5 as neutral
                    features_df[col] = features_df[col].fillna(0.5)
                elif 'goals' in col.lower() or 'avg' in col.lower():
                    # For goal-related features, use league average
                    features_df[col] = features_df[col].fillna(features_df[col].median())
                else:
                    # For other numeric features, use median
                    features_df[col] = features_df[col].fillna(features_df[col].median())
        
        # Create additional engineered features
        if self.training_config.features.enable_feature_engineering:
            features_df = self._engineer_features(features_df)
        
        # Store feature columns for later use
        self.feature_columns = features_df.columns.tolist()
        
        # Get target variable
        target = df['actual_total_goals']
        
        self.logger.info(f"Prepared {len(self.feature_columns)} features for training")
        
        return features_df, target
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features.
        
        Args:
            df: Features DataFrame
            
        Returns:
            Enhanced features DataFrame
        """
        df = df.copy()
        
        # Goal difference features
        if 'home_form_avg_goals' in df.columns and 'away_form_avg_goals' in df.columns:
            df['goal_difference_expectation'] = df['home_form_avg_goals'] - df['away_form_avg_goals']
        
        # Form strength ratios
        if 'home_form_points' in df.columns and 'away_form_points' in df.columns:
            df['form_points_ratio'] = df['home_form_points'] / (df['away_form_points'] + 1)
        
        # Attack vs Defense matchups
        if all(col in df.columns for col in ['home_attack_strength', 'away_defense_strength']):
            df['home_attack_vs_away_defense'] = df['home_attack_strength'] / (df['away_defense_strength'] + 0.1)
        
        if all(col in df.columns for col in ['away_attack_strength', 'home_defense_strength']):
            df['away_attack_vs_home_defense'] = df['away_attack_strength'] / (df['home_defense_strength'] + 0.1)
        
        # H2H momentum
        if 'h2h_home_wins' in df.columns and 'h2h_total_matches' in df.columns:
            df['h2h_home_advantage'] = df['h2h_home_wins'] / (df['h2h_total_matches'] + 1)
        
        # Weather impact on goals
        if 'weather_temperature' in df.columns:
            df['weather_goal_impact'] = np.where(
                df['weather_temperature'].between(15, 25), 1.1,  # Optimal temperature
                np.where(df['weather_temperature'] < 5, 0.9, 1.0)  # Cold weather
            )
        
        # Combined form momentum
        if 'form_momentum_home' in df.columns and 'form_momentum_away' in df.columns:
            df['combined_form_momentum'] = df['form_momentum_home'] + df['form_momentum_away']
        
        self.logger.info(f"Engineered {len(df.columns) - len(self.feature_columns)} additional features")
        
        return df
    
    def train_model(self, 
                   features: pd.DataFrame, 
                   target: pd.Series,
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the XGBoost model.
        
        Args:
            features: Training features
            target: Target variable
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, target, 
                test_size=validation_split,
                random_state=self.training_config.model.random_state,
                stratify=None  # For regression
            )
            
            # Scale features if enabled
            if self.training_config.features.enable_scaling:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Configure XGBoost parameters
            xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': self.training_config.xgboost.max_depth,
                'learning_rate': self.training_config.xgboost.learning_rate,
                'n_estimators': self.training_config.xgboost.n_estimators,
                'subsample': self.training_config.xgboost.subsample,
                'colsample_bytree': self.training_config.xgboost.colsample_bytree,
                'random_state': self.training_config.model.random_state,
                'n_jobs': -1
            }
            
            # Train model
            self.model = xgb.XGBRegressor(**xgb_params)
            
            # Enable early stopping if validation data is available
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.training_config.xgboost.early_stopping_rounds,
                verbose=False
            )
            
            # Make predictions
            train_pred = self.model.predict(X_train_scaled)
            val_pred = self.model.predict(X_val_scaled)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, train_pred)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            
            # Store model metadata
            self.model_metadata = {
                'training_date': datetime.now(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': len(self.feature_columns),
                'model_params': xgb_params,
                'feature_columns': self.feature_columns,
                'scaling_enabled': self.training_config.features.enable_scaling
            }
            
            results = {
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'feature_importance': feature_importance,
                'model_metadata': self.model_metadata,
                'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
            }
            
            # Validate model performance
            self._validate_model_performance(val_metrics)
            
            self.logger.info(f"Model training completed. Validation RMSE: {val_metrics['rmse']:.4f}")
            
            return results
            
        except Exception as e:
            raise TrainingError(f"Model training failed: {str(e)}")
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_prediction': np.mean(y_pred),
            'std_prediction': np.std(y_pred)
        }
    
    def _validate_model_performance(self, metrics: Dict[str, float]) -> None:
        """Validate model performance against thresholds.
        
        Args:
            metrics: Model performance metrics
        """
        # Check RMSE threshold
        max_rmse = self.training_config.validation.max_rmse
        if metrics['rmse'] > max_rmse:
            raise ModelValidationError(
                f"Model RMSE {metrics['rmse']:.4f} exceeds threshold {max_rmse}",
                metric_name='rmse',
                metric_value=metrics['rmse'],
                threshold=max_rmse
            )
        
        # Check R² threshold
        min_r2 = self.training_config.validation.min_r2
        if metrics['r2'] < min_r2:
            raise ModelValidationError(
                f"Model R² {metrics['r2']:.4f} below threshold {min_r2}",
                metric_name='r2',
                metric_value=metrics['r2'],
                threshold=min_r2
            )
        
        self.logger.info("Model performance validation passed")
    
    def save_model(self, model_path: str) -> None:
        """Save trained model and metadata.
        
        Args:
            model_path: Path to save the model
        """
        if not self.model:
            raise TrainingError("No trained model to save")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler if self.training_config.features.enable_scaling else None,
                'feature_columns': self.feature_columns,
                'metadata': self.model_metadata
            }
            
            joblib.dump(model_data, model_path)
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            raise TrainingError(f"Failed to save model: {str(e)}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model and metadata.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_columns = model_data['feature_columns']
            self.model_metadata = model_data['metadata']
            
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            raise TrainingError(f"Failed to load model: {str(e)}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            features: Features for prediction
            
        Returns:
            Predictions array
        """
        if not self.model:
            raise TrainingError("No trained model available for prediction")
        
        try:
            # Ensure features match training format
            features_aligned = features[self.feature_columns]
            
            # Apply scaling if used during training
            if self.scaler and self.training_config.features.enable_scaling:
                features_scaled = self.scaler.transform(features_aligned)
            else:
                features_scaled = features_aligned
            
            predictions = self.model.predict(features_scaled)
            
            return predictions
            
        except Exception as e:
            raise TrainingError(f"Prediction failed: {str(e)}")
    
    def cross_validate(self, 
                      features: pd.DataFrame, 
                      target: pd.Series,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation.
        
        Args:
            features: Training features
            target: Target variable
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        try:
            # Configure model for CV
            xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': self.training_config.xgboost.max_depth,
                'learning_rate': self.training_config.xgboost.learning_rate,
                'n_estimators': self.training_config.xgboost.n_estimators,
                'random_state': self.training_config.model.random_state,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**xgb_params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, features, target,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Convert to RMSE
            cv_rmse = np.sqrt(-cv_scores)
            
            results = {
                'cv_rmse_scores': cv_rmse,
                'cv_rmse_mean': np.mean(cv_rmse),
                'cv_rmse_std': np.std(cv_rmse),
                'cv_folds': cv_folds
            }
            
            self.logger.info(f"Cross-validation completed. Mean RMSE: {results['cv_rmse_mean']:.4f} ± {results['cv_rmse_std']:.4f}")
            
            return results
            
        except Exception as e:
            raise TrainingError(f"Cross-validation failed: {str(e)}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary.
        
        Returns:
            Training summary dictionary
        """
        if not self.model_metadata:
            return {"status": "No model trained"}
        
        return {
            "model_status": "trained" if self.model else "not_loaded",
            "training_date": self.model_metadata.get('training_date'),
            "training_samples": self.model_metadata.get('training_samples'),
            "validation_samples": self.model_metadata.get('validation_samples'),
            "feature_count": self.model_metadata.get('feature_count'),
            "scaling_enabled": self.model_metadata.get('scaling_enabled'),
            "feature_columns": self.feature_columns[:10] if self.feature_columns else [],  # First 10 features
            "total_features": len(self.feature_columns) if self.feature_columns else 0
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_to_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_database_connection()


def train_model_pipeline(config_path: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        model_save_path: Optional[str] = None) -> Dict[str, Any]:
    """Complete model training pipeline.
    
    Args:
        config_path: Path to configuration file
        start_date: Start date for training data
        end_date: End date for training data
        model_save_path: Path to save the trained model
        
    Returns:
        Training results
    """
    with TrainingEngine(config_path) as engine:
        # Load training data
        df = engine.load_training_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Prepare features
        features, target = engine.prepare_features(df)
        
        # Train model
        results = engine.train_model(features, target)
        
        # Perform cross-validation
        cv_results = engine.cross_validate(features, target)
        results['cross_validation'] = cv_results
        
        # Save model if path provided
        if model_save_path:
            engine.save_model(model_save_path)
            results['model_saved'] = model_save_path
        
        return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FormFinder2 model")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--model-path", help="Path to save trained model")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None
    
    results = train_model_pipeline(
        config_path=args.config,
        start_date=start_date,
        end_date=end_date,
        model_save_path=args.model_path
    )
    
    print(f"Training completed successfully!")
    print(f"Validation RMSE: {results['validation_metrics']['rmse']:.4f}")
    print(f"Validation R²: {results['validation_metrics']['r2']:.4f}")