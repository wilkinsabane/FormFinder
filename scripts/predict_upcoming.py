from __future__ import annotations
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import time
from pathlib import Path

# Set up logging
log = logging.getLogger(__name__)

# Add parent directory to path to import formfinder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load configuration first
from formfinder.config import load_config, get_config
load_config()

# Now import modules that depend on configuration
from formfinder.database import get_db_session
from formfinder.logger import log

def safe_save_csv(df, filepath, max_retries=3, retry_delay=1.0):
    """Safely save CSV with retry logic and fallback naming."""
    for attempt in range(max_retries):
        try:
            df.to_csv(filepath, index=False)
            return filepath, True
        except PermissionError as e:
            if attempt < max_retries - 1:
                log.warning(f"Permission denied for {filepath}, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Final attempt with alternative filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
                base_path = Path(filepath)
                fallback_path = base_path.parent / f"{base_path.stem}_fallback_{timestamp}{base_path.suffix}"
                try:
                    df.to_csv(fallback_path, index=False)
                    log.warning(f"Saved to fallback file: {fallback_path} (original file may be open in another application)")
                    return str(fallback_path), False
                except Exception as fallback_e:
                    log.error(f"Failed to save even with fallback filename: {fallback_e}")
                    return None, False
        except Exception as e:
            log.error(f"Unexpected error saving CSV: {e}")
            return None, False
    return None, False

def load_latest_models():
    """Load the most recent trained models."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError("No models directory found")
    
    # Find the latest timestamped models
    model_files = [f for f in os.listdir(models_dir) if f.startswith('metadata_') and f.endswith('.json')]
    if not model_files:
        raise FileNotFoundError("No trained models found. Please run training first.")
    
    # Get the latest model by timestamp
    latest_timestamp = max([f.replace('metadata_', '').replace('.json', '') for f in model_files])
    
    # Load metadata
    metadata_path = os.path.join(models_dir, f"metadata_{latest_timestamp}.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model components
    regressor_path = os.path.join(models_dir, f"goal_regressor_{latest_timestamp}.joblib")
    classifier_path = os.path.join(models_dir, f"over25_classifier_{latest_timestamp}.joblib")
    scaler_path = os.path.join(models_dir, f"feature_scaler_{latest_timestamp}.joblib")
    
    if not all(os.path.exists(p) for p in [regressor_path, classifier_path, scaler_path]):
        raise FileNotFoundError(f"Model files for timestamp {latest_timestamp} are incomplete")
    
    regressor = joblib.load(regressor_path)
    classifier = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    
    metadata['model_type'] = 'timestamped'
    metadata['scaler_needs_fitting'] = False
    
    return regressor, classifier, scaler, metadata

def load_upcoming_fixtures(db_session, leagues):
    """Load upcoming fixtures for prediction."""
    query = text("""
        SELECT 
            f.id as fixture_id,
            f.match_date,
            f.home_team_id,
            f.away_team_id,
            f.league_id,
            NULL as match_api_id,
            t1.name as home_team_name,
            t2.name as away_team_name
        FROM fixtures f
        JOIN teams t1 ON f.home_team_id = t1.id
        JOIN teams t2 ON f.away_team_id = t2.id
        WHERE f.status != 'finished'
        AND f.match_date > NOW()
        AND f.league_id = ANY(:leagues)
        ORDER BY f.match_date ASC
    """)
    
    rows = db_session.execute(query, {"leagues": leagues}).mappings().all()
    return pd.DataFrame(rows)

def load_precomputed_features(df, db_session):
    """Load pre-computed features from database for prediction."""
    fixture_ids = df['fixture_id'].tolist()
    
    if not fixture_ids:
        log.warning("No fixture IDs provided for feature loading")
        return pd.DataFrame()
    
    try:
        # Load all 87 features that the model expects (same as training)
        query = text("""
            SELECT 
                pcf.fixture_id,
                
                -- Core strength features
                pcf.home_attack_strength,
                pcf.home_defense_strength,
                pcf.away_attack_strength,
                pcf.away_defense_strength,
                pcf.home_team_strength,
                pcf.away_team_strength,
                pcf.home_form_diff,
                pcf.away_form_diff,
                pcf.home_team_form_score,
                pcf.away_team_form_score,
                pcf.home_team_position,
                pcf.away_team_position,
                pcf.home_advantage,
                pcf.defensive_home_advantage,
                
                -- H2H features
                pcf.h2h_total_goals,
                pcf.h2h_competitiveness,
                
                -- xG features
                pcf.home_xg,
                pcf.away_xg,
                
                -- Goal averages
                pcf.home_avg_goals_for,
                pcf.home_avg_goals_against,
                pcf.away_avg_goals_for,
                pcf.away_avg_goals_against,
                pcf.league_avg_goals,
                
                -- Home team form features
                pcf.home_avg_goals_scored,
                pcf.home_avg_goals_conceded,
                pcf.home_avg_goals_scored_home,
                pcf.home_avg_goals_conceded_home,
                pcf.home_form_last_5_games,
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
                pcf.away_form_last_5_games,
                pcf.away_wins_last_5,
                pcf.away_draws_last_5,
                pcf.away_losses_last_5,
                pcf.away_goals_for_last_5,
                pcf.away_goals_against_last_5,
                
                -- H2H detailed features
                pcf.h2h_total_matches,
                pcf.h2h_avg_goals,
                pcf.h2h_home_wins,
                pcf.h2h_away_wins,
                
                -- Markov chain features for home team
                pcf.markov_home_current_state,
                pcf.markov_home_state_duration,
                pcf.markov_home_state_confidence,
                pcf.markov_home_momentum_score,
                pcf.markov_home_trend_direction,
                pcf.markov_home_state_stability,
                pcf.markov_home_transition_entropy,
                pcf.markov_home_performance_volatility,
                pcf.markov_home_expected_next_state,
                pcf.markov_home_next_state_probability,
                pcf.markov_home_mean_return_time,
                pcf.markov_home_steady_state_probability,
                pcf.markov_home_absorption_probability,
                
                -- Markov chain features for away team
                pcf.markov_away_current_state,
                pcf.markov_away_state_duration,
                pcf.markov_away_state_confidence,
                pcf.markov_away_momentum_score,
                pcf.markov_away_trend_direction,
                pcf.markov_away_state_stability,
                pcf.markov_away_transition_entropy,
                pcf.markov_away_performance_volatility,
                pcf.markov_away_expected_next_state,
                pcf.markov_away_next_state_probability,
                pcf.markov_away_mean_return_time,
                pcf.markov_away_steady_state_probability,
                pcf.markov_away_absorption_probability,
                
                -- Markov differential features
                pcf.markov_momentum_diff,
                pcf.markov_volatility_diff,
                pcf.markov_entropy_diff,
                pcf.markov_match_prediction_confidence,
                pcf.markov_outcome_probabilities,
                
                -- Team momentum and sentiment
                pcf.home_team_momentum,
                pcf.away_team_momentum,
                pcf.home_team_sentiment,
                pcf.away_team_sentiment,
                
                -- Match date for temporal features
                pcf.match_date
                
            FROM pre_computed_features pcf
            WHERE pcf.fixture_id = ANY(:fixture_ids)
        """)
        
        result = db_session.execute(query, {"fixture_ids": fixture_ids})
        features_data = result.mappings().all()
        
        if not features_data:
            log.warning(f"No pre-computed features found for fixtures: {fixture_ids}")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_data)
        log.info(f"Loaded pre-computed features for {len(features_df)} fixtures")
        
        # Fill missing values with appropriate defaults
        for col in features_df.columns:
            if col != 'fixture_id' and features_df[col].dtype in ['float64', 'int64']:
                features_df[col] = features_df[col].fillna(0.0)
        
        return features_df
        
    except Exception as e:
        log.error(f"Error loading pre-computed features: {e}")
        return pd.DataFrame()



def save_predictions(db_session, predictions_df):
    """Save predictions to the database and CSV file."""
    try:
        # Save to database
        for _, row in predictions_df.iterrows():
            try:
                # Calculate confidence score based on how far the probability is from 0.5 (neutral)
                over_2_5_prob = float(row['over_2_5_probability'])
                confidence_score = max(over_2_5_prob, 1 - over_2_5_prob)  # Distance from 0.5, scaled to [0.5, 1.0]
                
                # First, try to update an existing prediction for this fixture
                update_result = db_session.execute(text("""
                    UPDATE predictions
                    SET
                        predicted_total_goals = :predicted_total_goals,
                        over_2_5_probability = :over_2_5_probability,
                        confidence_score = :confidence_score,
                        updated_at = NOW()
                    WHERE fixture_id = :fixture_id
                """), {
                    'fixture_id': int(row['fixture_id']),
                    'predicted_total_goals': float(row['predicted_total_goals']),
                    'over_2_5_probability': over_2_5_prob,
                    'confidence_score': confidence_score
                })

                if update_result.rowcount == 0:
                    # If no row was updated, insert a new prediction
                    db_session.execute(text("""
                        INSERT INTO predictions (
                            fixture_id, 
                            predicted_total_goals, 
                            over_2_5_probability,
                            confidence_score,
                            created_at,
                            updated_at
                        ) VALUES (
                            :fixture_id, 
                            :predicted_total_goals, 
                            :over_2_5_probability,
                            :confidence_score,
                            NOW(),
                            NOW()
                        )
                    """), {
                        'fixture_id': int(row['fixture_id']),
                        'predicted_total_goals': float(row['predicted_total_goals']),
                        'over_2_5_probability': over_2_5_prob,
                        'confidence_score': confidence_score
                    })
            except Exception as e:
                log.warning(f"Failed to save prediction for fixture {row['fixture_id']}: {e}")
                db_session.rollback()
                continue
        
        db_session.commit()
        log.info(f"Saved {len(predictions_df)} predictions to database")
        
        # Create predictions directory if it doesn't exist
        predictions_dir = 'predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save to CSV file with enhanced information
        try:
            # Fetch fixture details for CSV using individual queries to avoid PostgreSQL function issues
            fixture_details = []
            for fixture_id in predictions_df['fixture_id']:
                try:
                    detail = db_session.execute(text("""
                        SELECT 
                            f.id,
                            ht.name as home_team,
                            at.name as away_team,
                            f.match_date,
                            f.league_id,
                            l.name as league_name
                        FROM fixtures f
                        LEFT JOIN teams ht ON f.home_team_id = ht.id
                        LEFT JOIN teams at ON f.away_team_id = at.id
                        LEFT JOIN leagues l ON f.league_id = l.id
                        WHERE f.id = :fixture_id
                    """), {'fixture_id': int(fixture_id)}).fetchone()
                    
                    if detail:
                        fixture_details.append(detail)
                except Exception as e:
                    log.warning(f"Failed to fetch details for fixture {fixture_id}: {e}")
                    continue
            
            # Create a mapping of fixture details
            fixture_map = {}
            for detail in fixture_details:
                fixture_map[detail[0]] = {
                    'home_team': detail[1],
                    'away_team': detail[2],
                    'match_date': detail[3].strftime('%Y-%m-%d %H:%M:%S') if detail[3] else 'Unknown',
                    'league_id': detail[4],
                    'league_name': detail[5] or f'League {detail[4]}'
                }
            
            # Add enhanced information to predictions
            csv_df = predictions_df.copy()
            csv_df['prediction_timestamp'] = datetime.now().isoformat()
            csv_df['under_2_5_probability'] = 1 - csv_df['over_2_5_probability']
            
            # Add fixture details
            csv_df['home_team'] = csv_df['fixture_id'].map(lambda x: fixture_map.get(x, {}).get('home_team', 'Unknown'))
            csv_df['away_team'] = csv_df['fixture_id'].map(lambda x: fixture_map.get(x, {}).get('away_team', 'Unknown'))
            csv_df['match_date'] = csv_df['fixture_id'].map(lambda x: fixture_map.get(x, {}).get('match_date', 'Unknown'))
            csv_df['league_name'] = csv_df['fixture_id'].map(lambda x: fixture_map.get(x, {}).get('league_name', 'Unknown'))
            csv_df['league_id'] = csv_df['fixture_id'].map(lambda x: fixture_map.get(x, {}).get('league_id', 'Unknown'))
            
            # Add match description for easy identification
            csv_df['match_description'] = csv_df['home_team'] + ' vs ' + csv_df['away_team']
            
            # Round probabilities and goals for better readability
            csv_df['predicted_total_goals'] = csv_df['predicted_total_goals'].round(2)
            csv_df['over_2_5_probability'] = csv_df['over_2_5_probability'].round(3)
            csv_df['under_2_5_probability'] = csv_df['under_2_5_probability'].round(3)
            
            # Reorder columns for better readability
            column_order = [
                'fixture_id', 'match_description', 'home_team', 'away_team', 'match_date', 
                'league_name', 'league_id', 'predicted_total_goals', 'over_2_5_probability', 
                'under_2_5_probability', 'prediction_timestamp'
            ]
            csv_df = csv_df[column_order]
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = os.path.join(predictions_dir, f'predictions_{timestamp}.csv')
            latest_filename = os.path.join(predictions_dir, 'latest_predictions.csv')
            root_latest_filename = 'latest_predictions.csv'  # Root directory file
            
            # Save to timestamped CSV
            csv_df.to_csv(csv_filename, index=False)
            log.info(f"Saved {len(predictions_df)} predictions to CSV file: {csv_filename}")
            
            # Also save to a latest predictions file in predictions directory
            saved_path, success = safe_save_csv(csv_df, latest_filename)
            if success:
                log.info(f"Updated {latest_filename} with {len(predictions_df)} predictions")
            elif saved_path:
                log.info(f"Saved predictions to fallback file: {saved_path}")
            else:
                log.error(f"Failed to save latest predictions file in predictions directory")
            
            # Also update the root directory latest_predictions.csv file
            root_saved_path, root_success = safe_save_csv(csv_df, root_latest_filename)
            if root_success:
                log.info(f"Updated {root_latest_filename} with {len(predictions_df)} predictions")
            elif root_saved_path:
                log.info(f"Saved predictions to root fallback file: {root_saved_path}")
            else:
                log.error(f"Failed to save latest predictions file in root directory")
            
        except Exception as e:
            log.warning(f"Failed to save enhanced CSV: {e}")
            # Fallback to basic CSV if enhanced version fails
            try:
                basic_csv_df = predictions_df.copy()
                basic_csv_df['prediction_timestamp'] = datetime.now().isoformat()
                basic_csv_df['under_2_5_probability'] = 1 - basic_csv_df['over_2_5_probability']
                
                # Round values for better readability
                basic_csv_df['predicted_total_goals'] = basic_csv_df['predicted_total_goals'].round(2)
                basic_csv_df['over_2_5_probability'] = basic_csv_df['over_2_5_probability'].round(3)
                basic_csv_df['under_2_5_probability'] = basic_csv_df['under_2_5_probability'].round(3)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                basic_csv_filename = os.path.join(predictions_dir, f'predictions_basic_{timestamp}.csv')
                latest_basic_filename = os.path.join(predictions_dir, 'latest_predictions.csv')
                
                basic_csv_df.to_csv(basic_csv_filename, index=False)
                
                # Try to save to latest file with retry logic
                saved_path, success = safe_save_csv(basic_csv_df, latest_basic_filename)
                if success:
                    log.info(f"Saved basic CSV as fallback: {basic_csv_filename} and updated {latest_basic_filename}")
                elif saved_path:
                    log.info(f"Saved basic CSV as fallback: {basic_csv_filename} and to fallback file: {saved_path}")
                else:
                    log.info(f"Saved basic CSV as fallback: {basic_csv_filename} (could not update latest file)")
            except Exception as fallback_e:
                log.error(f"Failed to save even basic CSV: {fallback_e}")
            
    except Exception as e:
        log.error(f"Error saving predictions: {e}")
        db_session.rollback()
        raise

def main():
    """Main prediction pipeline."""
    load_config()
    config = get_config()
    
    # Performance tracking
    start_time = time.time()
    summary_data = {
        'total_fixtures': 0,
        'predictions_generated': 0,
        'features_loaded': 0,
        'feature_loading_failures': 0,
        'high_probability_matches': 0,
        'processing_time': 0.0
    }
    
    # Use proper database session management
    with get_db_session() as db_session:
        try:
            # Load latest models
            log.info("Loading latest models...")
            regressor, classifier, scaler, metadata = load_latest_models()
            feature_columns = metadata['feature_columns']
            scaler_needs_fitting = metadata.get('scaler_needs_fitting', False)
            
            # Log model info
            model_type = metadata.get('model_type', 'unknown')
            if 'training_samples' in metadata:
                log.info(f"Loaded {model_type} models trained on {metadata['training_samples']} samples")
            else:
                log.info(f"Loaded {model_type} models (version: {metadata.get('version', 'unknown')})")
            
            log.info(f"Feature columns available: {len(feature_columns)}")
            
            # Load leagues from free_leagues.txt
            with open('free_leagues.txt', 'r') as f:
                leagues = [int(line.strip()) for line in f if line.strip().isdigit()]
            
            log.info(f"Predicting for leagues: {leagues}")
            
            # Load upcoming fixtures
            log.info("Loading upcoming fixtures...")
            df = load_upcoming_fixtures(db_session, leagues)
            summary_data['total_fixtures'] = len(df)
            log.info(f"Found {len(df)} upcoming fixtures")
            
            if len(df) == 0:
                log.info("No upcoming fixtures found")
                return
            
            # Load precomputed features from database
            log.info("Loading precomputed features...")
            features_df = load_precomputed_features(df, db_session)
            
            if len(features_df) == 0:
                log.error("No precomputed features found for upcoming fixtures")
                log.error("Please run the feature computation scripts first")
                summary_data['feature_loading_failures'] = len(df)
                return
            
            summary_data['features_loaded'] = len(features_df)
            log.info(f"Loaded precomputed features for {len(features_df)} fixtures")
            
            # Use the exact features that the model expects from metadata
            expected_features = feature_columns
            
            # Create feature matrix with expected features, filling missing ones with defaults
            feature_matrix = []
            for _, row in features_df.iterrows():
                feature_row = []
                for feature in expected_features:
                    if feature in features_df.columns:
                        value = row[feature]
                        # Handle NaN values
                        if pd.isna(value):
                            feature_row.append(0.0)  # Default value for missing data
                        else:
                            # Handle different data types properly
                            if pd.api.types.is_datetime64_any_dtype(type(value)) or hasattr(value, 'timestamp'):
                                # Convert Timestamp to Unix timestamp (seconds since epoch)
                                try:
                                    feature_row.append(float(value.timestamp()))
                                except (AttributeError, TypeError):
                                    # Fallback for other datetime-like objects
                                    feature_row.append(0.0)
                            else:
                                try:
                                    feature_row.append(float(value))
                                except (TypeError, ValueError):
                                    # Handle any other non-numeric types
                                    feature_row.append(0.0)
                    else:
                        # Feature not available in current schema, use default
                        feature_row.append(0.0)
                feature_matrix.append(feature_row)
            
            # Create DataFrame with proper column names to avoid StandardScaler warning
            X_df = pd.DataFrame(feature_matrix, columns=expected_features)
            log.info(f"Prepared feature matrix with shape {X_df.shape} for {len(expected_features)} expected features")
            
            # Fit scaler if needed
            if scaler_needs_fitting:
                log.info("Fitting new scaler to current feature set")
                scaler.fit(X_df)
                scaler_needs_fitting = False
            
            X_scaled = scaler.transform(X_df)
            
            # Generate predictions
            log.info("Generating predictions...")
            predicted_total_goals = regressor.predict(X_scaled)
            over_2_5_probability = classifier.predict_proba(X_scaled)[:, 1]
            
            # Combine with fixture data
            predictions_df = pd.DataFrame({
                'fixture_id': features_df['fixture_id'],
                'predicted_total_goals': predicted_total_goals,
                'over_2_5_probability': over_2_5_probability
            })
            
            summary_data['predictions_generated'] = len(predictions_df)
            
            # Save predictions
            save_predictions(db_session, predictions_df)
            
            # Calculate summary statistics
            high_over = predictions_df[predictions_df['over_2_5_probability'] > 0.7]
            summary_data['high_probability_matches'] = len(high_over)
            summary_data['processing_time'] = time.time() - start_time
            
            # Log comprehensive summary
            log.info("=== PREDICTION SUMMARY ===")
            log.info(f"Total fixtures processed: {summary_data['total_fixtures']}")
            log.info(f"Features loaded: {summary_data['features_loaded']}")
            log.info(f"Feature loading failures: {summary_data['feature_loading_failures']}")
            log.info(f"Predictions generated: {summary_data['predictions_generated']}")
            log.info(f"High probability matches (>70%): {summary_data['high_probability_matches']}")
            log.info(f"Total processing time: {summary_data['processing_time']:.2f} seconds")
            
            # Save summary to JSON file
            try:
                with open('prediction_summary.json', 'w') as f:
                    json.dump({
                        **summary_data,
                        'timestamp': datetime.now().isoformat(),
                        'leagues_processed': leagues
                    }, f, indent=2)
                log.info("Summary saved to prediction_summary.json")
            except Exception as e:
                log.warning(f"Failed to save summary: {e}")
        
        except Exception as e:
            log.error(f"Error in prediction pipeline: {e}")
            raise
        finally:
            log.info("Prediction pipeline completed")

if __name__ == "__main__":
    main()