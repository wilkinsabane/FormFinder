"""Migration to add missing features to pre_computed_features table.

This migration adds all 87 features expected by the trained model to ensure
feature consistency between training and prediction.
"""

import logging
import sys
import os
from sqlalchemy import text

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formfinder.database import get_db_session
from formfinder.config import load_config

logger = logging.getLogger(__name__)

# List of all 87 expected features from training_metadata.json
EXPECTED_FEATURES = [
    "home_attack_strength",
    "home_defense_strength", 
    "away_attack_strength",
    "away_defense_strength",
    "home_team_strength",
    "away_team_strength",
    "home_form_diff",
    "away_form_diff",
    "home_team_form_score",
    "away_team_form_score",
    "home_team_position",
    "away_team_position",
    "home_position_confidence",
    "away_position_confidence",
    "home_advantage",
    "defensive_home_advantage",
    "h2h_total_goals",
    "h2h_competitiveness",
    "home_xg",
    "away_xg",
    "home_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "league_avg_goals",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_avg_goals_scored_home",
    "home_avg_goals_conceded_home",
    "home_form_last_5_games",
    "home_wins_last_5",
    "home_draws_last_5",
    "home_losses_last_5",
    "home_goals_for_last_5",
    "home_goals_against_last_5",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_avg_goals_scored_away",
    "away_avg_goals_conceded_away",
    "away_form_last_5_games",
    "away_wins_last_5",
    "away_draws_last_5",
    "away_losses_last_5",
    "away_goals_for_last_5",
    "away_goals_against_last_5",
    "h2h_total_matches",
    "h2h_avg_goals",
    "h2h_overall_home_goals",
    "h2h_overall_away_goals",
    "h2h_home_advantage",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "markov_home_current_state",
    "markov_home_state_duration",
    "markov_home_state_confidence",
    "markov_home_momentum_score",
    "markov_home_trend_direction",
    "markov_home_state_stability",
    "markov_home_transition_entropy",
    "markov_home_performance_volatility",
    "markov_home_expected_next_state",
    "markov_home_next_state_probability",
    "markov_home_mean_return_time",
    "markov_home_steady_state_probability",
    "markov_home_absorption_probability",
    "markov_away_current_state",
    "markov_away_state_duration",
    "markov_away_state_confidence",
    "markov_away_momentum_score",
    "markov_away_trend_direction",
    "markov_away_state_stability",
    "markov_away_transition_entropy",
    "markov_away_performance_volatility",
    "markov_away_expected_next_state",
    "markov_away_next_state_probability",
    "markov_away_mean_return_time",
    "markov_away_steady_state_probability",
    "markov_away_absorption_probability",
    "markov_momentum_diff",
    "markov_volatility_diff",
    "markov_entropy_diff",
    "markov_match_prediction_confidence",
    "markov_outcome_probabilities",
    "home_team_momentum",
    "away_team_momentum",
    "home_team_sentiment",
    "away_team_sentiment",
    "match_date"
]

# Map feature names to SQL column definitions
FEATURE_COLUMN_DEFINITIONS = {
    # Attack/Defense strength features
    "home_attack_strength": "FLOAT",
    "home_defense_strength": "FLOAT", 
    "away_attack_strength": "FLOAT",
    "away_defense_strength": "FLOAT",
    
    # Team strength and form features
    "home_team_strength": "FLOAT",
    "away_team_strength": "FLOAT",
    "home_form_diff": "FLOAT",
    "away_form_diff": "FLOAT",
    "home_team_form_score": "FLOAT",
    "away_team_form_score": "FLOAT",
    
    # Position features
    "home_team_position": "INTEGER",
    "away_team_position": "INTEGER",
    "home_position_confidence": "FLOAT",
    "away_position_confidence": "FLOAT",
    
    # Home advantage features
    "home_advantage": "FLOAT",
    "defensive_home_advantage": "FLOAT",
    
    # H2H features
    "h2h_total_goals": "FLOAT",
    "h2h_competitiveness": "FLOAT",
    "h2h_total_matches": "INTEGER",
    "h2h_avg_goals": "FLOAT",
    "h2h_home_wins": "INTEGER",
    "h2h_away_wins": "INTEGER",
    
    # xG features
    "home_xg": "FLOAT",
    "away_xg": "FLOAT",
    
    # Goals features
    "home_avg_goals_for": "FLOAT",
    "home_avg_goals_against": "FLOAT",
    "away_avg_goals_for": "FLOAT",
    "away_avg_goals_against": "FLOAT",
    
    # League features
    "league_avg_goals": "FLOAT",
    
    # Markov chain features - home team
    "markov_home_current_state": "VARCHAR(20)",
    "markov_home_state_duration": "INTEGER",
    "markov_home_state_confidence": "FLOAT",
    "markov_home_momentum_score": "FLOAT",
    "markov_home_trend_direction": "FLOAT",
    "markov_home_state_stability": "FLOAT",
    "markov_home_transition_entropy": "FLOAT",
    "markov_home_performance_volatility": "FLOAT",
    "markov_home_expected_next_state": "VARCHAR(20)",
    "markov_home_next_state_probability": "FLOAT",
    "markov_home_mean_return_time": "FLOAT",
    "markov_home_steady_state_probability": "FLOAT",
    "markov_home_absorption_probability": "FLOAT",
    
    # Markov chain features - away team
    "markov_away_current_state": "VARCHAR(20)",
    "markov_away_state_duration": "INTEGER",
    "markov_away_state_confidence": "FLOAT",
    "markov_away_momentum_score": "FLOAT",
    "markov_away_trend_direction": "FLOAT",
    "markov_away_state_stability": "FLOAT",
    "markov_away_transition_entropy": "FLOAT",
    "markov_away_performance_volatility": "FLOAT",
    "markov_away_expected_next_state": "VARCHAR(20)",
    "markov_away_next_state_probability": "FLOAT",
    "markov_away_mean_return_time": "FLOAT",
    "markov_away_steady_state_probability": "FLOAT",
    "markov_away_absorption_probability": "FLOAT",
    
    # Markov comparison features
    "markov_momentum_diff": "FLOAT",
    "markov_volatility_diff": "FLOAT",
    "markov_entropy_diff": "FLOAT",
    "markov_match_prediction_confidence": "FLOAT",
    "markov_outcome_probabilities": "TEXT",
    
    # Momentum and sentiment features
    "home_team_momentum": "FLOAT",
    "away_team_momentum": "FLOAT",
    "home_team_sentiment": "FLOAT",
    "away_team_sentiment": "FLOAT",
    
    # Date feature (already exists as match_date)
    "match_date": "DATETIME"
}


def check_column_exists(session, table_name: str, column_name: str) -> bool:
    """Check if a column exists in the specified table."""
    try:
        result = session.execute(text(f"""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}' 
            AND column_name = '{column_name}'
        """))
        return result.scalar() > 0
    except Exception as e:
        logger.warning(f"Error checking column {column_name}: {e}")
        return False


def add_missing_columns():
    """Add all missing columns to the pre_computed_features table."""
    # Load configuration first
    load_config()
    
    logger.info("Starting migration to add missing features to pre_computed_features table")
    
    with get_db_session() as session:
        table_name = 'pre_computed_features'
        added_columns = []
        skipped_columns = []
        
        for feature_name in EXPECTED_FEATURES:
            if feature_name in FEATURE_COLUMN_DEFINITIONS:
                column_type = FEATURE_COLUMN_DEFINITIONS[feature_name]
                
                # Check if column already exists
                if check_column_exists(session, table_name, feature_name):
                    logger.debug(f"Column {feature_name} already exists, skipping")
                    skipped_columns.append(feature_name)
                    continue
                
                try:
                    # Add the column
                    alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {feature_name} {column_type}"
                    session.execute(text(alter_sql))
                    session.commit()
                    
                    logger.info(f"Added column: {feature_name} ({column_type})")
                    added_columns.append(feature_name)
                    
                except Exception as e:
                    logger.error(f"Failed to add column {feature_name}: {e}")
                    session.rollback()
            else:
                logger.warning(f"No column definition found for feature: {feature_name}")
        
        logger.info(f"Migration completed. Added {len(added_columns)} columns, skipped {len(skipped_columns)} existing columns")
        logger.info(f"Added columns: {added_columns}")
        logger.info(f"Skipped columns: {skipped_columns}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    add_missing_columns()