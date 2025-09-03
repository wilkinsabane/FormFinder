#!/usr/bin/env python3
"""
Direct Insert Sample Features Script

This script directly inserts sample feature data into the pre_computed_features table
using psycopg2 to avoid SQLAlchemy table creation issues.
"""

import sys
import logging
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
import random
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from formfinder.config import load_config, get_config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_db_connection():
    """Get direct database connection."""
    config = get_config()
    db_config = config.database
    
    if db_config.type == 'postgresql':
        pg_config = db_config.postgresql
        return psycopg2.connect(
            host=pg_config.host,
            port=pg_config.port,
            database=pg_config.database,
            user=pg_config.username,
            password=pg_config.password
        )
    else:
        raise ValueError("Only PostgreSQL is supported for direct connection")


def generate_sample_features(num_samples=100):
    """Generate sample feature data."""
    features = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_samples):
        # Generate random but realistic feature values
        feature_data = (
            1000 + i,  # fixture_id
            random.randint(1, 20),  # home_team_id
            random.randint(1, 20),  # away_team_id
            base_date + timedelta(days=random.randint(0, 365)),  # match_date
            random.randint(1, 5),  # league_id
            round(random.uniform(0.5, 3.0), 2),  # home_avg_goals_scored
            round(random.uniform(0.5, 3.0), 2),  # home_avg_goals_conceded
            round(random.uniform(0.5, 3.5), 2),  # home_avg_goals_scored_home
            round(random.uniform(0.3, 2.5), 2),  # home_avg_goals_conceded_home
            round(random.uniform(0.3, 2.5), 2),  # home_avg_goals_scored_away
            round(random.uniform(0.5, 3.5), 2),  # home_avg_goals_conceded_away
            round(random.uniform(0.5, 3.0), 2),  # away_avg_goals_scored
            round(random.uniform(0.5, 3.0), 2),  # away_avg_goals_conceded
            round(random.uniform(0.5, 3.5), 2),  # away_avg_goals_scored_home
            round(random.uniform(0.3, 2.5), 2),  # away_avg_goals_conceded_home
            round(random.uniform(0.3, 2.5), 2),  # away_avg_goals_scored_away
            round(random.uniform(0.5, 3.5), 2),  # away_avg_goals_conceded_away
            random.randint(0, 15),  # home_form_points
            random.randint(0, 15),  # away_form_points
            round(random.uniform(0.0, 1.0), 3),  # home_win_rate
            round(random.uniform(0.0, 1.0), 3),  # away_win_rate
            round(random.uniform(0.0, 0.5), 3),  # home_draw_rate
            round(random.uniform(0.0, 0.5), 3),  # away_draw_rate
            round(random.uniform(0.0, 1.0), 3),  # home_loss_rate
            round(random.uniform(0.0, 1.0), 3),  # away_loss_rate
            random.randint(0, 10),  # head_to_head_home_wins
            random.randint(0, 10),  # head_to_head_away_wins
            random.randint(0, 5),  # head_to_head_draws
            round(random.uniform(0.3, 2.0), 3),  # home_team_strength
            round(random.uniform(0.3, 2.0), 3),  # away_team_strength
            round(random.uniform(0.3, 2.5), 3),  # home_attack_strength
            round(random.uniform(0.3, 2.5), 3),  # home_defense_strength
            round(random.uniform(0.3, 2.5), 3),  # away_attack_strength
            round(random.uniform(0.3, 2.5), 3),  # away_defense_strength
            round(random.uniform(0.0, 3.0), 2),  # home_recent_form
            round(random.uniform(0.0, 3.0), 2),  # away_recent_form
            round(random.uniform(0.0, 3.0), 2),  # home_home_form
            round(random.uniform(0.0, 3.0), 2),  # away_away_form
            round(random.uniform(0.5, 1.5), 3),  # home_motivation
            round(random.uniform(0.5, 1.5), 3),  # away_motivation
            round(random.uniform(0.8, 1.2), 3),  # home_fatigue_factor
            round(random.uniform(0.8, 1.2), 3),  # away_fatigue_factor
            round(random.uniform(0.9, 1.1), 3),  # weather_impact
            round(random.uniform(0.95, 1.05), 3),  # referee_bias
            round(random.uniform(0.95, 1.15), 3),  # crowd_impact
            round(random.uniform(0.8, 1.0), 3),  # injury_impact_home
            round(random.uniform(0.8, 1.0), 3),  # injury_impact_away
            round(random.uniform(0.9, 1.0), 3),  # suspension_impact_home
            round(random.uniform(0.9, 1.0), 3),  # suspension_impact_away
            round(random.uniform(0.3, 3.0), 3),  # market_confidence_home
            round(random.uniform(0.3, 3.0), 3),  # market_confidence_away
            datetime.now()  # computed_at
        )
        
        # Ensure away team is different from home team
        while feature_data[2] == feature_data[1]:  # away_team_id == home_team_id
            feature_data = list(feature_data)
            feature_data[2] = random.randint(1, 20)
            feature_data = tuple(feature_data)
        
        features.append(feature_data)
    
    return features


def insert_features(conn, features):
    """Insert features into the database."""
    logger = logging.getLogger(__name__)
    
    # Build the insert query (49 columns, excluding id which is auto-generated)
    insert_query = """
        INSERT INTO pre_computed_features (
            fixture_id, home_team_id, away_team_id, match_date, league_id,
            home_avg_goals_scored, home_avg_goals_conceded, home_avg_goals_scored_home,
            home_avg_goals_conceded_home, home_avg_goals_scored_away, home_avg_goals_conceded_away,
            away_avg_goals_scored, away_avg_goals_conceded, away_avg_goals_scored_home,
            away_avg_goals_conceded_home, away_avg_goals_scored_away, away_avg_goals_conceded_away,
            home_form_points, away_form_points, home_win_rate, away_win_rate,
            home_draw_rate, away_draw_rate, home_loss_rate, away_loss_rate,
            head_to_head_home_wins, head_to_head_away_wins, head_to_head_draws,
            home_team_strength, away_team_strength, home_attack_strength, home_defense_strength,
            away_attack_strength, away_defense_strength, home_recent_form, away_recent_form,
            home_home_form, away_away_form, home_motivation, away_motivation,
            home_fatigue_factor, away_fatigue_factor, weather_impact, referee_bias,
            crowd_impact, injury_impact_home, injury_impact_away, suspension_impact_home,
            suspension_impact_away, market_confidence_home, market_confidence_away, computed_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    
    try:
        with conn.cursor() as cursor:
            # Insert all features
            cursor.executemany(insert_query, features)
            conn.commit()
            logger.info(f"Successfully inserted {len(features)} feature records")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to insert features: {e}")
        raise


def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        load_config()
        logger.info("Configuration loaded")
        
        # Generate sample features
        logger.info("Generating sample features...")
        features = generate_sample_features(100)
        
        # Insert into database
        with get_db_connection() as conn:
            logger.info("Database connection established")
            insert_features(conn, features)
        
        logger.info("Sample features inserted successfully")
        
    except Exception as e:
        logger.error(f"Failed to insert sample features: {e}")
        raise


if __name__ == '__main__':
    main()