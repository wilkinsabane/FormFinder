#!/usr/bin/env python3
"""
Insert Sample Features Script

This script inserts sample feature data into the pre_computed_features table
for testing the training pipeline.
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_sample_features(num_samples=100):
    """Generate sample feature data."""
    features = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_samples):
        # Generate random but realistic feature values
        feature_data = {
            'fixture_id': 1000 + i,
            'home_team_id': random.randint(1, 20),
            'away_team_id': random.randint(1, 20),
            'match_date': base_date + timedelta(days=random.randint(0, 365)),
            'league_id': random.randint(1, 5),
            'home_avg_goals_scored': round(random.uniform(0.5, 3.0), 2),
            'home_avg_goals_conceded': round(random.uniform(0.5, 3.0), 2),
            'home_avg_goals_scored_home': round(random.uniform(0.5, 3.5), 2),
            'home_avg_goals_conceded_home': round(random.uniform(0.3, 2.5), 2),
            'home_avg_goals_scored_away': round(random.uniform(0.3, 2.5), 2),
            'home_avg_goals_conceded_away': round(random.uniform(0.5, 3.5), 2),
            'away_avg_goals_scored': round(random.uniform(0.5, 3.0), 2),
            'away_avg_goals_conceded': round(random.uniform(0.5, 3.0), 2),
            'away_avg_goals_scored_home': round(random.uniform(0.5, 3.5), 2),
            'away_avg_goals_conceded_home': round(random.uniform(0.3, 2.5), 2),
            'away_avg_goals_scored_away': round(random.uniform(0.3, 2.5), 2),
            'away_avg_goals_conceded_away': round(random.uniform(0.5, 3.5), 2),
            'home_form_points': random.randint(0, 15),
            'away_form_points': random.randint(0, 15),
            'home_win_rate': round(random.uniform(0.0, 1.0), 3),
            'away_win_rate': round(random.uniform(0.0, 1.0), 3),
            'home_draw_rate': round(random.uniform(0.0, 0.5), 3),
            'away_draw_rate': round(random.uniform(0.0, 0.5), 3),
            'home_loss_rate': round(random.uniform(0.0, 1.0), 3),
            'away_loss_rate': round(random.uniform(0.0, 1.0), 3),
            'head_to_head_home_wins': random.randint(0, 10),
            'head_to_head_away_wins': random.randint(0, 10),
            'head_to_head_draws': random.randint(0, 5),
            'home_team_strength': round(random.uniform(0.3, 2.0), 3),
            'away_team_strength': round(random.uniform(0.3, 2.0), 3),
            'home_attack_strength': round(random.uniform(0.3, 2.5), 3),
            'home_defense_strength': round(random.uniform(0.3, 2.5), 3),
            'away_attack_strength': round(random.uniform(0.3, 2.5), 3),
            'away_defense_strength': round(random.uniform(0.3, 2.5), 3),
            'home_recent_form': round(random.uniform(0.0, 3.0), 2),
            'away_recent_form': round(random.uniform(0.0, 3.0), 2),
            'home_home_form': round(random.uniform(0.0, 3.0), 2),
            'away_away_form': round(random.uniform(0.0, 3.0), 2),
            'home_motivation': round(random.uniform(0.5, 1.5), 3),
            'away_motivation': round(random.uniform(0.5, 1.5), 3),
            'home_fatigue_factor': round(random.uniform(0.8, 1.2), 3),
            'away_fatigue_factor': round(random.uniform(0.8, 1.2), 3),
            'weather_impact': round(random.uniform(0.9, 1.1), 3),
            'referee_bias': round(random.uniform(0.95, 1.05), 3),
            'crowd_impact': round(random.uniform(0.95, 1.15), 3),
            'injury_impact_home': round(random.uniform(0.8, 1.0), 3),
            'injury_impact_away': round(random.uniform(0.8, 1.0), 3),
            'suspension_impact_home': round(random.uniform(0.9, 1.0), 3),
            'suspension_impact_away': round(random.uniform(0.9, 1.0), 3),
            'market_confidence_home': round(random.uniform(0.3, 3.0), 3),
            'market_confidence_away': round(random.uniform(0.3, 3.0), 3),
            'computed_at': datetime.now()
        }
        
        # Ensure away team is different from home team
        while feature_data['away_team_id'] == feature_data['home_team_id']:
            feature_data['away_team_id'] = random.randint(1, 20)
        
        features.append(feature_data)
    
    return features


def insert_features(db_session, features):
    """Insert features into the database."""
    logger = logging.getLogger(__name__)
    
    # Build the insert query
    columns = list(features[0].keys())
    placeholders = ', '.join([f':{col}' for col in columns])
    column_names = ', '.join(columns)
    
    query = text(f"""
        INSERT INTO pre_computed_features ({column_names})
        VALUES ({placeholders})
    """)
    
    try:
        # Insert all features
        for feature in features:
            db_session.execute(query, feature)
        
        db_session.commit()
        logger.info(f"Successfully inserted {len(features)} feature records")
        
    except Exception as e:
        db_session.rollback()
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
        with get_db_session() as db_session:
            logger.info("Database connection established")
            insert_features(db_session, features)
        
        logger.info("Sample features inserted successfully")
        
    except Exception as e:
        logger.error(f"Failed to insert sample features: {e}")
        raise


if __name__ == '__main__':
    main()