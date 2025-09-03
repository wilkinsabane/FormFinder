#!/usr/bin/env python3
"""
Script to populate the high_form_teams table with wins and total_matches data.

This script processes historical match data to calculate wins and total matches
for high-form teams and populates the database accordingly.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.DataProcessor import DataProcessor
from formfinder.database import init_database, get_db_session
from formfinder.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def populate_high_form_teams_from_historical():
    """Populate high_form_teams table from historical match data."""
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Initialize database
    init_database(config.get_database_url())
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Get historical data directory
    historical_dir = "data/historical"
    if not os.path.exists(historical_dir):
        logger.error(f"Historical data directory {historical_dir} does not exist")
        return
    
    # Find all historical match files
    historical_files = [
        os.path.join(historical_dir, f) 
        for f in os.listdir(historical_dir) 
        if f.startswith('league_') and f.endswith('_historical_matches.csv')
    ]
    
    if not historical_files:
        logger.warning("No historical match files found")
        return
    
    logger.info(f"Found {len(historical_files)} historical match files")
    
    total_teams_saved = 0
    
    for filepath in historical_files:
        filename = os.path.basename(filepath)
        logger.info(f"Processing {filename}")
        
        # Extract league ID from filename
        try:
            league_id = int(filename.split('_')[1])
        except (IndexError, ValueError):
            logger.warning(f"Could not extract league ID from {filename}")
            continue
        
        try:
            # Process the league to get high-form teams
            high_form_df = processor.process_league(filepath=filepath)
            
            if high_form_df.empty:
                logger.info(f"No high-form teams found for league {league_id}")
                continue
            
            # Save to database
            teams_saved = processor.save_high_form_teams_to_database(
                high_form_df, 
                league_id=league_id
            )
            
            total_teams_saved += teams_saved
            logger.info(f"Saved {teams_saved} high-form teams for league {league_id}")
            
        except Exception as e:
            logger.error(f"Error processing league {league_id}: {e}")
            continue
    
    logger.info(f"Completed! Total teams saved to database: {total_teams_saved}")


if __name__ == "__main__":
    logger.info("Starting high-form teams population...")
    populate_high_form_teams_from_historical()
    logger.info("Population complete!")