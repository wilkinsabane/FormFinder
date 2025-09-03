#!/usr/bin/env python3
"""
Feature Precomputation Script

This script populates the pre_computed_features table with computed features
for training the machine learning model.

Usage:
    python scripts/precompute_features.py [--leagues LEAGUES] [--days DAYS]
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session
from formfinder.clients.api_client import SoccerDataAPIClient
from formfinder.feature_precomputer import FeaturePrecomputer
from formfinder.config import DatabaseConfig, load_config
from sqlalchemy import text


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('precompute_features.log')
        ]
    )


def get_recent_fixtures(db_session, leagues: list, days: int = 30):
    """Get recent fixtures for feature computation."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Convert league names to IDs if needed
    league_ids = []
    for league in leagues:
        if isinstance(league, str):
            # Try to find league ID by name
            query = text("SELECT id FROM leagues WHERE name ILIKE :name LIMIT 1")
            result = db_session.execute(query, {'name': f'%{league}%'}).fetchone()
            if result:
                league_ids.append(result[0])
        else:
            league_ids.append(league)
    
    if not league_ids:
        logging.warning("No valid league IDs found")
        return []
    
    # Get fixtures from the specified leagues and date range
    query = text("""
        SELECT id FROM fixtures 
        WHERE league_id = ANY(:league_ids)
        AND fixture_date BETWEEN :start_date AND :end_date
        AND status IN ('FT', 'AET', 'PEN')
        ORDER BY fixture_date DESC
        LIMIT 100
    """)
    
    result = db_session.execute(query, {
        'league_ids': league_ids,
        'start_date': start_date,
        'end_date': end_date
    })
    
    fixture_ids = [row[0] for row in result.fetchall()]
    logging.info(f"Found {len(fixture_ids)} fixtures for computation")
    return fixture_ids


async def main():
    """Main function to run feature precomputation."""
    parser = argparse.ArgumentParser(description='Precompute features for training')
    parser.add_argument('--leagues', nargs='+', default=['Premier League', 'La Liga'],
                       help='Leagues to process (default: Premier League, La Liga)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days back to process (default: 30)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of existing features')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        load_config()
        logger.info("Configuration loaded")
        
        # Use database session context manager
        with get_db_session() as db_session:
            logger.info("Database connection established")
            
            # Initialize API client
            api_client = SoccerDataAPIClient(db_session)
            logger.info("API client initialized")
            
            # Get fixtures to process
            fixture_ids = get_recent_fixtures(db_session, args.leagues, args.days)
            
            if not fixture_ids:
                logger.warning("No fixtures found for processing")
                return
            
            # Initialize feature precomputer
            precomputer = FeaturePrecomputer(db_session, api_client)
            
            # Run feature computation
            logger.info(f"Starting feature computation for {len(fixture_ids)} fixtures")
            stats = await precomputer.compute_all_features(
                fixture_ids, 
                force_refresh=args.force_refresh
            )
            
            # Log final results
            logger.info("Feature computation completed successfully")
            logger.info(f"Statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())