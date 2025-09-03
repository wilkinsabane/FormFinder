#!/usr/bin/env python3
"""
Upcoming Feature Precomputation Script

This script populates the pre_computed_features table with computed features
for all upcoming fixtures to enable predictions.

Usage:
    python scripts/precompute_upcoming_features.py [--leagues LEAGUES] [--days DAYS]
"""


import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session

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
            logging.FileHandler('precompute_upcoming_features.log')
        ]
    )


def get_upcoming_fixtures(db_session, leagues: list = None, days: int = 30):
    """Get upcoming fixtures for feature computation."""
    start_date = datetime.now()
    end_date = start_date + timedelta(days=days)
    
    # If no leagues specified, get all leagues from free_leagues.txt
    if not leagues:
        try:
            with open('free_leagues.txt', 'r') as f:
                league_ids = [int(line.strip()) for line in f if line.strip()]
        except FileNotFoundError:
            logging.warning("free_leagues.txt not found, using all leagues")
            league_ids = None
    else:
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
    
    # Build query based on whether league filtering is needed
    if league_ids:
        query = text("""
            SELECT f.id 
            FROM fixtures f
            LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.league_id = ANY(:league_ids)
            AND f.match_date >= :start_date
            AND f.match_date <= :end_date
            AND f.status != 'FINISHED'
            AND pcf.fixture_id IS NULL
            ORDER BY f.match_date
        """)
        
        result = db_session.execute(query, {
            'league_ids': league_ids,
            'start_date': start_date,
            'end_date': end_date
        })
    else:
        query = text("""
            SELECT f.id 
            FROM fixtures f
            LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.match_date >= :start_date
            AND f.match_date <= :end_date
            AND f.status != 'FINISHED'
            AND pcf.fixture_id IS NULL
            ORDER BY f.match_date
        """)
        
        result = db_session.execute(query, {
            'start_date': start_date,
            'end_date': end_date
        })
    
    fixture_ids = [row[0] for row in result.fetchall()]
    logging.info(f"Found {len(fixture_ids)} upcoming fixtures for computation")
    return fixture_ids


def main():
    """Main function to run upcoming feature precomputation."""
    parser = argparse.ArgumentParser(description='Precompute features for upcoming fixtures')
    parser.add_argument('--leagues', nargs='+', default=None,
                       help='Leagues to process (default: use free_leagues.txt)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days ahead to process (default: 30)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of existing features')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for processing (default: 50)')
    
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
            
            # Get upcoming fixtures to process
            fixture_ids = get_upcoming_fixtures(db_session, args.leagues, args.days)
            
            if not fixture_ids:
                logger.warning("No upcoming fixtures found for processing")
                return
            
            # Initialize feature precomputer
            precomputer = FeaturePrecomputer(db_session)
            
            # Process fixtures in batches to avoid overwhelming the system
            batch_size = args.batch_size
            total_processed = 0
            total_successful = 0
            total_failed = 0
            
            logger.info(f"Starting feature computation for {len(fixture_ids)} upcoming fixtures")
            logger.info(f"Processing in batches of {batch_size}")
            
            for i in range(0, len(fixture_ids), batch_size):
                batch = fixture_ids[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(fixture_ids) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} fixtures)")
                
                try:
                    # Run feature computation for this batch
                    stats = precomputer.compute_all_features(
                        batch, 
                        force_refresh=args.force_refresh
                    )
                    
                    batch_successful = stats.get('successful_computations', 0)
                    batch_failed = stats.get('failed_computations', 0)
                    
                    total_processed += len(batch)
                    total_successful += batch_successful
                    total_failed += batch_failed
                    
                    logger.info(f"Batch {batch_num} completed: {batch_successful} successful, {batch_failed} failed")
                    
                    # Small delay between batches to avoid overwhelming the API
                    if i + batch_size < len(fixture_ids):
                        import time
                        time.sleep(2)
                        
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    total_failed += len(batch)
                    continue
            
            # Log final results
            logger.info("=" * 50)
            logger.info("Feature computation completed")
            logger.info(f"Total fixtures processed: {total_processed}")
            logger.info(f"Successful computations: {total_successful}")
            logger.info(f"Failed computations: {total_failed}")
            logger.info(f"Success rate: {(total_successful/total_processed)*100:.1f}%" if total_processed > 0 else "N/A")
            logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        raise


if __name__ == '__main__':
    main()