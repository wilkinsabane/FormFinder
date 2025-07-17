#!/usr/bin/env python3
"""
Test script for DatabaseDataFetcher to verify database functionality.
"""

import asyncio
import logging
from database_data_fetcher import DatabaseDataFetcher
from formfinder.database import get_db_session, League, Team, Standing, Fixture

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database_contents():
    """Check what's currently in the database."""
    try:
        # Load config if not already loaded
        from formfinder.config import load_config, get_config
        try:
            get_config()
        except RuntimeError:
            load_config("config.yaml")
        
        with get_db_session() as session:
            leagues_count = session.query(League).count()
            teams_count = session.query(Team).count()
            standings_count = session.query(Standing).count()
            fixtures_count = session.query(Fixture).count()
            
            logger.info(f"Database contents:")
            logger.info(f"  - Leagues: {leagues_count}")
            logger.info(f"  - Teams: {teams_count}")
            logger.info(f"  - Standings: {standings_count}")
            logger.info(f"  - Fixtures: {fixtures_count}")
            
            # Show some sample leagues
            sample_leagues = session.query(League).limit(5).all()
            logger.info(f"\nSample leagues:")
            for league in sample_leagues:
                logger.info(f"  - {league.name} ({league.country}) - Season: {league.season}")
            
            return {
                'leagues': leagues_count,
                'teams': teams_count,
                'standings': standings_count,
                'fixtures': fixtures_count
            }
            
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return None

async def test_database_fetcher():
    """Test the DatabaseDataFetcher functionality."""
    logger.info("Testing DatabaseDataFetcher...")
    
    try:
        # Initialize fetcher
        fetcher = DatabaseDataFetcher(use_database=True)
        logger.info("DatabaseDataFetcher initialized successfully")
        
        # Get database summary
        summary = fetcher.get_database_summary()
        logger.info(f"Database summary: {summary}")
        
        # Test fetching standings for a specific league
        test_league_id = 244  # German 3. Liga
        logger.info(f"\nTesting standings fetch for league {test_league_id}...")
        
        standings = await fetcher.fetch_standings_async(test_league_id, f"League-{test_league_id}")
        logger.info(f"Fetched {len(standings)} standings")
        
        if standings:
            # Save to database
            saved_count = fetcher.save_standings_to_database(standings, test_league_id, "2024-2025")
            logger.info(f"Saved {saved_count} standings to database")
        
        # Test fetching historical matches
        logger.info(f"\nTesting historical matches fetch for league {test_league_id}...")
        matches = await fetcher.fetch_historical_matches(test_league_id, f"League-{test_league_id}")
        logger.info(f"Fetched {len(matches)} historical matches")
        
        if matches:
            # Save to database
            saved_count = fetcher.save_matches_to_database(matches, test_league_id, "2024-2025")
            logger.info(f"Saved {saved_count} matches to database")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing DatabaseDataFetcher: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("=== Database Fetcher Test ===")
    
    # Check initial database state
    logger.info("\n1. Checking initial database state...")
    initial_state = check_database_contents()
    
    if initial_state is None:
        logger.error("Failed to check database state")
        return
    
    # Test the fetcher
    logger.info("\n2. Testing DatabaseDataFetcher...")
    success = asyncio.run(test_database_fetcher())
    
    if not success:
        logger.error("DatabaseDataFetcher test failed")
        return
    
    # Check final database state
    logger.info("\n3. Checking final database state...")
    final_state = check_database_contents()
    
    if final_state:
        logger.info("\n=== Test Summary ===")
        logger.info(f"Leagues: {initial_state['leagues']} -> {final_state['leagues']}")
        logger.info(f"Teams: {initial_state['teams']} -> {final_state['teams']}")
        logger.info(f"Standings: {initial_state['standings']} -> {final_state['standings']}")
        logger.info(f"Fixtures: {initial_state['fixtures']} -> {final_state['fixtures']}")
        logger.info("\n✅ Test completed successfully!")
    else:
        logger.error("Failed to check final database state")

if __name__ == "__main__":
    main()