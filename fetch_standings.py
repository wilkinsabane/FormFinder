#!/usr/bin/env python3
"""
Standalone standings fetcher script.

Fetches league standings for specified leagues and seasons, saving them to the database.
Useful for filling gaps in standings data or updating specific league standings.

Usage:
    python fetch_standings.py --league-id 311 --season 2024
    python fetch_standings.py --all-leagues --season 2024
    python fetch_standings.py --league-id 39,40,78 --season 2023
"""

import argparse
import asyncio
import sys
from typing import List, Optional

from formfinder.config import load_config
from formfinder.database import get_db_session
from database_data_fetcher import DatabaseDataFetcher
from formfinder.logger import get_logger


def parse_league_ids(league_ids_str: str) -> List[int]:
    """Parse comma-separated league IDs into a list of integers."""
    try:
        return [int(lid.strip()) for lid in league_ids_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid league ID format: {e}")


def get_all_configured_leagues() -> List[int]:
    """Get all league IDs from the configuration."""
    config = load_config()
    from formfinder.config import FormFinderConfig
    ff_config = FormFinderConfig(config)
    return ff_config.get_league_ids()


async def fetch_standings_for_leagues(league_ids: List[int], season: str, force: bool = False) -> None:
    """Fetch standings for specified leagues and season."""
    logger = get_logger(__name__)
    
    # Initialize database fetcher
    fetcher = DatabaseDataFetcher("config.yaml", use_database=True)
    
    logger.info(f"Starting standings fetch for {len(league_ids)} leagues, season {season}")
    
    success_count = 0
    error_count = 0
    
    for league_id in league_ids:
        try:
            logger.info(f"Fetching standings for league {league_id}, season {season}")
            
            # Check if standings already exist for this league/season
            if not force:
                from formfinder.database import get_db_session
                from sqlalchemy import text
                with get_db_session() as session:
                    existing_query = text("""
                        SELECT COUNT(*) FROM standings 
                        WHERE league_id = :league_id AND season = :season
                    """)
                    existing_count = session.execute(
                        existing_query, 
                        {'league_id': league_id, 'season': season}
                    ).scalar()
                    
                    if existing_count > 0:
                        logger.info(f"Standings already exist for league {league_id}, season {season}. Use --force to overwrite.")
                        continue
            
            # Get league name for the async call
            from formfinder.database import get_db_session
            from sqlalchemy import text
            with get_db_session() as session:
                league_query = text("SELECT name FROM leagues WHERE id = :league_id")
                league_result = session.execute(league_query, {'league_id': league_id}).fetchone()
                league_name = league_result[0] if league_result else f"League {league_id}"
            
            # Fetch standings from API
            standings = await fetcher.fetch_standings_async(league_id, league_name)
            
            if standings:
                # Save to database
                await fetcher.fetch_standings_to_db(standings, league_id, season)
                logger.info(f"Successfully fetched and saved {len(standings)} standings for league {league_id}")
                success_count += 1
            else:
                logger.warning(f"No standings data returned for league {league_id}, season {season}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"Error fetching standings for league {league_id}: {e}")
            error_count += 1
    
    logger.info(f"Standings fetch completed. Success: {success_count}, Errors: {error_count}")


def main():
    """Main entry point for the standings fetcher script."""
    parser = argparse.ArgumentParser(
        description="Fetch league standings for specified leagues and seasons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --league-id 311 --season 2024
  %(prog)s --all-leagues --season 2024
  %(prog)s --league-id 39,40,78 --season 2023
  %(prog)s --league-id 311 --season 2024 --force
        """
    )
    
    # League selection arguments (mutually exclusive)
    league_group = parser.add_mutually_exclusive_group(required=True)
    league_group.add_argument(
        '--league-id', 
        type=str,
        help='Comma-separated list of league IDs to fetch standings for (e.g., "311,39,40")'
    )
    league_group.add_argument(
        '--all-leagues',
        action='store_true',
        help='Fetch standings for all configured leagues'
    )
    
    # Season and options
    parser.add_argument(
        '--season',
        type=str,
        required=True,
        help='Season to fetch standings for (e.g., "2024", "2023")'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force fetch even if standings already exist for the league/season'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine league IDs to fetch
    if args.all_leagues:
        try:
            league_ids = get_all_configured_leagues()
            print(f"Found {len(league_ids)} configured leagues")
        except Exception as e:
            print(f"Error loading configured leagues: {e}")
            sys.exit(1)
    else:
        try:
            league_ids = parse_league_ids(args.league_id)
        except ValueError as e:
            print(f"Error parsing league IDs: {e}")
            sys.exit(1)
    
    if not league_ids:
        print("No league IDs specified")
        sys.exit(1)
    
    print(f"Fetching standings for leagues: {league_ids}")
    print(f"Season: {args.season}")
    print(f"Force overwrite: {args.force}")
    print()
    
    # Run the async fetch function
    try:
        asyncio.run(fetch_standings_for_leagues(league_ids, args.season, args.force))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during standings fetch: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()