#!/usr/bin/env python3
"""
Standalone script to fetch missing stadium_city data for fixtures.

This script identifies fixtures with NULL stadium_city values and attempts
to fetch detailed match information to populate the missing data.
"""

import asyncio
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from formfinder.database import get_db_session
from formfinder.DataFetcher import DataFetcher
from formfinder.config import load_config


class StadiumDataFetcher:
    """Fetches missing stadium data for fixtures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.logger = logging.getLogger(__name__)
        
    async def get_fixtures_missing_stadium_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get fixtures that are missing stadium_city data."""
        from formfinder.database import Fixture, Team, League
        from sqlalchemy.orm import aliased
        
        with get_db_session() as session:
            # Create aliases for home and away teams
            home_team = aliased(Team)
            away_team = aliased(Team)
            
            query = session.query(
                Fixture.id.label('fixture_id'),
                Fixture.api_fixture_id,
                Fixture.home_team_id,
                Fixture.away_team_id,
                Fixture.match_date,
                Fixture.league_id,
                League.season,
                home_team.name.label('home_team_name'),
                away_team.name.label('away_team_name'),
                League.name.label('league_name')
            ).join(
                home_team, Fixture.home_team_id == home_team.id
            ).join(
                away_team, Fixture.away_team_id == away_team.id
            ).join(
                League, Fixture.league_id == League.league_pk
            ).filter(
                Fixture.stadium_city.is_(None),
                Fixture.match_date.isnot(None),
                Fixture.api_fixture_id.isnot(None),
                ~Fixture.api_fixture_id.like('mock_%')
            ).order_by(
                Fixture.match_date.desc()
            )
            
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            return [{
                'fixture_id': r.fixture_id,
                'api_fixture_id': r.api_fixture_id,
                'home_team_id': r.home_team_id,
                'away_team_id': r.away_team_id,
                'match_date': r.match_date,
                'league_id': r.league_id,
                'season': r.season,
                'home_team_name': r.home_team_name,
                'away_team_name': r.away_team_name,
                'league_name': r.league_name
            } for r in results]
    
    async def fetch_detailed_data_for_fixture(self, fixture: Dict[str, Any]) -> bool:
        """Fetch detailed data for a single fixture."""
        try:
            self.logger.info(f"Fetching detailed data for fixture {fixture['fixture_id']}: "
                           f"{fixture['home_team_name']} vs {fixture['away_team_name']}")
            
            # Fetch detailed match info using api_fixture_id
            detailed_data = await self.data_fetcher.fetch_detailed_match_info(
                fixture['api_fixture_id']
            )
            
            if detailed_data:
                # Save the detailed data to database
                await self.data_fetcher.save_detailed_match_data(
                    detailed_data
                )
                self.logger.info(f"Successfully fetched detailed data for fixture {fixture['fixture_id']}")
                return True
            else:
                self.logger.warning(f"No detailed data returned for fixture {fixture['fixture_id']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error fetching detailed data for fixture {fixture['fixture_id']}: {e}")
            return False
    
    async def process_fixtures_batch(self, fixtures: List[Dict[str, Any]], 
                                   delay_seconds: float = 1.0) -> Dict[str, int]:
        """Process a batch of fixtures with rate limiting."""
        results = {
            'success': 0,
            'failed': 0,
            'total': len(fixtures)
        }
        
        for i, fixture in enumerate(fixtures):
            self.logger.info(f"Processing fixture {i+1}/{len(fixtures)}")
            
            success = await self.fetch_detailed_data_for_fixture(fixture)
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
            
            # Rate limiting - wait between requests
            if i < len(fixtures) - 1:  # Don't wait after the last request
                await asyncio.sleep(delay_seconds)
        
        return results
    
    async def run(self, limit: Optional[int] = None, delay_seconds: float = 1.0) -> None:
        """Main execution method."""
        self.logger.info("Starting stadium data fetcher...")
        
        # Get fixtures missing stadium data
        fixtures = await self.get_fixtures_missing_stadium_data(limit)
        
        if not fixtures:
            self.logger.info("No fixtures found with missing stadium data")
            return
        
        self.logger.info(f"Found {len(fixtures)} fixtures missing stadium data")
        
        # Process fixtures in batches
        start_time = time.time()
        results = await self.process_fixtures_batch(fixtures, delay_seconds)
        end_time = time.time()
        
        # Log summary
        self.logger.info(f"\n=== SUMMARY ===")
        self.logger.info(f"Total fixtures processed: {results['total']}")
        self.logger.info(f"Successful: {results['success']}")
        self.logger.info(f"Failed: {results['failed']}")
        self.logger.info(f"Success rate: {results['success']/results['total']*100:.1f}%")
        self.logger.info(f"Total time: {end_time - start_time:.1f} seconds")
        
        # Check updated stadium data coverage
        await self.check_stadium_data_coverage()
    
    async def check_stadium_data_coverage(self) -> None:
        """Check current stadium data coverage."""
        from formfinder.database import Fixture
        from sqlalchemy import func
        
        with get_db_session() as session:
            result = session.query(
                func.count(Fixture.id).label('total_fixtures'),
                func.count(Fixture.stadium_city).label('fixtures_with_stadium_city')
            ).filter(
                Fixture.match_date.isnot(None)
            ).first()
            
            total = result.total_fixtures
            with_stadium = result.fixtures_with_stadium_city
            missing_stadium = total - with_stadium
            coverage_pct = (with_stadium / total * 100) if total > 0 else 0
            
            self.logger.info(f"\n=== STADIUM DATA COVERAGE ===")
            self.logger.info(f"Total fixtures: {total}")
            self.logger.info(f"With stadium_city: {with_stadium} ({coverage_pct:.1f}%)")
            self.logger.info(f"Missing stadium_city: {missing_stadium}")


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('stadium_data_fetcher.log')
        ]
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fetch missing stadium data for fixtures")
    parser.add_argument('--limit', type=int, help="Limit number of fixtures to process")
    parser.add_argument('--delay', type=float, default=1.0, 
                       help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument('--log-level', default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument('--config', default="config.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and run fetcher
    fetcher = StadiumDataFetcher(config)
    await fetcher.run(limit=args.limit, delay_seconds=args.delay)


if __name__ == "__main__":
    asyncio.run(main())