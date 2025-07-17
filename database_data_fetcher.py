#!/usr/bin/env python3
"""
Database-enabled DataFetcher that saves data directly to PostgreSQL/SQLite.

This module extends the existing DataFetcher to save data to database
instead of CSV files, while maintaining backward compatibility.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from formfinder.DataFetcher import EnhancedDataFetcher, Match, Standing
from formfinder.database import (
    get_db_session, League, Team, Standing as DBStanding,
    Fixture, DataFetchLog, init_database
)
from formfinder.config import get_config

logger = logging.getLogger(__name__)


class DatabaseDataFetcher(EnhancedDataFetcher):
    """Enhanced DataFetcher that saves data to database instead of CSV files."""
    
    def __init__(self, config_path: str = "config.yaml", use_database: bool = True):
        """Initialize the database-enabled data fetcher.
        
        Args:
            config_path: Path to configuration file
            use_database: Whether to use database storage (True) or CSV files (False)
        """
        # Load configuration first
        from formfinder.config import load_config
        load_config(config_path)
        config = get_config()
        
        # Use the config directly - it should already be a DataFetcherConfig
        super().__init__(config)
        self.use_database = use_database
        self.leagues_cache = {}  # Cache for league information
        
        if use_database:
            # Initialize database
            init_database(config.get_database_url())
            logger.info("Database initialized for data storage")
    
    def _load_leagues_data(self) -> Dict[int, Dict]:
        """Load league information from leagues.json for database operations."""
        if hasattr(self, '_leagues_data_cache'):
            return self._leagues_data_cache
        
        try:
            import json
            leagues_file = Path("leagues.json")
            
            if not leagues_file.exists():
                logger.warning("leagues.json not found, using empty league data")
                return {}
            
            with open(leagues_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            leagues = {}
            for league in data.get('results', []):
                leagues[league['id']] = {
                    'name': league['name'],
                    'country': league['country']['name'],
                    'is_cup': league.get('is_cup', False)
                }
            
            self._leagues_data_cache = leagues
            logger.info(f"Loaded {len(leagues)} leagues for database operations")
            return leagues
            
        except Exception as e:
            logger.error(f"Failed to load leagues data: {e}")
            return {}
    
    def _get_or_create_league(self, session, league_id: int, season: str) -> League:
        """Get or create a league record in the database.
        
        Args:
            session: Database session
            league_id: League ID
            season: Season string
            
        Returns:
            League database object
        """
        # Check cache first
        cache_key = f"{league_id}_{season}"
        if cache_key in self.leagues_cache:
            return self.leagues_cache[cache_key]
        
        # Check if league exists in database
        league = session.query(League).filter_by(id=league_id, season=season).first()
        
        if league:
            self.leagues_cache[cache_key] = league
            return league
        
        # Get league info from loaded data
        leagues_data = self._load_leagues_data()
        league_info = leagues_data.get(league_id, {})
        
        # Create new league
        league = League(
            id=league_id,
            name=league_info.get('name', f'League {league_id}'),
            country=league_info.get('country', 'Unknown'),
            season=season
        )
        
        session.add(league)
        session.flush()  # Get the ID without committing
        
        self.leagues_cache[cache_key] = league
        logger.info(f"Created league: {league.name} ({league.country}) - {season}")
        return league
    
    def _get_or_create_team(self, session, team_id: int, team_name: str, league_id: int) -> Team:
        """Get or create a team record in the database.
        
        Args:
            session: Database session
            team_id: Team ID
            team_name: Team name
            league_id: League ID
            
        Returns:
            Team database object
        """
        # Check if team exists
        team = session.query(Team).filter_by(id=team_id, league_id=league_id).first()
        
        if team:
            # Update team name if it has changed
            if team.name != team_name:
                team.name = team_name
                team.updated_at = datetime.utcnow()
            return team
        
        # Create new team
        team = Team(
            id=team_id,
            league_id=league_id,
            name=team_name,
            short_name=team_name[:50] if len(team_name) > 50 else team_name
        )
        
        session.add(team)
        session.flush()
        
        logger.debug(f"Created team: {team_name} (ID: {team_id})")
        return team
    
    def save_matches_to_database(self, matches: List[Match], league_id: int, season: str) -> int:
        """Save matches to database.
        
        Args:
            matches: List of Match objects
            league_id: League ID
            season: Season string
            
        Returns:
            Number of matches saved
        """
        if not matches:
            logger.warning("No matches to save to database")
            return 0
        
        saved_count = 0
        
        try:
            with get_db_session() as session:
                # Get or create league
                league = self._get_or_create_league(session, league_id, season)
                
                for match in matches:
                    try:
                        # Get or create teams
                        home_team = self._get_or_create_team(
                            session, match.home_team_id, match.home_team_name, league_id
                        )
                        away_team = self._get_or_create_team(
                            session, match.away_team_id, match.away_team_name, league_id
                        )
                        
                        # Check if fixture already exists
                        existing_fixture = session.query(Fixture).filter_by(id=match.id).first()
                        
                        if existing_fixture:
                            # Update existing fixture
                            existing_fixture.status = match.status
                            existing_fixture.home_score = match.home_score
                            existing_fixture.away_score = match.away_score
                            existing_fixture.updated_at = datetime.utcnow()
                            logger.debug(f"Updated fixture {match.id}")
                        else:
                            # Parse match date and time
                            match_datetime = None
                            if match.date and match.time:
                                try:
                                    match_datetime = datetime.strptime(
                                        f"{match.date} {match.time}", "%Y-%m-%d %H:%M:%S"
                                    )
                                except ValueError:
                                    try:
                                        match_datetime = datetime.strptime(
                                            match.date, "%Y-%m-%d"
                                        )
                                    except ValueError:
                                        logger.warning(f"Could not parse date for match {match.id}: {match.date}")
                            
                            # Create new fixture
                            fixture = Fixture(
                                id=match.id,
                                league_id=league_id,
                                home_team_id=match.home_team_id,
                                away_team_id=match.away_team_id,
                                match_date=match_datetime or datetime.utcnow(),
                                status=match.status or 'unknown',
                                home_score=match.home_score,
                                away_score=match.away_score
                            )
                            
                            session.add(fixture)
                            logger.debug(f"Created fixture {match.id}")
                        
                        saved_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to save match {match.id}: {e}")
                        continue
                
                session.commit()
                logger.info(f"Saved {saved_count} matches to database")
                
        except Exception as e:
            logger.error(f"Error saving matches to database: {e}")
            return 0
        
        return saved_count
    
    def save_standings_to_database(self, standings: List[Standing], league_id: int, season: str) -> int:
        """Save standings to database.
        
        Args:
            standings: List of Standing objects
            league_id: League ID
            season: Season string
            
        Returns:
            Number of standings saved
        """
        if not standings:
            logger.warning("No standings to save to database")
            return 0
        
        saved_count = 0
        
        try:
            with get_db_session() as session:
                # Get or create league
                league = self._get_or_create_league(session, league_id, season)
                
                for standing in standings:
                    try:
                        # Get or create team
                        team = self._get_or_create_team(
                            session, standing.team_id, standing.team_name, league_id
                        )
                        
                        # Check if standing already exists
                        existing_standing = session.query(DBStanding).filter_by(
                            league_id=league_id,
                            team_id=standing.team_id
                        ).first()
                        
                        if existing_standing:
                            # Update existing standing
                            existing_standing.position = standing.position
                            existing_standing.played = standing.games_played
                            existing_standing.won = standing.wins
                            existing_standing.drawn = standing.draws
                            existing_standing.lost = standing.losses
                            existing_standing.goals_for = standing.goals_for
                            existing_standing.goals_against = standing.goals_against
                            existing_standing.goal_difference = standing.goal_difference
                            existing_standing.points = standing.points
                            existing_standing.updated_at = datetime.utcnow()
                            
                            logger.debug(f"Updated standing for {standing.team_name}")
                        else:
                            # Create new standing
                            db_standing = DBStanding(
                                league_id=league_id,
                                team_id=standing.team_id,
                                position=standing.position,
                                played=standing.games_played,
                                won=standing.wins,
                                drawn=standing.draws,
                                lost=standing.losses,
                                goals_for=standing.goals_for,
                                goals_against=standing.goals_against,
                                goal_difference=standing.goal_difference,
                                points=standing.points
                            )
                            
                            session.add(db_standing)
                            logger.debug(f"Created standing for {standing.team_name}")
                        
                        saved_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to save standing for {standing.team_name}: {e}")
                        continue
                
                session.commit()
                logger.info(f"Saved {saved_count} standings to database")
                
        except Exception as e:
            logger.error(f"Error saving standings to database: {e}")
            return 0
        
        return saved_count
    
    def log_data_fetch(self, league_id: int, data_type: str, status: str, 
                      records_count: int = 0, error_message: str = None, 
                      duration_seconds: float = None):
        """Log data fetch operation to database.
        
        Args:
            league_id: League ID
            data_type: Type of data fetched ('fixtures', 'standings', 'historical')
            status: Status of the operation ('success', 'error', 'partial')
            records_count: Number of records fetched
            error_message: Error message if any
            duration_seconds: Duration of the operation
        """
        try:
            with get_db_session() as session:
                log_entry = DataFetchLog(
                    league_id=league_id,
                    data_type=data_type,
                    status=status,
                    records_fetched=records_count,
                    error_message=error_message,
                    duration_seconds=duration_seconds
                )
                
                session.add(log_entry)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log data fetch operation: {e}")
    
    async def process_league(self, league_id: int, league_name: str, semaphore: asyncio.Semaphore):
        """Process data for a single league with database storage."""
        async with semaphore:
            start_time = datetime.utcnow()
            self.logger.info(f"Processing league: {league_name} (ID: {league_id})")
            
            season = self.config.processing.season_year
            
            if self.use_database:
                # Database storage mode
                tasks = []
                
                # Fetch historical matches
                tasks.append(('historical', self.fetch_historical_matches(league_id, league_name)))
                
                # Fetch standings
                tasks.append(('standings', self.fetch_standings_async(league_id, league_name)))
                
                # Execute tasks
                results = {}
                for task_name, task in tasks:
                    task_start = datetime.utcnow()
                    try:
                        results[task_name] = await task
                        duration = (datetime.utcnow() - task_start).total_seconds()
                        
                        # Log successful fetch
                        self.log_data_fetch(
                            league_id, task_name, 'success', 
                            len(results[task_name]), duration_seconds=duration
                        )
                        
                    except Exception as e:
                        duration = (datetime.utcnow() - task_start).total_seconds()
                        self.logger.error(f"Error in {task_name} task for {league_name}: {e}")
                        results[task_name] = []
                        
                        # Log failed fetch
                        self.log_data_fetch(
                            league_id, task_name, 'error', 
                            0, str(e), duration_seconds=duration
                        )
                
                # Save results to database
                if 'historical' in results and results['historical']:
                    saved_matches = self.save_matches_to_database(
                        results['historical'], league_id, season
                    )
                    self.logger.info(f"Saved {saved_matches} historical matches for {league_name}")
                
                if 'standings' in results and results['standings']:
                    saved_standings = self.save_standings_to_database(
                        results['standings'], league_id, season
                    )
                    self.logger.info(f"Saved {saved_standings} standings for {league_name}")
                
            else:
                # Fallback to CSV storage (original behavior)
                await super().process_league(league_id, league_name, semaphore)
            
            total_duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Completed processing {league_name} in {total_duration:.2f} seconds")
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get a summary of data stored in the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with get_db_session() as session:
                leagues_count = session.query(League).count()
                teams_count = session.query(Team).count()
                fixtures_count = session.query(Fixture).count()
                standings_count = session.query(DBStanding).count()
                
                # Get recent fetch logs
                recent_logs = session.query(DataFetchLog).order_by(
                    DataFetchLog.fetch_date.desc()
                ).limit(10).all()
                
                summary = {
                    'database_stats': {
                        'leagues': leagues_count,
                        'teams': teams_count,
                        'fixtures': fixtures_count,
                        'standings': standings_count
                    },
                    'recent_operations': [
                        {
                            'league_id': log.league_id,
                            'data_type': log.data_type,
                            'status': log.status,
                            'records': log.records_fetched,
                            'date': log.fetch_date.isoformat()
                        }
                        for log in recent_logs
                    ]
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get database summary: {e}")
            return {'error': str(e)}


def main():
    """Example usage of DatabaseDataFetcher."""
    import asyncio
    
    async def run_example():
        # Initialize database-enabled fetcher
        fetcher = DatabaseDataFetcher(use_database=True)
        
        # Process a few leagues
        league_ids = [244, 228, 235]  # German 3. Liga, Premier League, Ligue 1
        
        semaphore = asyncio.Semaphore(3)
        tasks = []
        
        for league_id in league_ids:
            league_name = f"League-{league_id}"
            tasks.append(fetcher.process_league(league_id, league_name, semaphore))
        
        await asyncio.gather(*tasks)
        
        # Print database summary
        summary = fetcher.get_database_summary()
        print("\n=== Database Summary ===")
        print(f"Leagues: {summary['database_stats']['leagues']}")
        print(f"Teams: {summary['database_stats']['teams']}")
        print(f"Fixtures: {summary['database_stats']['fixtures']}")
        print(f"Standings: {summary['database_stats']['standings']}")
    
    asyncio.run(run_example())


if __name__ == "__main__":
    main()