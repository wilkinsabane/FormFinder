#!/usr/bin/env python3
"""
Fixed migration script to transfer CSV data to PostgreSQL database.

This script fixes the duplicate key violation issue by:
1. Using global team ID checks instead of league-specific checks
2. Adding proper error handling for constraint violations
3. Implementing upsert logic for teams and standings
4. Adding data validation and cleanup
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

from formfinder.database import (
    init_database, get_db_session, League, Team, Standing,
    DataFetchLog
)
from formfinder.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixedCSVToDBMigrator:
    """Fixed migrator that handles duplicate key violations properly."""
    
    def __init__(self, data_dir: str = "data", leagues_file: str = "leagues.json"):
        """Initialize the migrator.
        
        Args:
            data_dir: Path to data directory containing CSV files
            leagues_file: Path to leagues.json file
        """
        self.data_dir = Path(data_dir)
        self.leagues_file = Path(leagues_file)
        self.leagues_data = self._load_leagues_data()
        self.processed_teams = set()  # Track processed team IDs globally
        
    def _load_leagues_data(self) -> Dict[int, Dict]:
        """Load league information from leagues.json.
        
        Returns:
            Dictionary mapping league_id to league information
        """
        try:
            with open(self.leagues_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            leagues = {}
            for league in data.get('results', []):
                leagues[league['id']] = {
                    'name': league['name'],
                    'country': league['country']['name'],
                    'is_cup': league.get('is_cup', False)
                }
            
            logger.info(f"Loaded {len(leagues)} leagues from {self.leagues_file}")
            return leagues
            
        except Exception as e:
            logger.error(f"Failed to load leagues data: {e}")
            return {}
    
    def _extract_league_info_from_filename(self, filename: str) -> Optional[Dict]:
        """Extract league ID and season from CSV filename.
        
        Args:
            filename: CSV filename like 'league_244_2024-2025_standings.csv'
            
        Returns:
            Dictionary with league_id and season, or None if parsing fails
        """
        try:
            # Remove .csv extension and split by underscores
            parts = filename.replace('.csv', '').split('_')
            
            if len(parts) >= 3 and parts[0] == 'league':
                league_id = int(parts[1])
                season = parts[2]
                
                return {
                    'league_id': league_id,
                    'season': season
                }
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse filename {filename}: {e}")
        
        return None
    
    def _create_or_get_league(self, session, league_id: int, season: str) -> League:
        """Create or retrieve a league record.
        
        Args:
            session: Database session
            league_id: League ID
            season: Season string
            
        Returns:
            League database object
        """
        # Check if league already exists
        league = session.query(League).filter_by(id=league_id, season=season).first()
        
        if league:
            return league
        
        # Get league info from loaded data
        league_info = self.leagues_data.get(league_id, {})
        
        # Create new league
        league = League(
            id=league_id,
            name=league_info.get('name', f'League {league_id}'),
            country=league_info.get('country', 'Unknown'),
            season=season
        )
        
        try:
            session.add(league)
            session.flush()  # Get the ID without committing
            logger.info(f"Created league: {league.name} ({league.country}) - {season}")
        except IntegrityError as e:
            session.rollback()
            # League might have been created by another process, try to get it again
            league = session.query(League).filter_by(id=league_id, season=season).first()
            if not league:
                raise e
            logger.debug(f"League already exists: {league.name} ({league.country}) - {season}")
        
        return league
    
    def _create_or_get_team(self, session, team_id: int, team_name: str, league_id: int) -> Team:
        """Create or retrieve a team record using global team ID.
        
        Args:
            session: Database session
            team_id: Team ID (global, not league-specific)
            team_name: Team name
            league_id: League ID (for reference, but team ID is global)
            
        Returns:
            Team database object
        """
        # Check if team already exists globally (not league-specific)
        team = session.query(Team).filter_by(id=team_id).first()
        
        if team:
            # Update league_id if it's different (team might play in multiple leagues)
            if team.league_id != league_id:
                logger.debug(f"Team {team_name} (ID: {team_id}) found in different league. "
                           f"Original league: {team.league_id}, Current league: {league_id}")
                # Keep the original league_id for consistency
            return team
        
        # Create new team only if it doesn't exist globally
        team = Team(
            id=team_id,
            league_id=league_id,
            name=team_name,
            short_name=team_name[:50] if len(team_name) > 50 else team_name
        )
        
        try:
            session.add(team)
            session.flush()
            logger.debug(f"Created team: {team_name} (ID: {team_id})")
            self.processed_teams.add(team_id)
        except IntegrityError as e:
            session.rollback()
            # Team might have been created by another process, try to get it again
            team = session.query(Team).filter_by(id=team_id).first()
            if not team:
                logger.error(f"Failed to create team {team_name} (ID: {team_id}): {e}")
                raise e
            logger.debug(f"Team already exists: {team_name} (ID: {team_id})")
        
        return team
    
    def _upsert_standing(self, session, league_id: int, team_id: int, row: pd.Series) -> bool:
        """Create or update a standing record.
        
        Args:
            session: Database session
            league_id: League ID
            team_id: Team ID
            row: CSV row data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if standing already exists
            existing_standing = session.query(Standing).filter_by(
                league_id=league_id,
                team_id=team_id
            ).first()
            
            if existing_standing:
                # Update existing record
                existing_standing.position = int(row['position'])
                existing_standing.played = int(row['games_played'])
                existing_standing.won = int(row['wins'])
                existing_standing.drawn = int(row['draws'])
                existing_standing.lost = int(row['losses'])
                existing_standing.goals_for = int(row['goals_for'])
                existing_standing.goals_against = int(row['goals_against'])
                existing_standing.goal_difference = int(row['goal_difference'])
                existing_standing.points = int(row['points'])
                existing_standing.updated_at = datetime.utcnow()
                
                logger.debug(f"Updated standing for {row['team_name']}")
            else:
                # Create new standing
                standing = Standing(
                    league_id=league_id,
                    team_id=team_id,
                    position=int(row['position']),
                    played=int(row['games_played']),
                    won=int(row['wins']),
                    drawn=int(row['draws']),
                    lost=int(row['losses']),
                    goals_for=int(row['goals_for']),
                    goals_against=int(row['goals_against']),
                    goal_difference=int(row['goal_difference']),
                    points=int(row['points'])
                )
                
                session.add(standing)
                logger.debug(f"Created standing for {row['team_name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert standing for {row.get('team_name', 'unknown')}: {e}")
            return False
    
    def _migrate_standings_csv(self, session, csv_file: Path) -> int:
        """Migrate a single standings CSV file to database.
        
        Args:
            session: Database session
            csv_file: Path to CSV file
            
        Returns:
            Number of records migrated
        """
        logger.info(f"Migrating {csv_file.name}")
        
        # Extract league info from filename
        league_info = self._extract_league_info_from_filename(csv_file.name)
        if not league_info:
            logger.error(f"Could not extract league info from {csv_file.name}")
            return 0
        
        league_id = league_info['league_id']
        season = league_info['season']
        
        # Create or get league
        try:
            league = self._create_or_get_league(session, league_id, season)
        except Exception as e:
            logger.error(f"Failed to create/get league {league_id}: {e}")
            return 0
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Read {len(df)} rows from {csv_file.name}")
        except Exception as e:
            logger.error(f"Failed to read {csv_file}: {e}")
            return 0
        
        # Validate required columns
        required_columns = [
            'team_id', 'team_name', 'position', 'games_played', 'wins', 'draws', 
            'losses', 'goals_for', 'goals_against', 'goal_difference', 'points'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in {csv_file.name}: {missing_columns}")
            return 0
        
        records_migrated = 0
        
        for index, row in df.iterrows():
            try:
                # Validate team_id
                team_id = int(row['team_id'])
                team_name = str(row['team_name']).strip()
                
                if not team_name:
                    logger.warning(f"Empty team name for team_id {team_id}, skipping")
                    continue
                
                # Create or get team
                team = self._create_or_get_team(session, team_id, team_name, league_id)
                
                # Upsert standing
                if self._upsert_standing(session, league_id, team_id, row):
                    records_migrated += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate row {index} for {row.get('team_name', 'unknown')}: {e}")
                continue
        
        return records_migrated
    
    def clear_existing_data(self) -> bool:
        """Clear existing data to avoid conflicts.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_session() as session:
                # Delete in reverse order of dependencies
                session.execute(text("DELETE FROM standings"))
                session.execute(text("DELETE FROM teams"))
                session.execute(text("DELETE FROM leagues"))
                session.execute(text("DELETE FROM data_fetch_logs WHERE data_type = 'standings_migration'"))
                session.commit()
                
                logger.info("Cleared existing migration data")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear existing data: {e}")
            return False
    
    def migrate_all_standings(self, clear_first: bool = False) -> Dict[str, int]:
        """Migrate all standings CSV files to database.
        
        Args:
            clear_first: Whether to clear existing data first
            
        Returns:
            Dictionary with migration statistics
        """
        if clear_first:
            if not self.clear_existing_data():
                return {'files_processed': 0, 'records_migrated': 0, 'errors': 1}
        
        standings_dir = self.data_dir / "standings"
        
        if not standings_dir.exists():
            logger.error(f"Standings directory not found: {standings_dir}")
            return {'files_processed': 0, 'records_migrated': 0, 'errors': 1}
        
        csv_files = list(standings_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to migrate")
        
        stats = {
            'files_processed': 0,
            'records_migrated': 0,
            'errors': 0
        }
        
        for csv_file in csv_files:
            try:
                with get_db_session() as session:
                    records = self._migrate_standings_csv(session, csv_file)
                    
                    if records > 0:
                        stats['records_migrated'] += records
                        stats['files_processed'] += 1
                        
                        # Log the migration
                        league_info = self._extract_league_info_from_filename(csv_file.name)
                        if league_info:
                            log_entry = DataFetchLog(
                                league_id=league_info['league_id'],
                                data_type='standings_migration',
                                status='success',
                                records_fetched=records
                            )
                            session.add(log_entry)
                        
                        session.commit()
                        logger.info(f"Successfully migrated {records} records from {csv_file.name}")
                    else:
                        logger.warning(f"No records migrated from {csv_file.name}")
                        stats['errors'] += 1
                        
            except Exception as e:
                logger.error(f"Failed to migrate {csv_file.name}: {e}")
                stats['errors'] += 1
        
        return stats
    
    def verify_migration(self) -> Dict[str, int]:
        """Verify the migration by counting records in database.
        
        Returns:
            Dictionary with verification statistics
        """
        with get_db_session() as session:
            leagues_count = session.query(League).count()
            teams_count = session.query(Team).count()
            standings_count = session.query(Standing).count()
            
            logger.info(f"Database contains:")
            logger.info(f"  - {leagues_count} leagues")
            logger.info(f"  - {teams_count} teams")
            logger.info(f"  - {standings_count} standings")
            
            # Check for potential issues
            duplicate_teams = session.execute(text(
                "SELECT id, COUNT(*) as count FROM teams GROUP BY id HAVING COUNT(*) > 1"
            )).fetchall()
            
            if duplicate_teams:
                logger.warning(f"Found {len(duplicate_teams)} duplicate team IDs")
                for team_id, count in duplicate_teams:
                    logger.warning(f"  Team ID {team_id}: {count} duplicates")
            
            return {
                'leagues': leagues_count,
                'teams': teams_count,
                'standings': standings_count,
                'duplicate_teams': len(duplicate_teams)
            }


def main():
    """Main migration function."""
    logger.info("Starting FIXED CSV to Database migration")
    
    # Load and initialize configuration
    from formfinder.config import load_config
    load_config("config.yaml")
    config = get_config()
    
    # Check if we should use PostgreSQL
    if config.database.type == "postgresql":
        logger.info("Using PostgreSQL database")
        db_url = config.get_database_url()
    else:
        logger.info("Using SQLite database")
        db_url = config.get_database_url()
    
    # Initialize database with the configured URL
    init_database(db_url)
    
    # Create migrator and run migration
    migrator = FixedCSVToDBMigrator()
    
    # Ask user if they want to clear existing data
    clear_first = input("Clear existing data before migration? (y/N): ").lower().startswith('y')
    
    # Migrate standings data
    stats = migrator.migrate_all_standings(clear_first=clear_first)
    
    logger.info("Migration completed!")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Records migrated: {stats['records_migrated']}")
    logger.info(f"Errors: {stats['errors']}")
    
    # Verify migration
    verification = migrator.verify_migration()
    
    if stats['errors'] == 0:
        logger.info("✅ Migration completed successfully!")
    else:
        logger.warning(f"⚠️ Migration completed with {stats['errors']} errors")
    
    return stats


if __name__ == "__main__":
    main()