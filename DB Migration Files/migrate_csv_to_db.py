#!/usr/bin/env python3
"""
Migration script to transfer CSV data to PostgreSQL database.

This script:
1. Reads existing CSV files from data/standings directory
2. Extracts league information from leagues.json
3. Creates database tables if they don't exist
4. Migrates data to PostgreSQL with proper relationships
5. Adds league names and countries for easy identification
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


class CSVToDBMigrator:
    """Handles migration of CSV data to PostgreSQL database."""
    
    def __init__(self, data_dir: str = "data", leagues_file: str = "leagues.json"):
        """Initialize the migrator.
        
        Args:
            data_dir: Path to data directory containing CSV files
            leagues_file: Path to leagues.json file
        """
        self.data_dir = Path(data_dir)
        self.leagues_file = Path(leagues_file)
        self.leagues_data = self._load_leagues_data()
        
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
        
        session.add(league)
        session.flush()  # Get the ID without committing
        
        logger.info(f"Created league: {league.name} ({league.country}) - {season}")
        return league
    
    def _create_or_get_team(self, session, team_id: int, team_name: str, league_id: int) -> Team:
        """Create or retrieve a team record.
        
        Args:
            session: Database session
            team_id: Team ID
            team_name: Team name
            league_id: League ID
            
        Returns:
            Team database object
        """
        # Check if team already exists
        team = session.query(Team).filter_by(id=team_id, league_id=league_id).first()
        
        if team:
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
        league = self._create_or_get_league(session, league_id, season)
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.error(f"Failed to read {csv_file}: {e}")
            return 0
        
        records_migrated = 0
        
        for _, row in df.iterrows():
            try:
                # Create or get team
                team = self._create_or_get_team(
                    session, 
                    int(row['team_id']), 
                    row['team_name'], 
                    league_id
                )
                
                # Check if standing already exists
                existing_standing = session.query(Standing).filter_by(
                    league_id=league_id,
                    team_id=int(row['team_id'])
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
                        team_id=int(row['team_id']),
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
                
                records_migrated += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate row for {row.get('team_name', 'unknown')}: {e}")
                continue
        
        return records_migrated
    
    def migrate_all_standings(self) -> Dict[str, int]:
        """Migrate all standings CSV files to database.
        
        Returns:
            Dictionary with migration statistics
        """
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
        
        with get_db_session() as session:
            for csv_file in csv_files:
                try:
                    records = self._migrate_standings_csv(session, csv_file)
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
                    
                except Exception as e:
                    logger.error(f"Failed to migrate {csv_file.name}: {e}")
                    session.rollback()
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
            
            return {
                'leagues': leagues_count,
                'teams': teams_count,
                'standings': standings_count
            }


def main():
    """Main migration function."""
    logger.info("Starting CSV to Database migration")
    
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
    migrator = CSVToDBMigrator()
    
    # Migrate standings data
    stats = migrator.migrate_all_standings()
    
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