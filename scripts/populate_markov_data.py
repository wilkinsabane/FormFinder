#!/usr/bin/env python3
"""
Markov Data Population Pipeline

This script processes historical fixture data to populate:
1. Team performance states
2. Markov transition matrices
3. Markov features for existing fixtures

Usage:
    python scripts/populate_markov_data.py [--league-id LEAGUE_ID] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Optional, List
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from formfinder.config import load_config, get_config
    from formfinder.markov_state_classifier import MarkovStateClassifier
    from formfinder.markov_transition_calculator import MarkovTransitionCalculator
    from formfinder.markov_feature_generator import MarkovFeatureGenerator
    from formfinder.logger import get_logger
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    sys.exit(1)

# Initialize console and logger
console = Console()
logger = get_logger(__name__)
# Set debug level for detailed logging
logger.setLevel(logging.DEBUG)

class MarkovDataPopulator:
    """Pipeline for populating historical Markov chain data."""
    
    def __init__(self, config=None):
        """Initialize the Markov data populator.
        
        Args:
            config: Configuration object. If None, loads from default config.
        """
        try:
            # Load configuration
            if config is None:
                load_config()
                config = get_config()
            self.config = config
            
            # Initialize database connection
            self.engine = create_engine(
                config.get_database_url(),
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            Session = sessionmaker(bind=self.engine)
            self.db_session = Session()
            
            # Test connection
            self.db_session.execute(text("SELECT 1"))
            logger.info("✅ Database connection established")
            
            # Initialize Markov components
            self.state_classifier = MarkovStateClassifier()
            self.transition_calculator = MarkovTransitionCalculator()
            self.feature_generator = MarkovFeatureGenerator()
            
            logger.info("✅ Markov components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MarkovDataPopulator: {e}")
            raise
    
    def get_leagues_to_process(self, league_id: Optional[int] = None) -> List[int]:
        """Get list of league IDs to process.
        
        Args:
            league_id: Specific league ID to process. If None, processes all leagues.
            
        Returns:
            List of league IDs to process.
        """
        try:
            if league_id:
                return [league_id]
            
            # Get all leagues with fixtures
            query = """
            SELECT DISTINCT league_id 
            FROM fixtures 
            WHERE home_score IS NOT NULL
                AND away_score IS NOT NULL
            ORDER BY league_id
            """
            
            result = self.db_session.execute(text(query))
            league_ids = [row[0] for row in result.fetchall()]
            
            logger.info(f"Found {len(league_ids)} leagues with completed fixtures")
            return league_ids
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting leagues: {e}")
            return []
    
    def get_date_range(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> tuple:
        """Get date range for processing.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. If None, uses earliest fixture date.
            end_date: End date in YYYY-MM-DD format. If None, uses latest fixture date.
            
        Returns:
            Tuple of (start_date, end_date) as datetime objects.
        """
        try:
            if start_date and end_date:
                return (
                    datetime.strptime(start_date, '%Y-%m-%d'),
                    datetime.strptime(end_date, '%Y-%m-%d')
                )
            
            # Get date range from fixtures
            query = """
            SELECT 
                MIN(match_date) as min_date,
                MAX(match_date) as max_date
            FROM fixtures 
            WHERE home_score IS NOT NULL 
                AND away_score IS NOT NULL
            """
            
            result = self.db_session.execute(text(query))
            row = result.fetchone()
            
            min_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else row[0]
            max_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else row[1]
            
            logger.info(f"Processing date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            return min_date, max_date
            
        except Exception as e:
            logger.error(f"Error determining date range: {e}")
            # Default to last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            return start_date, end_date
    
    def populate_team_states(self, league_ids: List[int], start_date: datetime, end_date: datetime) -> bool:
        """Populate team performance states for the given period.
        
        Args:
            league_ids: List of league IDs to process.
            start_date: Start date for processing.
            end_date: End date for processing.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            rprint(Panel.fit(
                "[bold blue]🎯 Populating Team Performance States[/bold blue]",
                border_style="blue"
            ))
            
            total_processed = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                league_task = progress.add_task("Processing leagues", total=len(league_ids))
                
                for league_id in league_ids:
                    progress.update(league_task, description=f"League {league_id}")
                    
                    # Get all teams in this league
                    teams_query = """
                    SELECT DISTINCT team_id 
                    FROM (
                        SELECT home_team_id as team_id FROM fixtures WHERE league_id = :league_id
                        UNION
                        SELECT away_team_id as team_id FROM fixtures WHERE league_id = :league_id
                    ) teams
                    """
                    
                    from formfinder.database import get_db_session
                    with get_db_session() as session:
                        teams = session.execute(
                            text(teams_query),
                            {'league_id': league_id}
                        ).fetchall()
                    
                    processed = 0
                    # Process states for each team in this league
                    for team_row in teams:
                        team_id = team_row[0]
                        try:
                            states = self.state_classifier.process_team_states(
                                team_id=team_id,
                                league_id=league_id,
                                start_date=start_date,
                                end_date=end_date
                            )
                            processed += len(states)
                        except Exception as e:
                            logger.warning(f"Failed to process team {team_id} in league {league_id}: {e}")
                            continue
                    
                    total_processed += processed
                    logger.info(f"Processed {processed} team states for league {league_id}")
                    
                    progress.advance(league_task)
            
            logger.info(f"✅ Successfully processed {total_processed} team performance states")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error populating team states: {e}")
            return False
    
    def populate_transition_matrices(self, league_ids: List[int]) -> bool:
        """Populate Markov transition matrices for all leagues.
        
        Args:
            league_ids: List of league IDs to process.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            rprint(Panel.fit(
                "[bold green]🔗 Populating Transition Matrices[/bold green]",
                border_style="green"
            ))
            
            total_matrices = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                league_task = progress.add_task("Processing leagues", total=len(league_ids))
                
                for league_id in league_ids:
                    progress.update(league_task, description=f"League {league_id} matrices")
                    
                    # Get teams in this league
                    teams_query = """
                    SELECT DISTINCT team_id 
                    FROM (
                        SELECT home_team_id as team_id FROM fixtures WHERE league_id = :league_id
                        UNION
                        SELECT away_team_id as team_id FROM fixtures WHERE league_id = :league_id
                    ) teams
                    """
                    
                    result = self.db_session.execute(text(teams_query), {'league_id': league_id})
                    team_ids = [row[0] for row in result.fetchall()]
                    
                    # Process transition matrices for each team
                    for team_id in team_ids:
                        # Process all contexts (overall, home, away) for this team
                        results = self.transition_calculator.process_team_transitions(
                            team_id=team_id,
                            league_id=league_id,
                            contexts=['overall', 'home', 'away']
                        )
                        
                        total_matrices += len(results)  # Count all contexts processed
                    
                    # Note: League-wide transitions are handled per team above
                    
                    logger.info(f"Processed transition matrices for league {league_id} ({len(team_ids)} teams)")
                    progress.advance(league_task)
            
            logger.info(f"✅ Successfully created {total_matrices} transition matrices")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error populating transition matrices: {e}")
            return False
    
    def populate_markov_features(self, league_ids: List[int], start_date: datetime, end_date: datetime) -> bool:
        """Populate Markov features for fixtures in the given period.
        
        Args:
            league_ids: List of league IDs to process.
            start_date: Start date for processing.
            end_date: End date for processing.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            rprint(Panel.fit(
                "[bold yellow]📊 Populating Markov Features[/bold yellow]",
                border_style="yellow"
            ))
            
            total_features = 0
            
            # Get fixtures in date range
            fixtures_query = """
            SELECT id, home_team_id, away_team_id, match_date, league_id
            FROM fixtures 
            WHERE match_date BETWEEN :start_date AND :end_date
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
            """
            
            if league_ids:
                league_filter = ','.join(map(str, league_ids))
                fixtures_query += f" AND league_id IN ({league_filter})"
            
            fixtures_query += " ORDER BY match_date"
            
            result = self.db_session.execute(text(fixtures_query), {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            })
            fixtures = result.fetchall()
            
            logger.info(f"Found {len(fixtures)} fixtures to process for Markov features")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                fixture_task = progress.add_task("Processing fixtures", total=len(fixtures))
                
                for fixture in fixtures:
                    fixture_id, home_team_id, away_team_id, match_date, league_id = fixture
                    logger.debug(f"Processing fixture {fixture_id}: {home_team_id} vs {away_team_id} on {match_date}")
                    
                    try:
                        # Generate and store individual team features
                        # Home team features
                        logger.debug(f"Generating home team features for team {home_team_id}")
                        home_features = self.feature_generator.generate_team_features(
                            team_id=home_team_id,
                            league_id=league_id,
                            reference_date=match_date,
                            context='overall'
                        )
                        
                        if home_features:
                            logger.debug(f"Home team features generated: {len(home_features)} features")
                            self.feature_generator.store_team_features(
                                team_id=home_team_id,
                                league_id=league_id,
                                features=home_features,
                                fixture_id=fixture_id
                            )
                            total_features += 1
                            logger.debug(f"Home team features stored. Total so far: {total_features}")
                        else:
                            logger.warning(f"No home team features generated for team {home_team_id}")
                        
                        # Away team features
                        logger.debug(f"Generating away team features for team {away_team_id}")
                        away_features = self.feature_generator.generate_team_features(
                            team_id=away_team_id,
                            league_id=league_id,
                            reference_date=match_date,
                            context='overall'
                        )
                        
                        if away_features:
                            logger.debug(f"Away team features generated: {len(away_features)} features")
                            self.feature_generator.store_team_features(
                                team_id=away_team_id,
                                league_id=league_id,
                                features=away_features,
                                fixture_id=fixture_id
                            )
                            total_features += 1
                            logger.debug(f"Away team features stored. Total so far: {total_features}")
                        else:
                            logger.warning(f"No away team features generated for team {away_team_id}")
                    
                    except Exception as e:
                        logger.warning(f"Error processing fixture {fixture_id}: {e}")
                        import traceback
                        logger.debug(f"Full traceback: {traceback.format_exc()}")
                    
                    progress.advance(fixture_task)
            
            logger.info(f"✅ Successfully generated Markov features for {total_features} fixtures")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error populating Markov features: {e}")
            return False
    
    def run_pipeline(self, league_id: Optional[int] = None, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> bool:
        """Run the complete Markov data population pipeline.
        
        Args:
            league_id: Specific league ID to process. If None, processes all leagues.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            start_time = datetime.now()
            
            rprint(Panel.fit(
                "[bold magenta]🚀 Markov Data Population Pipeline[/bold magenta]\n"
                "[dim]Populating historical Markov chain data[/dim]",
                border_style="magenta"
            ))
            
            # Get processing parameters
            league_ids = self.get_leagues_to_process(league_id)
            if not league_ids:
                logger.error("No leagues found to process")
                return False
            
            date_start, date_end = self.get_date_range(start_date, end_date)
            
            # Display processing summary
            summary_table = Table(title="🎯 Processing Summary", border_style="cyan")
            summary_table.add_column("Parameter", style="cyan")
            summary_table.add_column("Value", style="white")
            
            summary_table.add_row("Leagues", f"{len(league_ids)} leagues")
            summary_table.add_row("Date Range", f"{date_start.strftime('%Y-%m-%d')} to {date_end.strftime('%Y-%m-%d')}")
            summary_table.add_row("Duration", f"{(date_end - date_start).days} days")
            
            console.print(summary_table)
            
            # Step 1: Populate team performance states
            logger.info("🎯 Step 1: Populating team performance states")
            if not self.populate_team_states(league_ids, date_start, date_end):
                logger.error("Failed to populate team states")
                return False
            
            # Step 2: Populate transition matrices
            logger.info("🔗 Step 2: Populating transition matrices")
            if not self.populate_transition_matrices(league_ids):
                logger.error("Failed to populate transition matrices")
                return False
            
            # Step 3: Populate Markov features
            logger.info("📊 Step 3: Populating Markov features")
            if not self.populate_markov_features(league_ids, date_start, date_end):
                logger.error("Failed to populate Markov features")
                return False
            
            # Pipeline completed successfully
            elapsed_time = datetime.now() - start_time
            
            rprint(Panel.fit(
                f"[bold green]✅ Pipeline Completed Successfully[/bold green]\n"
                f"[dim]Total time: {elapsed_time.total_seconds():.1f} seconds[/dim]",
                border_style="green"
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
        finally:
            # Cleanup
            try:
                self.db_session.close()
            except:
                pass

def main():
    """Main entry point for the Markov data population script."""
    parser = argparse.ArgumentParser(
        description="Populate historical Markov chain data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all leagues for the last year
  python scripts/populate_markov_data.py
  
  # Process specific league
  python scripts/populate_markov_data.py --league-id 39
  
  # Process specific date range
  python scripts/populate_markov_data.py --start-date 2023-01-01 --end-date 2023-12-31
  
  # Process specific league and date range
  python scripts/populate_markov_data.py --league-id 39 --start-date 2023-08-01 --end-date 2024-05-31
        """
    )
    
    parser.add_argument(
        '--league-id',
        type=int,
        help='Specific league ID to process (if not specified, processes all leagues)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format (if not specified, uses earliest fixture date)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format (if not specified, uses latest fixture date)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize and run pipeline
        populator = MarkovDataPopulator()
        success = populator.run_pipeline(
            league_id=args.league_id,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()