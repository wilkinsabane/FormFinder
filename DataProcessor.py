import pandas as pd
import logging
import os
import json
from datetime import datetime, timedelta

# Database imports
from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy import desc, func
from database_setup import SessionLocal, League, Team, Match as DBMatch, HighFormTeam as DBHighFormTeam

from typing import Optional
# Configuration import
from config.config import app_config, DataProcessorAppConfig

# Configure logging (will be managed by Prefect or global config later)
# For now, ensure basic logging is set up if run standalone, but respect AppConfig log_dir
# This initial setup might be overridden by Prefect's logging or a central logger setup.
if app_config and app_config.data_processor and app_config.data_processor.log_dir:
    LOG_PROCESSOR_DIR_PATH = app_config.data_processor.log_dir
else:
    LOG_PROCESSOR_DIR_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'logs') # Fallback
os.makedirs(LOG_PROCESSOR_DIR_PATH, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_PROCESSOR_DIR_PATH,'data_processor.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Adding a module-specific logger
logger = logging.getLogger(__name__)
# Ensure console output for Prefect tasks if needed, or rely on Prefect's log capture.
# Prefect usually captures stdout/stderr and logger outputs.
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(console_handler)
# logger.propagate = False # To avoid double logging if root logger also has stream handler


class DataProcessor:
    """Processes match data from DB to identify and store high-form teams in DB."""
    
    def __init__(self, db_session: Session, config: DataProcessorAppConfig):
        self.db_session = db_session
        self.config = config
        self.recent_period = config.recent_period
        self.win_rate_threshold = config.win_rate_threshold
        self.season_year = config.season_year
        logger.info(f"Initialized DataProcessor with recent_period={self.recent_period}, win_rate_threshold={self.win_rate_threshold}, season_year={self.season_year}")

    def calculate_team_win_rate(self, team_db_id: int) -> float:
        """
        Calculate the win rate for a team based on their last N finished, non-fixture games
        within the current season_year.
        """
        # Query for the team's recent matches (home or away)
        # Ensure matches are finished, not fixtures, and within the season_year
        # Note: DBMatch.date is a Date object, not DateTime.
        
        # Get current date to filter matches up to today
        today = datetime.utcnow().date()

        team_matches_query = self.db_session.query(DBMatch).join(League).\
            filter(
                League.standings.any(season_year=self.season_year), # Ensure league is active in this season
                (DBMatch.home_team_id == team_db_id) | (DBMatch.away_team_id == team_db_id),
                DBMatch.status == 'finished',
                DBMatch.is_fixture == 0, # Explicitly non-fixtures
                DBMatch.date <= today, # Only consider matches up to today
                # Add a filter for season_year if matches span multiple years and league does too
                # This might require joining DBMatch.league and checking League.season_year or similar
                # For now, assuming matches within a league are generally for one main season at a time
                # or that DataFetcher is already season-aware.
                # A simple way: if DBMatch had a season_year field, or filter by date range.
                # Let's assume for now, the `is_fixture == 0` and `status == 'finished'` along with
                # processing per league (which is tied to a season in DataFetcher) is enough.
                # We might need to refine this if historical data for a league_id spans many seasons.
            ).order_by(desc(DBMatch.date), desc(DBMatch.time)).limit(self.recent_period)
        
        recent_matches = team_matches_query.all()

        if not recent_matches:
            logger.debug(f"No recent finished matches found for team DB ID {team_db_id} in season {self.season_year}.")
            return 0.0

        wins = 0
        valid_games_for_win_rate = 0

        for match in recent_matches:
            if match.home_score is None or match.away_score is None:
                logger.warning(f"Match API ID {match.api_match_id} for team {team_db_id} has no scores, skipping for win rate.")
                continue
            
            valid_games_for_win_rate += 1
            if (match.home_team_id == team_db_id and match.home_score > match.away_score) or \
               (match.away_team_id == team_db_id and match.away_score > match.home_score):
                wins += 1
        
        if valid_games_for_win_rate == 0:
            logger.debug(f"No valid games with scores for win rate calculation for team DB ID {team_db_id} among recent matches.")
            return 0.0
        
        win_rate = wins / valid_games_for_win_rate
        logger.debug(f"Team DB ID {team_db_id}: win rate {win_rate:.2f} over {valid_games_for_win_rate} valid recent games in season {self.season_year}.")
        return win_rate

    def process_league(self, league_db_id: int, league_name: str):
        """Process a single league: find all teams, calculate their form, and save high-form teams to DB."""
        logger.info(f"Processing league: {league_name} (DB ID: {league_db_id}) for season {self.season_year}")
        
        teams_from_standings = self.db_session.query(Team).join(DBStandings, Team.id == DBStandings.team_id).\
            filter(DBStandings.league_id == league_db_id).\
            filter(DBStandings.season_year == self.season_year).distinct().all()

        if not teams_from_standings:
            logger.info(f"No teams found in standings for league {league_name} (DB ID: {league_db_id}) for season {self.season_year}. Skipping form calculation.")
            return

        logger.info(f"Found {len(teams_from_standings)} teams in standings for league {league_name} (DB ID: {league_db_id}), season {self.season_year}.")
        
        high_form_teams_count = 0
        for team_obj in teams_from_standings:
            team_db_id = team_obj.id
            team_api_id = team_obj.api_team_id

            win_rate = self.calculate_team_win_rate(team_db_id)

            if win_rate >= self.win_rate_threshold:
                existing_high_form_entry = self.db_session.query(DBHighFormTeam).filter_by(
                    league_id=league_db_id,
                    team_id=team_db_id,
                    season_year=self.season_year,
                    recent_period_games=self.recent_period
                ).first()

                if existing_high_form_entry:
                    if abs(existing_high_form_entry.win_rate - win_rate) > 0.001:
                        existing_high_form_entry.win_rate = win_rate
                        existing_high_form_entry.calculated_at = datetime.utcnow()
                        logger.info(f"Updating high-form entry for team API ID {team_api_id} in league {league_name} to win_rate {win_rate:.2f}")
                    else:
                        logger.debug(f"High-form entry for team API ID {team_api_id} in league {league_name} (win_rate {win_rate:.2f}) is unchanged.")
                else:
                    new_high_form_entry = DBHighFormTeam(
                        league_id=league_db_id,
                        team_id=team_db_id,
                        win_rate=win_rate,
                        recent_period_games=self.recent_period,
                        season_year=self.season_year
                    )
                    self.db_session.add(new_high_form_entry)
                    logger.info(f"Adding new high-form entry for team API ID {team_api_id} in league {league_name} with win_rate {win_rate:.2f}")
                
                high_form_teams_count +=1
        
        if high_form_teams_count > 0:
            try:
                self.db_session.commit()
                logger.info(f"Committed {high_form_teams_count} high-form team updates/inserts for league {league_name} (DB ID: {league_db_id}), season {self.season_year}.")
            except Exception as e:
                self.db_session.rollback()
                logger.error(f"DB error committing high-form teams for league {league_name}: {e}", exc_info=True)
        else:
            logger.info(f"No teams met high-form threshold for league {league_name} (DB ID: {league_db_id}), season {self.season_year}.")

    def run(self):
        """Process all leagues in the database for the configured season."""
        logger.info(f"Starting DataProcessor run for season {self.season_year}...")

        leagues_to_process = self.db_session.query(League).\
            join(DBStandings, League.id == DBStandings.league_id).\
            filter(DBStandings.season_year == self.season_year).\
            distinct().all()

        if not leagues_to_process:
            logger.warning(f"No leagues found with standings for season {self.season_year}. DataProcessor run will not process any leagues.")
            return

        logger.info(f"Found {len(leagues_to_process)} leagues with standings for season {self.season_year} to process.")

        for league_obj in leagues_to_process:
            self.process_league(league_db_id=league_obj.id, league_name=league_obj.name)
        
        logger.info(f"DataProcessor run finished for season {self.season_year}.")


from prefect import task, get_run_logger as prefect_get_run_logger # Alias to avoid clash

@task(name="Run Data Processor")
def run_data_processor_task(db_session_override: Optional[Session] = None):
    """Prefect task to run the DataProcessor."""
    task_logger = prefect_get_run_logger() # Use Prefect's logger for task-level logging
    if not app_config:
        task_logger.error("DataProcessor: Global app_config not loaded. Cannot proceed.")
        raise ValueError("Global app_config not loaded.")

    session_managed_locally = False
    if db_session_override:
        db_sess = db_session_override
    else:
        db_sess = SessionLocal()
        session_managed_locally = True
        task_logger.info("DataProcessor task created its own DB session.")
    
    try:
        processor_config = app_config.data_processor
        # Ensure season_year consistency (could also be validated in Pydantic model if DataFetcher config is passed)
        if processor_config.season_year != app_config.data_fetcher.processing.season_year:
            task_logger.warning(f"Season year mismatch! Processor: {processor_config.season_year}, Fetcher: {app_config.data_fetcher.processing.season_year}. Using processor's.")
            # Decide on a consistent strategy: e.g., always use fetcher's, or make them a single top-level config.
            # For now, proceeding with processor_config.season_year as passed.

        processor = DataProcessor(
            db_session=db_sess,
            config=processor_config
        )
        processor.run()
        task_logger.info("DataProcessor task completed successfully.")
    except Exception as e:
        task_logger.error(f"DataProcessor: Critical error during execution: {e}", exc_info=True)
        raise
    finally:
        if session_managed_locally:
            db_sess.close()
            task_logger.info("DataProcessor task closed its locally managed DB session.")

if __name__ == "__main__":
    # Example of how to run the task directly (for testing)
    # if app_config: # Ensure config is loaded for standalone test
    #     run_data_processor_task()
    # else:
    #     print("Cannot run standalone DataProcessor task: app_config not loaded (config.yaml missing or invalid).")
    pass
