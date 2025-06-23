import pandas as pd
import os
from datetime import datetime, date as DateObject, timedelta
import logging
import json # For loading sdata_init_config.json temporarily

# Database imports
from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy import and_, or_, func
from database_setup import SessionLocal, League, Team, Match as DBMatch, HighFormTeam as DBHighFormTeam

from typing import Optional
# Configuration import
from config.config import app_config, PredictorOutputterAppConfig


# Configure logging (similar to DataProcessor, respect AppConfig)
if app_config and app_config.predictor_outputter and app_config.predictor_outputter.log_dir:
    LOG_PREDICTOR_DIR_PATH = app_config.predictor_outputter.log_dir
else:
    LOG_PREDICTOR_DIR_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'logs') # Fallback
os.makedirs(LOG_PREDICTOR_DIR_PATH, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_PREDICTOR_DIR_PATH,'predictor_outputter.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# console_handler_po = logging.StreamHandler() # Avoid re-defining if using root logger's stream
# console_handler_po.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(console_handler_po)
# logger.propagate = False


class PredictorOutputter:
    def __init__(self, db_session: Session, config: PredictorOutputterAppConfig):
        self.db_session = db_session
        self.config = config
        self.season_year = config.season_year
        self.recent_period_for_form = config.recent_period_for_form
        self.days_ahead_default = config.days_ahead # For default in generate_predictions
        logger.info(f"Initialized PredictorOutputter for season {self.season_year}, form period {self.recent_period_for_form}, days ahead {self.days_ahead_default}")

    def get_upcoming_fixtures(self, days_ahead: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieves upcoming fixtures from the database for the configured season,
        up to 'days_ahead'. Uses class default if days_ahead not provided.
        """
        current_days_ahead = days_ahead if days_ahead is not None else self.days_ahead_default
        today = DateObject.today()
        future_date_limit = today + timedelta(days=current_days_ahead)

        fixtures_query = self.db_session.query(
                DBMatch.api_match_id.label("match_id"),
                DBMatch.date,
                DBMatch.time,
                League.name.label("league_name"),
                League.id.label("league_db_id"), # Keep DB ID for internal use
                Team_home.name.label("home_team_name"),
                Team_home.id.label("home_team_db_id"), # DB ID
                Team_away.name.label("away_team_name"),
                Team_away.id.label("away_team_db_id")  # DB ID
            ).\
            join(League, DBMatch.league_id == League.id).\
            join(Team_home, DBMatch.home_team_id == Team_home.id).\
            join(Team_away, DBMatch.away_team_id == Team_away.id).\
            filter(DBMatch.is_fixture == 1).\
            filter(DBMatch.date >= today).\
            filter(DBMatch.date <= future_date_limit).\
            filter(League.standings.any(season_year=self.season_year)).\
            order_by(DBMatch.date, DBMatch.time) # Ensure chronological order

        # Aliases for team joins
        Team_home = aliased(Team, name="home_team_alias")
        Team_away = aliased(Team, name="away_team_alias")
        
        # Rebuild query with aliases (SQLAlchemy quirk, alias needs to be defined before use in join)
        fixtures_query = self.db_session.query(
                DBMatch.api_match_id.label("match_id"),
                DBMatch.date,
                DBMatch.time,
                League.name.label("league_name"),
                League.id.label("league_db_id"),
                Team_home.name.label("home_team_name"),
                Team_home.id.label("home_team_db_id"),
                Team_away.name.label("away_team_name"),
                Team_away.id.label("away_team_db_id")
            ).\
            join(League, DBMatch.league_id == League.id).\
            join(Team_home, DBMatch.home_team_id == Team_home.id).\
            join(Team_away, DBMatch.away_team_id == Team_away.id).\
            filter(DBMatch.is_fixture == 1).\
            filter(DBMatch.date >= today).\
            filter(DBMatch.date <= future_date_limit).\
            filter(League.standings.any(season_year=self.season_year)).\
            order_by(DBMatch.date, DBMatch.time)

        fixtures_df = pd.read_sql(fixtures_query.statement, self.db_session.bind)
        logger.info(f"Fetched {len(fixtures_df)} upcoming fixtures from DB for season {self.season_year} up to {future_date_limit}.")
        return fixtures_df

    def get_high_form_teams_for_season(self) -> pd.DataFrame:
        """
        Retrieves all high-form teams for the configured season and recent_period_for_form.
        """
        high_form_query = self.db_session.query(
                DBHighFormTeam.team_id.label("team_db_id"), # This is Team.id from DB
                DBHighFormTeam.win_rate,
                DBHighFormTeam.league_id.label("league_db_id") # League.id from DB
            ).\
            filter(DBHighFormTeam.season_year == self.season_year).\
            filter(DBHighFormTeam.recent_period_games == self.recent_period_for_form)
            # Ensure win_rate is also above a certain threshold if DataProcessor doesn't filter strictly
            # filter(DBHighFormTeam.win_rate >= MIN_INTERESTING_WIN_RATE) -- already handled by DataProcessor

        high_form_df = pd.read_sql(high_form_query.statement, self.db_session.bind)
        logger.info(f"Fetched {len(high_form_df)} high-form team entries from DB for season {self.season_year} and period {self.recent_period_for_form}.")
        return high_form_df

    def generate_predictions(self, days_ahead: Optional[int] = None, output_dir: Optional[str] = None):
        """
        Generates predictions by combining upcoming fixtures with high-form teams
        and saves the output to a CSV file. Uses class defaults if parameters not provided.
        """
        current_days_ahead = days_ahead if days_ahead is not None else self.config.days_ahead
        current_output_dir = output_dir if output_dir is not None else str(self.config.output_dir)

        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
            logger.info(f"Created output directory: {current_output_dir}")

        upcoming_fixtures_df = self.get_upcoming_fixtures(days_ahead=current_days_ahead)
        if upcoming_fixtures_df.empty:
            logger.info("No upcoming fixtures found. No predictions to generate.")
            return

        high_form_teams_df = self.get_high_form_teams_for_season()
        if high_form_teams_df.empty:
            logger.info("No high-form teams found for the season. Predictions will not show win rates.")
        
        # Merge fixtures with high-form teams for home team
        # We need to ensure the merge is league-specific if a team could be high-form in one league but not another (unlikely with current setup)
        # Current HighFormTeam table has league_id, so a merge on team_db_id AND league_db_id is more precise.
        
        # Rename columns for merging convenience if names clash or for clarity
        high_form_home = high_form_teams_df.rename(columns={'win_rate': 'home_win_rate', 'team_db_id': 'home_team_db_id', 'league_db_id': 'h_league_db_id'})
        
        # Merge for home team win rates
        predictions_df = pd.merge(
            upcoming_fixtures_df,
            high_form_home[['home_team_db_id', 'home_win_rate', 'h_league_db_id']], # Select relevant columns
            on=['home_team_db_id'], # Merge on team's DB ID
            how='left'
        )
        # Filter out merges where league_db_id of fixture doesn't match league_db_id of high form entry
        # This is important if a team_id could appear in multiple leagues' high_form_teams table (though unlikely for a single season)
        if not high_form_teams_df.empty:
             predictions_df = predictions_df[predictions_df['league_db_id'] == predictions_df['h_league_db_id']].drop(columns=['h_league_db_id'])


        # Merge for away team win rates
        high_form_away = high_form_teams_df.rename(columns={'win_rate': 'away_win_rate', 'team_db_id': 'away_team_db_id', 'league_db_id': 'a_league_db_id'})
        predictions_df = pd.merge(
            predictions_df,
            high_form_away[['away_team_db_id', 'away_win_rate', 'a_league_db_id']],
            on=['away_team_db_id'],
            how='left'
        )
        if not high_form_teams_df.empty:
            predictions_df = predictions_df[predictions_df['league_db_id'] == predictions_df['a_league_db_id']].drop(columns=['a_league_db_id'])

        # Filter for matches where at least one team has a win rate (is in high-form)
        flagged_matches_df = predictions_df[
            predictions_df['home_win_rate'].notnull() | predictions_df['away_win_rate'].notnull()
        ].copy()

        if flagged_matches_df.empty:
            logger.info("No matches found where at least one participating team is in high form.")
        else:
            logger.info(f"Found {len(flagged_matches_df)} matches with at least one high-form team.")

        # Select and order output columns
        output_columns = [
            'league_name', 'match_id', 'date', 'time',
            'home_team_name', 'home_win_rate',
            'away_team_name', 'away_win_rate'
        ]
        # Ensure all output columns exist, fill with NA if necessary before reordering
        for col in output_columns:
            if col not in flagged_matches_df.columns:
                flagged_matches_df[col] = pd.NA

        final_predictions_df = flagged_matches_df[output_columns].sort_values(by=['date', 'time'])

        # Format date for display if needed (it's already a date object from DB)
        final_predictions_df['date'] = pd.to_datetime(final_predictions_df['date']).dt.strftime('%Y-%m-%d')


        # Save to a timestamped CSV
        date_str = datetime.now().strftime("%Y%m%d")
        # Use current_output_dir which is derived from config or parameter
        output_filename = os.path.join(current_output_dir, f"predictions_{date_str}.csv")
        try:
            final_predictions_df.to_csv(output_filename, index=False)
            logger.info(f"Saved {len(final_predictions_df)} flagged matches to {output_filename}")
        except Exception as e:
            logger.error(f"Error saving predictions to {output_filename}: {e}")

from prefect import task, get_run_logger as prefect_get_run_logger

@task(name="Run Predictor Outputter")
def run_predictor_outputter_task(db_session_override: Optional[Session] = None):
    """Prefect task to run the PredictorOutputter."""
    task_logger = prefect_get_run_logger()
    if not app_config:
        task_logger.error("PredictorOutputter: Global app_config not loaded. Cannot proceed.")
        raise ValueError("Global app_config not loaded.")

    session_managed_locally = False
    if db_session_override:
        db_sess = db_session_override
    else:
        db_sess = SessionLocal()
        session_managed_locally = True
        task_logger.info("PredictorOutputter task created its own DB session.")

    try:
        predictor_config = app_config.predictor_outputter

        # Parameter consistency checks (optional, Pydantic might handle some)
        # These ensure that the predictor is using parameters consistent with how data was processed.
        if predictor_config.season_year != app_config.data_processor.season_year:
            task_logger.warning(
                f"Predictor season_year ({predictor_config.season_year}) "
                f"differs from DataProcessor's ({app_config.data_processor.season_year}). Using predictor's."
            )
        if predictor_config.recent_period_for_form != app_config.data_processor.recent_period:
             task_logger.warning(
                f"Predictor recent_period_for_form ({predictor_config.recent_period_for_form}) "
                f"differs from DataProcessor's ({app_config.data_processor.recent_period}). Using predictor's."
            )

        predictor = PredictorOutputter(
            db_session=db_sess,
            config=predictor_config
        )
        # generate_predictions will use days_ahead and output_dir from its own config by default
        predictor.generate_predictions()
        task_logger.info("PredictorOutputter task completed successfully.")
    except Exception as e:
        task_logger.error(f"PredictorOutputter: Critical error during execution: {e}", exc_info=True)
        raise
    finally:
        if session_managed_locally:
            db_sess.close()
            task_logger.info("PredictorOutputter task closed its locally managed DB session.")

if __name__ == "__main__":
    # Example of how to run the task directly (for testing)
    # if app_config:
    #     run_predictor_outputter_task() # This would create its own session
    # else:
    #     print("Cannot run standalone PredictorOutputter: app_config not loaded (config.yaml missing or invalid).")
    pass
