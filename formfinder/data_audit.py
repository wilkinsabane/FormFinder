"""Script to perform data audit for FormFinder database."""

import logging
from sqlalchemy import func
from .database import get_db_session, League, Team, Standing, Fixture, Prediction, DataFetchLog

logger = logging.getLogger(__name__)

def audit_database():
    """Perform audit of database contents and log summaries."""
    with get_db_session() as session:
        # Count leagues
        league_count = session.query(func.count(League.id)).scalar()
        logger.info(f"Number of leagues: {league_count}")

        # Count teams
        team_count = session.query(func.count(Team.id)).scalar()
        logger.info(f"Number of teams: {team_count}")

        # Count standings
        standing_count = session.query(func.count(Standing.id)).scalar()
        logger.info(f"Number of standings entries: {standing_count}")

        # Count fixtures
        fixture_count = session.query(func.count(Fixture.id)).scalar()
        logger.info(f"Number of fixtures: {fixture_count}")

        # Historical depth (earliest and latest fixture dates)
        earliest_fixture = session.query(func.min(Fixture.match_date)).scalar()
        latest_fixture = session.query(func.max(Fixture.match_date)).scalar()
        logger.info(f"Fixture date range: {earliest_fixture} to {latest_fixture}")

        # Data quality: Check for missing scores in finished fixtures
        finished_fixtures = session.query(Fixture).filter(Fixture.status == 'finished').count()
        missing_scores = session.query(Fixture).filter(
            Fixture.status == 'finished',
            (Fixture.home_score.is_(None)) | (Fixture.away_score.is_(None))
        ).count()
        logger.info(f"Finished fixtures with missing scores: {missing_scores} out of {finished_fixtures}")

        # Predictions count
        prediction_count = session.query(func.count(Prediction.id)).scalar()
        logger.info(f"Number of predictions: {prediction_count}")

        # Fetch logs
        fetch_log_count = session.query(func.count(DataFetchLog.id)).scalar()
        logger.info(f"Number of data fetch logs: {fetch_log_count}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    audit_database()