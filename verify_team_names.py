#!/usr/bin/env python3
"""
Script to verify that team names are correctly stored in the high_form_teams table.
"""

import os
import sys
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import init_database, get_db_session
from formfinder.config import load_config
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_team_names():
    """Verify team names in high_form_teams table."""
    try:
        with get_db_session() as session:
            # Query high_form_teams with team names
            result = session.execute(text("""
                SELECT hft.team_name, hft.team_id, hft.win_rate, hft.wins, hft.total_matches, l.name as league_name
                FROM high_form_teams hft
                JOIN leagues l ON hft.league_id = l.id
                ORDER BY hft.win_rate DESC
                LIMIT 10
            """))
            
            teams = result.fetchall()
            
            if teams:
                logger.info("High-form teams with team names:")
                logger.info("-" * 80)
                for team in teams:
                    logger.info(
                        f"Team: {team.team_name} (ID: {team.team_id}) | "
                        f"League: {team.league_name} | "
                        f"Win Rate: {team.win_rate:.1%} | "
                        f"Wins: {team.wins}/{team.total_matches}"
                    )
                logger.info(f"\nTotal teams in high_form_teams: {len(teams)}")
            else:
                logger.info("No teams found in high_form_teams table")
                
            return True
            
    except Exception as e:
        logger.error(f"Error verifying team names: {e}")
        return False


def main():
    """Main verification function."""
    logger.info("Starting verification of team names in high_form_teams...")
    
    # Load configuration
    load_config()
    
    # Initialize database
    init_database()
    
    # Verify team names
    success = verify_team_names()
    
    if success:
        logger.info("Verification completed successfully!")
    else:
        logger.error("Verification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()