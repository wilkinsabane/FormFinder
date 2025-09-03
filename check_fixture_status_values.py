#!/usr/bin/env python3
"""
Check the actual status values in the fixtures table.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import DatabaseManager
from sqlalchemy import text
import logging

# Load configuration
load_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_status_values():
    """Check what status values exist in the fixtures table."""
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # Check distinct status values
        status_query = text("""
            SELECT status, COUNT(*) as count
            FROM fixtures
            GROUP BY status
            ORDER BY count DESC
        """)
        
        status_results = session.execute(status_query).fetchall()
        
        logger.info("=== Fixture Status Values ===")
        for status, count in status_results:
            logger.info(f"Status: '{status}' - Count: {count}")
        
        # Check fixtures with scores
        scores_query = text("""
            SELECT status, COUNT(*) as count
            FROM fixtures
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            GROUP BY status
            ORDER BY count DESC
        """)
        
        scores_results = session.execute(scores_query).fetchall()
        
        logger.info("\n=== Fixtures with Scores by Status ===")
        for status, count in scores_results:
            logger.info(f"Status: '{status}' - Count: {count}")
        
        # Sample some finished fixtures
        sample_query = text("""
            SELECT id, home_team_id, away_team_id, status, home_score, away_score, match_date
            FROM fixtures
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 5
        """)
        
        sample_results = session.execute(sample_query).fetchall()
        
        logger.info("\n=== Sample Fixtures with Scores ===")
        for fixture in sample_results:
            fixture_id, home_team, away_team, status, home_score, away_score, match_date = fixture
            logger.info(f"Fixture {fixture_id}: {home_team} vs {away_team} ({home_score}-{away_score}) - Status: '{status}' - Date: {match_date}")

if __name__ == "__main__":
    check_status_values()