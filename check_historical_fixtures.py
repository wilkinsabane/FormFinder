#!/usr/bin/env python3
"""
Check for historical fixtures and test H2H with older data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import DatabaseManager
from sqlalchemy import text
import logging
from datetime import datetime, timedelta

# Load configuration
load_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_historical_fixtures():
    """Check for historical fixtures and test H2H computation."""
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # Check date range of fixtures
        date_range_query = text("""
            SELECT 
                MIN(match_date) as earliest,
                MAX(match_date) as latest,
                COUNT(*) as total
            FROM fixtures
            WHERE status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
        """)
        
        date_result = session.execute(date_range_query).fetchone()
        logger.info(f"=== Fixture Date Range ===")
        logger.info(f"Earliest: {date_result[0]}")
        logger.info(f"Latest: {date_result[1]}")
        logger.info(f"Total finished with scores: {date_result[2]}")
        
        # Find teams that have played multiple times
        teams_query = text("""
            SELECT 
                LEAST(home_team_id, away_team_id) as team1,
                GREATEST(home_team_id, away_team_id) as team2,
                COUNT(*) as match_count,
                MIN(match_date) as first_match,
                MAX(match_date) as last_match
            FROM fixtures
            WHERE status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
            GROUP BY LEAST(home_team_id, away_team_id), GREATEST(home_team_id, away_team_id)
            HAVING COUNT(*) > 1
            ORDER BY match_count DESC
            LIMIT 10
        """)
        
        teams_results = session.execute(teams_query).fetchall()
        
        logger.info(f"\n=== Teams with Multiple Matches ===")
        for team1, team2, count, first, last in teams_results:
            logger.info(f"Teams {team1} vs {team2}: {count} matches from {first} to {last}")
        
        if teams_results:
            # Test H2H with the team pair that has the most matches
            team1, team2, match_count, first_match, last_match = teams_results[0]
            
            logger.info(f"\n=== Testing H2H for Teams {team1} vs {team2} ===")
            
            # Get all matches between these teams
            h2h_query = text("""
                SELECT id, home_team_id, away_team_id, home_score, away_score, match_date
                FROM fixtures
                WHERE ((home_team_id = :team1 AND away_team_id = :team2)
                       OR (home_team_id = :team2 AND away_team_id = :team1))
                    AND status = 'finished'
                    AND home_score IS NOT NULL
                    AND away_score IS NOT NULL
                ORDER BY match_date ASC
            """)
            
            h2h_results = session.execute(h2h_query, {
                'team1': team1,
                'team2': team2
            }).fetchall()
            
            logger.info(f"Found {len(h2h_results)} H2H matches:")
            for fixture_id, home, away, home_score, away_score, match_date in h2h_results:
                logger.info(f"  Fixture {fixture_id}: {home} vs {away} ({home_score}-{away_score}) on {match_date}")
            
            # Test H2H query for the last match
            if len(h2h_results) > 1:
                last_fixture = h2h_results[-1]
                test_date = last_fixture[5]  # match_date
                
                logger.info(f"\n=== Testing H2H Query for Date {test_date} ===")
                
                # This is the same query used in enhanced_feature_computer.py
                test_h2h_query = text("""
                    SELECT home_team_id, away_team_id, home_score, away_score, match_date
                    FROM fixtures
                    WHERE ((home_team_id = :home_team_id AND away_team_id = :away_team_id)
                           OR (home_team_id = :away_team_id AND away_team_id = :home_team_id))
                        AND match_date < :match_date
                        AND status = 'finished'
                        AND home_score IS NOT NULL
                        AND away_score IS NOT NULL
                    ORDER BY match_date DESC
                    LIMIT 10
                """)
                
                test_results = session.execute(test_h2h_query, {
                    'home_team_id': last_fixture[1],  # home_team_id
                    'away_team_id': last_fixture[2],  # away_team_id
                    'match_date': test_date
                }).fetchall()
                
                logger.info(f"H2H query found {len(test_results)} matches before {test_date}:")
                for home, away, home_score, away_score, match_date in test_results:
                    logger.info(f"  {home} vs {away} ({home_score}-{away_score}) on {match_date}")
        else:
            logger.info("No teams found with multiple matches")

if __name__ == "__main__":
    check_historical_fixtures()