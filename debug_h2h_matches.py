#!/usr/bin/env python3
"""
Debug script to investigate H2H match computation issues.
This will test the H2H query logic and check if historical matches exist.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import DatabaseManager
from sqlalchemy import text
from datetime import datetime
import logging

# Load configuration
load_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_h2h_query():
    """Test H2H query logic with actual data."""
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # First, let's check what data we have in fixtures table
        fixture_stats_query = text("""
            SELECT 
                COUNT(*) as total_fixtures,
                COUNT(CASE WHEN status = 'FINISHED' THEN 1 END) as finished_fixtures,
                COUNT(CASE WHEN home_score IS NOT NULL AND away_score IS NOT NULL THEN 1 END) as fixtures_with_scores,
                COUNT(CASE WHEN status = 'FINISHED' AND home_score IS NOT NULL AND away_score IS NOT NULL THEN 1 END) as finished_with_scores
            FROM fixtures
        """)
        
        stats = session.execute(fixture_stats_query).fetchone()
        logger.info(f"\n=== Fixture Statistics ===")
        logger.info(f"Total fixtures: {stats[0]}")
        logger.info(f"Finished fixtures: {stats[1]}")
        logger.info(f"Fixtures with scores: {stats[2]}")
        logger.info(f"Finished fixtures with scores: {stats[3]}")
        
        # Get a sample fixture to test with (any fixture with scores)
        sample_fixture_query = text("""
            SELECT id, home_team_id, away_team_id, match_date, league_id, status, home_score, away_score
            FROM fixtures 
            WHERE home_score IS NOT NULL 
            AND away_score IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 5
        """)
        
        sample_fixtures = session.execute(sample_fixture_query).fetchall()
        
        logger.info(f"\nFound {len(sample_fixtures)} sample fixtures to test")
        
        for fixture in sample_fixtures:
            fixture_id, home_team_id, away_team_id, match_date, league_id, status, home_score, away_score = fixture
            
            logger.info(f"\n=== Testing Fixture {fixture_id} ===")
            logger.info(f"Home Team: {home_team_id}, Away Team: {away_team_id}")
            logger.info(f"Match Date: {match_date}, League: {league_id}")
            logger.info(f"Status: {status}, Score: {home_score}-{away_score}")
            
            # Test the H2H query
            h2h_query = text("""
                SELECT home_team_id, away_team_id, home_score, away_score, match_date
                FROM fixtures
                WHERE ((home_team_id = :home_team_id AND away_team_id = :away_team_id)
                       OR (home_team_id = :away_team_id AND away_team_id = :home_team_id))
                    AND match_date < :match_date
                    AND status = 'FINISHED'
                    AND home_score IS NOT NULL
                    AND away_score IS NOT NULL
                ORDER BY match_date DESC
                LIMIT 10
            """)
            
            h2h_results = session.execute(h2h_query, {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'match_date': match_date
            }).fetchall()
            
            logger.info(f"H2H matches found: {len(h2h_results)}")
            
            if h2h_results:
                logger.info("H2H match details:")
                for i, match in enumerate(h2h_results[:3]):  # Show first 3
                    h_team, a_team, h_score, a_score, m_date = match
                    logger.info(f"  {i+1}. {h_team} vs {a_team}: {h_score}-{a_score} on {m_date}")
            else:
                logger.info("No H2H matches found. Let's check why...")
                
                # Check if these teams have any matches at all
                any_matches_query = text("""
                    SELECT COUNT(*) as total_matches
                    FROM fixtures
                    WHERE (home_team_id = :home_team_id OR away_team_id = :home_team_id
                           OR home_team_id = :away_team_id OR away_team_id = :away_team_id)
                        AND status = 'FINISHED'
                        AND home_score IS NOT NULL
                        AND away_score IS NOT NULL
                """)
                
                total_matches = session.execute(any_matches_query, {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id
                }).fetchone()[0]
                
                logger.info(f"Total finished matches for these teams: {total_matches}")
                
                # Check matches before this date
                before_date_query = text("""
                    SELECT COUNT(*) as matches_before
                    FROM fixtures
                    WHERE (home_team_id = :home_team_id OR away_team_id = :home_team_id
                           OR home_team_id = :away_team_id OR away_team_id = :away_team_id)
                        AND match_date < :match_date
                        AND status = 'FINISHED'
                        AND home_score IS NOT NULL
                        AND away_score IS NOT NULL
                """)
                
                matches_before = session.execute(before_date_query, {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'match_date': match_date
                }).fetchone()[0]
                
                logger.info(f"Matches before {match_date}: {matches_before}")
                
                # Check direct H2H without date restriction
                h2h_all_query = text("""
                    SELECT COUNT(*) as h2h_all
                    FROM fixtures
                    WHERE ((home_team_id = :home_team_id AND away_team_id = :away_team_id)
                           OR (home_team_id = :away_team_id AND away_team_id = :home_team_id))
                        AND status = 'FINISHED'
                        AND home_score IS NOT NULL
                        AND away_score IS NOT NULL
                """)
                
                h2h_all = session.execute(h2h_all_query, {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id
                }).fetchone()[0]
                
                logger.info(f"Total H2H matches (all time): {h2h_all}")
            
            logger.info("-" * 50)

def check_team_names():
    """Check team names to understand the data better."""
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # Get some team names
        team_query = text("""
            SELECT DISTINCT t.id, t.name, COUNT(f.id) as match_count
            FROM teams t
            LEFT JOIN fixtures f ON (f.home_team_id = t.id OR f.away_team_id = t.id)
            WHERE f.status = 'FINISHED'
            GROUP BY t.id, t.name
            ORDER BY match_count DESC
            LIMIT 10
        """)
        
        teams = session.execute(team_query).fetchall()
        
        logger.info("\n=== Top Teams by Match Count ===")
        for team_id, team_name, match_count in teams:
            logger.info(f"Team {team_id}: {team_name} ({match_count} matches)")

if __name__ == "__main__":
    logger.info("Starting H2H debug analysis...")
    
    check_team_names()
    test_h2h_query()
    
    logger.info("\nH2H debug analysis complete.")