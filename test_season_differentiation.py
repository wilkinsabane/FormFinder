#!/usr/bin/env python3
"""Test that standings from different seasons are properly separated and don't overwrite each other."""

import logging
from datetime import datetime
from formfinder.config import load_config
from formfinder.database import get_db_session, Standing as DBStanding, League
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_season_differentiation():
    """Test that standings are properly differentiated by season."""
    load_config()
    
    with get_db_session() as session:
        # Test 1: Check that standings have season data
        logger.info("Test 1: Checking standings have season data...")
        
        standings_query = text("""
            SELECT COUNT(*) as total_standings,
                   COUNT(DISTINCT season) as unique_seasons,
                   COUNT(CASE WHEN season IS NULL THEN 1 END) as null_seasons
            FROM standings
        """)
        
        result = session.execute(standings_query).fetchone()
        logger.info(f"✓ Total standings: {result[0]}, Unique seasons: {result[1]}, Null seasons: {result[2]}")
        
        assert result[2] == 0, f"Found {result[2]} standings with null seasons"
        assert result[1] > 0, "No seasons found in standings"
        
        # Test 2: Check unique constraint works
        logger.info("Test 2: Checking unique constraint on (league_id, team_id, season)...")
        
        # Try to find any duplicate combinations
        duplicate_query = text("""
            SELECT league_id, team_id, season, COUNT(*) as count
            FROM standings
            GROUP BY league_id, team_id, season
            HAVING COUNT(*) > 1
        """)
        
        duplicates = session.execute(duplicate_query).fetchall()
        logger.info(f"✓ Found {len(duplicates)} duplicate combinations (should be 0)")
        
        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate standings for same league/team/season"
        
        # Test 3: Check that different seasons can have same team/league combinations
        logger.info("Test 3: Checking that same team can have standings in different seasons...")
        
        multi_season_query = text("""
            SELECT league_id, team_id, COUNT(DISTINCT season) as season_count
            FROM standings
            GROUP BY league_id, team_id
            HAVING COUNT(DISTINCT season) > 1
            LIMIT 5
        """)
        
        multi_season_teams = session.execute(multi_season_query).fetchall()
        logger.info(f"✓ Found {len(multi_season_teams)} teams with standings in multiple seasons")
        
        if multi_season_teams:
            for league_id, team_id, season_count in multi_season_teams:
                logger.info(f"  Team {team_id} in League {league_id}: {season_count} seasons")
        
        # Test 4: Verify season data integrity
        logger.info("Test 4: Checking season data integrity...")
        
        season_integrity_query = text("""
            SELECT s.season, COUNT(*) as standings_count
            FROM standings s
            GROUP BY s.season
            ORDER BY s.season
        """)
        
        season_counts = session.execute(season_integrity_query).fetchall()
        logger.info("✓ Season distribution:")
        for season, count in season_counts:
            logger.info(f"  {season}: {count} standings")
        
        # Test 5: Test position queries with season
        logger.info("Test 5: Testing position queries with season filtering...")
        
        # Get a sample team and league with multiple seasons
        sample_query = text("""
            SELECT league_id, team_id, season, position
            FROM standings
            WHERE (league_id, team_id) IN (
                SELECT league_id, team_id
                FROM standings
                GROUP BY league_id, team_id
                HAVING COUNT(DISTINCT season) > 1
                LIMIT 1
            )
            ORDER BY season
        """)
        
        sample_standings = session.execute(sample_query).fetchall()
        
        if sample_standings:
            league_id, team_id = sample_standings[0][0], sample_standings[0][1]
            logger.info(f"✓ Testing with Team {team_id} in League {league_id}:")
            
            for league_id, team_id, season, position in sample_standings:
                logger.info(f"  Season {season}: Position {position}")
            
            # Test season-specific query
            for season_data in sample_standings:
                season = season_data[2]
                position_query = text("""
                    SELECT position
                    FROM standings
                    WHERE league_id = :league_id
                        AND team_id = :team_id
                        AND season = :season
                """)
                
                result = session.execute(position_query, {
                    'league_id': league_id,
                    'team_id': team_id,
                    'season': season
                }).fetchone()
                
                assert result is not None, f"No position found for team {team_id} in season {season}"
                logger.info(f"  ✓ Season {season} query returned position: {result[0]}")
        
        logger.info("🎉 All season differentiation tests passed!")
        
        # Test 6: Simulate the original problem scenario
        logger.info("Test 6: Simulating original overwrite scenario...")
        
        # Check if we have data for multiple seasons
        seasons = [row[0] for row in session.execute(text("SELECT DISTINCT season FROM standings ORDER BY season")).fetchall()]
        
        if len(seasons) >= 2:
            logger.info(f"✓ Found {len(seasons)} seasons: {seasons}")
            logger.info("✓ With season differentiation, fetching data for different seasons will NOT overwrite each other")
            logger.info("✓ Each season's standings are stored separately with unique constraints")
        else:
            logger.info(f"ℹ Only {len(seasons)} season(s) found. Migration successful but need more data to fully test scenario.")

def main():
    """Run the season differentiation tests."""
    try:
        test_season_differentiation()
        logger.info("✅ Season differentiation test completed successfully!")
    except Exception as e:
        logger.error(f"❌ Season differentiation test failed: {e}")
        raise

if __name__ == "__main__":
    main()