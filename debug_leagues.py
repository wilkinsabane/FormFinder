#!/usr/bin/env python3
"""
Debug script to investigate available leagues and fixtures in the database.

This script will:
1. Query the database for available leagues with finished fixtures
2. Compare with leagues listed in free_leagues.txt
3. Show fixture counts for each league
4. Help identify why training data loading fails
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main debug function."""
    try:
        # Load configuration
        load_config()
        
        # Read free leagues
        free_leagues_path = Path("free_leagues.txt")
        if free_leagues_path.exists():
            with open(free_leagues_path, 'r') as f:
                free_leagues = [int(line.strip()) for line in f if line.strip()]
            print(f"Free leagues from file: {free_leagues}")
        else:
            print("free_leagues.txt not found")
            free_leagues = []
        
        # Query database for available leagues
        with get_db_session() as session:
            # Get all leagues
            leagues_query = text("""
                SELECT DISTINCT l.id, l.name, l.country, l.season
                FROM leagues l
                ORDER BY l.id
            """)
            leagues_result = session.execute(leagues_query).fetchall()
            
            print(f"\nTotal leagues in database: {len(leagues_result)}")
            for league in leagues_result[:10]:  # Show first 10
                print(f"  League {league.id}: {league.name} ({league.country}, {league.season})")
            if len(leagues_result) > 10:
                print(f"  ... and {len(leagues_result) - 10} more")
            
            # Get leagues with finished fixtures
            finished_fixtures_query = text("""
                SELECT DISTINCT f.league_id, l.name, l.country, l.season,
                       COUNT(*) as fixture_count
                FROM fixtures f
                JOIN leagues l ON f.league_id = l.id
                WHERE f.status = 'FINISHED' 
                  AND f.home_score IS NOT NULL 
                  AND f.away_score IS NOT NULL
                GROUP BY f.league_id, l.name, l.country, l.season
                ORDER BY fixture_count DESC
            """)
            finished_result = session.execute(finished_fixtures_query).fetchall()
            
            print(f"\nLeagues with finished fixtures: {len(finished_result)}")
            available_league_ids = []
            for league in finished_result:
                available_league_ids.append(league.league_id)
                print(f"  League {league.league_id}: {league.name} ({league.country}, {league.season}) - {league.fixture_count} fixtures")
            
            # Check overlap with free leagues
            if free_leagues:
                overlap = set(free_leagues) & set(available_league_ids)
                missing = set(free_leagues) - set(available_league_ids)
                
                print(f"\nOverlap analysis:")
                print(f"  Free leagues: {len(free_leagues)}")
                print(f"  Available leagues: {len(available_league_ids)}")
                print(f"  Overlapping leagues: {len(overlap)} - {list(overlap)}")
                print(f"  Missing leagues: {len(missing)} - {list(missing)}")
                
                if overlap:
                    print(f"\nFixture details for overlapping leagues:")
                    for league_id in list(overlap)[:5]:  # Show first 5
                        fixture_detail_query = text("""
                            SELECT COUNT(*) as total_fixtures,
                                   COUNT(CASE WHEN status = 'FINISHED' THEN 1 END) as finished_fixtures,
                                   MIN(match_date) as earliest_match,
                                   MAX(match_date) as latest_match
                            FROM fixtures
                            WHERE league_id = :league_id
                        """)
                        detail_result = session.execute(fixture_detail_query, {"league_id": league_id}).fetchone()
                        print(f"  League {league_id}: {detail_result.total_fixtures} total, {detail_result.finished_fixtures} finished")
                        print(f"    Date range: {detail_result.earliest_match} to {detail_result.latest_match}")
            
            # Check fixture table structure
            print(f"\nFixture table structure check:")
            structure_query = text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'fixtures'
                ORDER BY ordinal_position
            """)
            try:
                structure_result = session.execute(structure_query).fetchall()
                for col in structure_result:
                    print(f"  {col.column_name}: {col.data_type}")
            except Exception as e:
                print(f"  Could not get structure (SQLite?): {e}")
                # Try a simple query to see what columns exist
                sample_query = text("SELECT * FROM fixtures LIMIT 1")
                sample_result = session.execute(sample_query).fetchone()
                if sample_result:
                    print(f"  Sample columns: {list(sample_result._mapping.keys())}")
    
    except Exception as e:
        logger.error(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()