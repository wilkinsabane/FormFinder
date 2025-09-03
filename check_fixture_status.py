#!/usr/bin/env python3
"""
Check fixture status values in the database to understand why no 'FINISHED' fixtures are found.
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
    """Check fixture status values."""
    try:
        # Load configuration
        load_config()
        
        with get_db_session() as session:
            # Check distinct status values
            status_query = text("""
                SELECT status, COUNT(*) as count
                FROM fixtures
                GROUP BY status
                ORDER BY count DESC
            """)
            status_result = session.execute(status_query).fetchall()
            
            print("Fixture status distribution:")
            for row in status_result:
                print(f"  {row.status}: {row.count} fixtures")
            
            # Check fixtures with scores
            scores_query = text("""
                SELECT status, 
                       COUNT(*) as total,
                       COUNT(CASE WHEN home_score IS NOT NULL AND away_score IS NOT NULL THEN 1 END) as with_scores
                FROM fixtures
                GROUP BY status
                ORDER BY total DESC
            """)
            scores_result = session.execute(scores_query).fetchall()
            
            print("\nFixtures with scores by status:")
            for row in scores_result:
                print(f"  {row.status}: {row.with_scores}/{row.total} have scores")
            
            # Sample some fixtures to see their data
            sample_query = text("""
                SELECT id, league_id, status, match_date, home_score, away_score
                FROM fixtures
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
                LIMIT 10
            """)
            sample_result = session.execute(sample_query).fetchall()
            
            print("\nSample fixtures with scores:")
            for row in sample_result:
                print(f"  ID {row.id}: League {row.league_id}, Status '{row.status}', Date {row.match_date}, Score {row.home_score}-{row.away_score}")
            
            # Check if any fixtures exist for free leagues
            free_leagues_path = Path("free_leagues.txt")
            if free_leagues_path.exists():
                with open(free_leagues_path, 'r') as f:
                    free_leagues = [int(line.strip()) for line in f if line.strip()]
                
                if free_leagues:
                    # Check first few free leagues
                    test_leagues = free_leagues[:5]
                    for league_id in test_leagues:
                        league_query = text("""
                            SELECT COUNT(*) as total_fixtures,
                                   COUNT(CASE WHEN home_score IS NOT NULL AND away_score IS NOT NULL THEN 1 END) as scored_fixtures,
                                   status
                            FROM fixtures
                            WHERE league_id = :league_id
                            GROUP BY status
                        """)
                        league_result = session.execute(league_query, {"league_id": league_id}).fetchall()
                        
                        print(f"\nLeague {league_id} fixtures:")
                        if league_result:
                            for row in league_result:
                                print(f"  Status '{row.status}': {row.scored_fixtures}/{row.total_fixtures} with scores")
                        else:
                            print(f"  No fixtures found for league {league_id}")
    
    except Exception as e:
        logger.error(f"Error during check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()