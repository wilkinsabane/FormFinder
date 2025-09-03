#!/usr/bin/env python3
"""Fix goal_difference column in standings table."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_goal_difference():
    """Fix goal_difference values in standings table."""
    load_config()
    
    with get_db_session() as session:
        try:
            # First, check how many records need fixing
            check_query = text("""
                SELECT COUNT(*) as total_mismatched
                FROM standings 
                WHERE goals_for IS NOT NULL 
                  AND goals_against IS NOT NULL 
                  AND goal_difference != (goals_for - goals_against)
            """)
            
            result = session.execute(check_query).fetchone()
            total_mismatched = result[0]
            
            logger.info(f"Found {total_mismatched} records with incorrect goal_difference")
            
            if total_mismatched == 0:
                logger.info("No records need fixing!")
                return
            
            # Update all records to have correct goal_difference
            update_query = text("""
                UPDATE standings 
                SET goal_difference = (goals_for - goals_against)
                WHERE goals_for IS NOT NULL 
                  AND goals_against IS NOT NULL 
                  AND goal_difference != (goals_for - goals_against)
            """)
            
            result = session.execute(update_query)
            updated_count = result.rowcount
            
            # Commit the changes
            session.commit()
            
            logger.info(f"Successfully updated {updated_count} records")
            
            # Verify the fix
            verify_query = text("""
                SELECT COUNT(*) as remaining_mismatched
                FROM standings 
                WHERE goals_for IS NOT NULL 
                  AND goals_against IS NOT NULL 
                  AND goal_difference != (goals_for - goals_against)
            """)
            
            result = session.execute(verify_query).fetchone()
            remaining_mismatched = result[0]
            
            logger.info(f"Verification: {remaining_mismatched} records still have incorrect goal_difference")
            
            if remaining_mismatched == 0:
                logger.info("✅ All goal_difference values have been fixed successfully!")
            else:
                logger.warning(f"⚠️  {remaining_mismatched} records still need fixing")
                
            # Show some sample corrected data
            sample_query = text("""
                SELECT team_id, goals_for, goals_against, goal_difference
                FROM standings 
                WHERE goals_for IS NOT NULL 
                  AND goals_against IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 10
            """)
            
            sample_results = session.execute(sample_query).fetchall()
            
            logger.info("Sample corrected data:")
            for team_id, gf, ga, gd in sample_results:
                calculated_gd = gf - ga
                status = "✅" if gd == calculated_gd else "❌"
                logger.info(f"  {status} Team {team_id}: GF={gf}, GA={ga}, GD={gd} (should be {calculated_gd})")
                
        except Exception as e:
            logger.error(f"Error fixing goal_difference: {e}")
            session.rollback()
            raise

if __name__ == '__main__':
    fix_goal_difference()