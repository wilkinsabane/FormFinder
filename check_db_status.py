#!/usr/bin/env python3
"""Check database status for feature processing."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_db_status():
    """Check database status."""
    try:
        # Load configuration
        load_config()
        
        with get_db_session() as session:
            # Check pre_computed_features count
            result = session.execute(text('SELECT COUNT(*) FROM pre_computed_features')).fetchone()
            print(f'Records in pre_computed_features: {result[0]}')
            
            # Check fixtures needing processing
            query = """
                SELECT COUNT(*) 
                FROM fixtures f 
                LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id 
                WHERE f.status = 'finished' 
                AND f.home_score IS NOT NULL 
                AND f.away_score IS NOT NULL 
                AND pcf.fixture_id IS NULL
            """
            result2 = session.execute(text(query)).fetchone()
            print(f'Fixtures needing processing: {result2[0]}')
            
            # Show sample fixtures needing processing
            query3 = """
                SELECT f.id, f.home_team_id, f.away_team_id, f.match_date 
                FROM fixtures f 
                LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id 
                WHERE f.status = 'finished' 
                AND f.home_score IS NOT NULL 
                AND f.away_score IS NOT NULL 
                AND pcf.fixture_id IS NULL
                ORDER BY f.match_date DESC
                LIMIT 5
            """
            result3 = session.execute(text(query3)).fetchall()
            print('\nSample fixtures needing processing:')
            for r in result3:
                print(f'  Fixture {r[0]}: Team {r[1]} vs Team {r[2]}, Date {r[3]}')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    check_db_status()