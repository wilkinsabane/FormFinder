#!/usr/bin/env python3
"""Check date range of fixtures."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_date_range():
    """Check date range of fixtures."""
    try:
        # Load configuration
        load_config()
        
        with get_db_session() as session:
            # Check overall date range
            result = session.execute(text(
                'SELECT MIN(match_date), MAX(match_date), COUNT(*) FROM fixtures WHERE home_score IS NOT NULL'
            )).fetchone()
            print(f'Date range: {result[0]} to {result[1]}, Total fixtures: {result[2]}')
            
            # Check fixtures before 2020-09-14
            result2 = session.execute(text(
                "SELECT COUNT(*) FROM fixtures WHERE match_date < '2020-09-14' AND home_score IS NOT NULL"
            )).fetchone()
            print(f'Fixtures before 2020-09-14: {result2[0]}')
            
            # Check specific fixture details
            result3 = session.execute(text(
                "SELECT id, league_id, home_team_id, away_team_id, match_date FROM fixtures WHERE id = 19440"
            )).fetchone()
            if result3:
                print(f'\nFixture 19440 details: League {result3[1]}, Teams {result3[2]} vs {result3[3]}, Date {result3[4]}')
                
                # Check historical data for this league and teams
                result4 = session.execute(text(
                    "SELECT COUNT(*) FROM fixtures WHERE league_id = :league_id AND match_date < :match_date AND home_score IS NOT NULL"
                ), {'league_id': result3[1], 'match_date': result3[4]}).fetchone()
                print(f'Historical fixtures in league {result3[1]} before {result3[4]}: {result4[0]}')
                
                # Check team history
                result5 = session.execute(text(
                    "SELECT COUNT(*) FROM fixtures WHERE (home_team_id = :team_id OR away_team_id = :team_id) AND match_date < :match_date AND home_score IS NOT NULL"
                ), {'team_id': result3[2], 'match_date': result3[4]}).fetchone()
                print(f'Historical fixtures for team {result3[2]} before {result3[4]}: {result5[0]}')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    check_date_range()