from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
from datetime import datetime

# Load configuration
load_config()

with get_db_session() as session:
    # Test the exact query used in populate_markov_data.py
    fixtures_query = """
    SELECT id, home_team_id, away_team_id, match_date, league_id
    FROM fixtures 
    WHERE match_date BETWEEN :start_date AND :end_date
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
        AND league_id IN (228)
    ORDER BY match_date
    """
    
    result = session.execute(text(fixtures_query), {
        'start_date': '2020-09-01',
        'end_date': '2021-05-31'
    })
    fixtures = result.fetchall()
    
    print(f'Found {len(fixtures)} fixtures matching the query')
    
    if fixtures:
        print('\nFirst 5 fixtures:')
        for i, fixture in enumerate(fixtures[:5]):
            print(f'  {i+1}. ID: {fixture[0]}, Date: {fixture[3]}, League: {fixture[4]}')
    
    # Also check without date filter
    result2 = session.execute(text(
        "SELECT COUNT(*) FROM fixtures WHERE league_id = 228 AND home_score IS NOT NULL AND away_score IS NOT NULL"
    ))
    total_with_scores = result2.scalar()
    print(f'\nTotal fixtures with scores in league 228: {total_with_scores}')
    
    # Check date range of fixtures in league 228
    result3 = session.execute(text(
        "SELECT MIN(match_date), MAX(match_date) FROM fixtures WHERE league_id = 228"
    ))
    date_range = result3.fetchone()
    print(f'Date range for league 228: {date_range[0]} to {date_range[1]}')