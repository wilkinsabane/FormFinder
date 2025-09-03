from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
import pandas as pd

# Load configuration
load_config()

with get_db_session() as session:
    # Check fixture 20816
    result = pd.read_sql_query(text('SELECT id, home_team_id, away_team_id, match_date, league_id FROM fixtures WHERE id = 20816'), session.bind)
    print('Fixture 20816:', result.to_dict('records') if not result.empty else 'Not found')
    
    if not result.empty:
        fixture = result.iloc[0]
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        league_id = fixture['league_id']
        match_date = fixture['match_date']
        
        print(f"Home team ID: {home_team_id}, Away team ID: {away_team_id}")
        print(f"League ID: {league_id}, Match date: {match_date}")
        
        # Check if teams have any previous matches
        home_matches = pd.read_sql_query(text('''
            SELECT COUNT(*) as count FROM fixtures 
            WHERE (home_team_id = :team_id OR away_team_id = :team_id) 
            AND home_score IS NOT NULL 
            AND match_date < :match_date
            AND league_id = :league_id
        '''), session.bind, params={'team_id': int(home_team_id), 'match_date': match_date, 'league_id': int(league_id)})
        
        away_matches = pd.read_sql_query(text('''
            SELECT COUNT(*) as count FROM fixtures 
            WHERE (home_team_id = :team_id OR away_team_id = :team_id) 
            AND home_score IS NOT NULL 
            AND match_date < :match_date
            AND league_id = :league_id
        '''), session.bind, params={'team_id': int(away_team_id), 'match_date': match_date, 'league_id': int(league_id)})
        
        print(f"Home team previous matches: {home_matches.iloc[0]['count']}")
        print(f"Away team previous matches: {away_matches.iloc[0]['count']}")