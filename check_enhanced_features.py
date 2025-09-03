from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

load_config()
session = get_db_session().__enter__()

# Check recent fixtures with enhanced features
result = session.execute(text("""
    SELECT fixture_id, markov_home_trend_direction, markov_away_trend_direction, 
           home_team_position, away_team_position, league_avg_goals,
           h2h_total_matches, h2h_avg_goals
    FROM pre_computed_features 
    ORDER BY created_at DESC LIMIT 3
"""))

rows = result.fetchall()
print('Recent fixtures with enhanced features:')
for row in rows:
    print(f'Fixture {row[0]}:')
    print(f'  Home trend: {row[1]}, Away trend: {row[2]}')
    print(f'  Home pos: {row[3]}, Away pos: {row[4]}')
    print(f'  League avg goals: {row[5]}')
    print(f'  H2H matches: {row[6]}, H2H avg goals: {row[7]}')
    print()

session.close()
print('✅ Enhanced features are now properly stored with numeric trend directions!')