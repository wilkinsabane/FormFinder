from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

load_config()

with get_db_session() as session:
    # Check state stability, transition entropy, and performance volatility features
    result = session.execute(text('''
        SELECT fixture_id, 
               home_team_state_stability, away_team_state_stability,
               home_team_transition_entropy, away_team_transition_entropy,
               home_team_performance_volatility, away_team_performance_volatility
        FROM pre_computed_features 
        ORDER BY id DESC LIMIT 10
    '''))
    rows = result.fetchall()
    
    print('Team state/entropy/volatility features:')
    for row in rows:
        print(f'Fixture {row[0]}:')
        print(f'  H_stability={row[1]}, A_stability={row[2]}')
        print(f'  H_entropy={row[3]}, A_entropy={row[4]}')
        print(f'  H_volatility={row[5]}, A_volatility={row[6]}')
        print()
    
    # Check if all values are identical
    print('Checking for identical values:')
    for row in rows:
        values = [row[1], row[2], row[3], row[4], row[5], row[6]]
        unique_values = set(values)
        if len(unique_values) == 1:
            print(f'Fixture {row[0]}: ALL VALUES IDENTICAL = {values[0]}')
        else:
            print(f'Fixture {row[0]}: Values vary = {unique_values}')