#!/usr/bin/env python3
"""Check pre-computed features position data."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # First check the schema
        schema_result = session.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'pre_computed_features' ORDER BY ordinal_position"
        ))
        columns = [row[0] for row in schema_result.fetchall()]
        print('Available columns in pre_computed_features:')
        print(columns)
        print('\n' + '=' * 60)
        
        # Check position data
        result = session.execute(text(
            'SELECT fixture_id, home_team_position, away_team_position FROM pre_computed_features LIMIT 10'
        ))
        rows = result.fetchall()
        
        print('Pre-computed features position data:')
        for row in rows:
            print(f'Fixture {row[0]}: Home pos {row[1]}, Away pos {row[2]}')
        
        # Check if all positions are the same
        positions = [(row[1], row[2]) for row in rows]
        unique_positions = set(positions)
        print(f'\nUnique position combinations: {len(unique_positions)}')
        print(f'Position combinations: {unique_positions}')

if __name__ == '__main__':
    main()