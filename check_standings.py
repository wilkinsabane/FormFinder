#!/usr/bin/env python3
"""Check standings data in the database."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check distinct positions
        result = session.execute(text('SELECT DISTINCT position FROM standings ORDER BY position'))
        positions = [row[0] for row in result.fetchall()]
        print('Available positions in standings:', positions)
        print('Total unique positions:', len(positions))
        
        # Check position distribution
        result = session.execute(text('SELECT position, COUNT(*) as count FROM standings GROUP BY position ORDER BY position'))
        position_counts = result.fetchall()
        print('\nPosition distribution:')
        for pos, count in position_counts:
            print(f'  Position {pos}: {count} teams')
        
        # Check some sample data
        result = session.execute(text('SELECT team_id, position, updated_at FROM standings ORDER BY updated_at DESC LIMIT 10'))
        samples = result.fetchall()
        print('\nSample standings data:')
        for team_id, position, updated_at in samples:
            print(f'  Team {team_id}: Position {position}, Updated: {updated_at}')

if __name__ == '__main__':
    main()