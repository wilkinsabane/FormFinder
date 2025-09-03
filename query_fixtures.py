#!/usr/bin/env python3

from formfinder.database import get_db_session, Fixture
from formfinder.config import load_config
from sqlalchemy import func

def main():
    # Load configuration
    load_config('config.yaml')
    with get_db_session() as session:
        # Get fixtures with api_fixture_id
        result = session.query(
            Fixture.id, 
            Fixture.api_fixture_id, 
            Fixture.home_team_id, 
            Fixture.away_team_id
        ).filter(
            Fixture.api_fixture_id.isnot(None)
        ).limit(10).all()
        
        print('Fixtures with api_fixture_id:')
        for r in result:
            print(f'ID: {r[0]}, API ID: {r[1]}, Home: {r[2]}, Away: {r[3]}')
        
        # Count total fixtures with api_fixture_id
        total_count = session.query(func.count(Fixture.id)).filter(
            Fixture.api_fixture_id.isnot(None)
        ).scalar()
        
        print(f'\nTotal fixtures with api_fixture_id: {total_count}')

if __name__ == '__main__':
    main()