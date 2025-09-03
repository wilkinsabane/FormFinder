#!/usr/bin/env python3
"""Check H2H data computation and sources."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check fixtures table for actual match results
        result = session.execute(text('''
            SELECT COUNT(*) FROM fixtures 
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        '''))
        completed_fixtures = result.fetchone()[0]
        print(f'Completed fixtures with scores: {completed_fixtures}')
        
        # Check a few specific fixtures to see their H2H computation
        result2 = session.execute(text('''
            SELECT f.id, f.home_team_id, f.away_team_id, f.home_score, f.away_score,
                   ht.name as home_team, at.name as away_team
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.home_score IS NOT NULL AND f.away_score IS NOT NULL
            ORDER BY f.match_date DESC
            LIMIT 5
        '''))
        fixtures = result2.fetchall()
        print('\nSample completed fixtures:')
        for fixture in fixtures:
            print(f'Fixture {fixture[0]}: {fixture[5]} vs {fixture[6]} ({fixture[3]}-{fixture[4]})')
            
            # Check if these teams have played before
            h2h_result = session.execute(text('''
                SELECT COUNT(*) FROM fixtures
                WHERE ((home_team_id = :home_id AND away_team_id = :away_id) OR
                       (home_team_id = :away_id AND away_team_id = :home_id))
                AND home_score IS NOT NULL AND away_score IS NOT NULL
                AND id != :fixture_id
            '''), {
                'home_id': fixture[1],
                'away_id': fixture[2], 
                'fixture_id': fixture[0]
            })
            h2h_count = h2h_result.fetchone()[0]
            print(f'  H2H matches available: {h2h_count}')
        
        # Check pre_computed_features H2H values for these fixtures
        print('\nH2H values in pre_computed_features:')
        for fixture in fixtures:
            pcf_result = session.execute(text('''
                SELECT h2h_total_matches, h2h_avg_goals, h2h_home_wins, h2h_away_wins
                FROM pre_computed_features
                WHERE fixture_id = :fixture_id
            '''), {'fixture_id': fixture[0]})
            pcf_row = pcf_result.fetchone()
            if pcf_row:
                print(f'  Fixture {fixture[0]}: matches={pcf_row[0]}, avg_goals={pcf_row[1]}, home_wins={pcf_row[2]}, away_wins={pcf_row[3]}')
            else:
                print(f'  Fixture {fixture[0]}: No pre-computed features found')

if __name__ == '__main__':
    main()