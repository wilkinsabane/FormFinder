#!/usr/bin/env python3

from formfinder.database import get_db_session, Fixture, MatchEvent, MatchOdds, MatchLineup
from formfinder.config import load_config

def check_database():
    # Load configuration first
    load_config('config.yaml')
    with get_db_session() as session:
        # Check fixtures
        fixtures = session.query(Fixture).limit(5).all()
        print(f'Found {len(fixtures)} fixtures in database')
        for f in fixtures:
            home_team = f.home_team.name if f.home_team else 'Unknown'
            away_team = f.away_team.name if f.away_team else 'Unknown'
            print(f'Fixture {f.id}: {home_team} vs {away_team} on {f.match_date}')
        
        # Check detailed match data tables
        events_count = session.query(MatchEvent).count()
        odds_count = session.query(MatchOdds).count()
        lineups_count = session.query(MatchLineup).count()
        
        print(f'\nDetailed match data:')
        print(f'MatchEvent records: {events_count}')
        print(f'MatchOdds records: {odds_count}')
        print(f'MatchLineup records: {lineups_count}')
        
        # Show sample data if available
        if events_count > 0:
            sample_event = session.query(MatchEvent).first()
            print(f'Sample event: {sample_event.event_type} at minute {sample_event.event_minute}')
        
        if odds_count > 0:
            sample_odds = session.query(MatchOdds).first()
            print(f'Sample odds: Home {sample_odds.home_win_odds}, Draw {sample_odds.draw_odds}, Away {sample_odds.away_win_odds}')
        
        if lineups_count > 0:
            sample_lineup = session.query(MatchLineup).first()
            print(f'Sample lineup: {sample_lineup.player_name} ({sample_lineup.team}) - {sample_lineup.lineup_type}')

if __name__ == '__main__':
    check_database()