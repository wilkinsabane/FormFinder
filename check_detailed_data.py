#!/usr/bin/env python3
"""Check detailed match data counts in database."""

from formfinder.config import load_config
from formfinder.database import get_db_session, MatchEvent, MatchOdds, MatchLineup

def main():
    """Check counts of detailed match data."""
    # Load configuration first
    load_config('config.yaml')
    with get_db_session() as session:
        events_count = session.query(MatchEvent).count()
        odds_count = session.query(MatchOdds).count()
        lineups_count = session.query(MatchLineup).count()
        
        print(f"Detailed Match Data Counts:")
        print(f"Events: {events_count}")
        print(f"Odds: {odds_count}")
        print(f"Lineups: {lineups_count}")
        
        # Show some sample data
        if events_count > 0:
            sample_event = session.query(MatchEvent).first()
            print(f"\nSample Event: {sample_event.event_type} at minute {sample_event.event_minute}")
            
        if odds_count > 0:
            sample_odds = session.query(MatchOdds).first()
            print(f"Sample Odds: Home {sample_odds.home_win_odds}, Draw {sample_odds.draw_odds}, Away {sample_odds.away_win_odds}")
            
        if lineups_count > 0:
            sample_lineup = session.query(MatchLineup).first()
            print(f"Sample Lineup: {sample_lineup.player_name} ({sample_lineup.position})")

if __name__ == "__main__":
    main()