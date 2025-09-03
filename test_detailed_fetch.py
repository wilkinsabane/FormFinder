#!/usr/bin/env python3

from formfinder.config import load_config
from formfinder.database import get_db_session, Fixture, MatchEvent, MatchOdds, MatchLineup
from formfinder.DataFetcher import EnhancedDataFetcher, DataFetcherConfig
import asyncio

async def test_detailed_fetch():
    """Test detailed match data fetching for existing fixtures."""
    # Load configuration
    load_config('config.yaml')
    
    # Create data fetcher
    config = DataFetcherConfig.from_file('config.yaml')
    fetcher = EnhancedDataFetcher(config)
    
    with get_db_session() as session:
        # Get a fixture with an API fixture ID
        fixture = session.query(Fixture).filter(
            Fixture.api_fixture_id.isnot(None)
        ).first()
        
        if not fixture:
            print("No fixtures with API fixture ID found. Let's check all fixtures:")
            fixtures = session.query(Fixture).limit(5).all()
            for f in fixtures:
                print(f"Fixture {f.id}: API ID = {f.api_fixture_id}")
            return
        
        print(f"Testing detailed fetch for fixture {fixture.id} (API ID: {fixture.api_fixture_id})")
        
        try:
            # Fetch detailed match info
            detailed_data = await fetcher.fetch_detailed_match_info(fixture.api_fixture_id)
            
            if detailed_data:
                print("Detailed data fetched successfully:")
                print(f"- Events: {len(detailed_data.get('events', []))}")
                print(f"- Odds available: {'odds' in detailed_data}")
                print(f"- Lineups available: {'lineups' in detailed_data}")
                
                # Save detailed data
                await fetcher.save_detailed_match_data(fixture.api_fixture_id, detailed_data)
                print("Detailed data saved to database")
                
                # Check if data was actually saved
                events_count = session.query(MatchEvent).filter_by(fixture_id=fixture.id).count()
                odds_count = session.query(MatchOdds).filter_by(fixture_id=fixture.id).count()
                lineups_count = session.query(MatchLineup).filter_by(fixture_id=fixture.id).count()
                
                print(f"\nData saved to database:")
                print(f"- MatchEvent records: {events_count}")
                print(f"- MatchOdds records: {odds_count}")
                print(f"- MatchLineup records: {lineups_count}")
                
            else:
                print("No detailed data returned from API")
                
        except Exception as e:
            print(f"Error fetching detailed data: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_detailed_fetch())