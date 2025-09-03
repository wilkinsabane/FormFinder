import asyncio
import argparse
import logging
import json
import time
import sys
import os
from datetime import datetime
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from formfinder.DataFetcher import EnhancedDataFetcher, DataFetcherConfig
from formfinder.config import load_config
from formfinder.feature_precomputer import FeaturePrecomputer
from formfinder.clients.api_client import SoccerDataAPIClient
from formfinder.database import get_db_session

# Set up logging
logging.basicConfig(level=logging.DEBUG if '--verbose' in sys.argv else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """
    Main function to fetch upcoming fixtures for specified leagues.
    
    Features:
    - Fetches upcoming fixtures for specified leagues from the API
    - Supports loading league IDs from leagues.json file
    - Supports saving data to CSV files and/or database
    - Implements exponential backoff for API rate limiting
    - Configurable delay between API requests
    - Caching mechanism for league data
    - Comprehensive summary reporting
    - Robust error handling with retries
    
    Command-line arguments:
    - leagues: League IDs to fetch (space-separated, optional if --from-json is used)
    - --config: Config file path (default: sdata_config.json)
    - --save_csv: Save to CSV
    - --save-to-db/--save_db: Save to DB
    - --from-json: Load league IDs from leagues.json
    - --verbose: Enable verbose logging
    - --delay: Delay between API requests in seconds (default: 1)
    - --max-retries: Maximum number of retries for API requests (default: 3)
    - --cache-ttl: Cache time-to-live in hours (default: 24)
    """
    try:
        logger.info('Starting upcoming fixture fetcher script')
        parser = argparse.ArgumentParser(description='Fetch upcoming fixture data for specified leagues.')
        parser.add_argument('leagues', nargs='*', type=int, help='League IDs to fetch (space-separated, optional if --from-json is used)')
        parser.add_argument('--config', default='sdata_config.json', help='Config file path (default: sdata_config.json)')
        parser.add_argument('--save_csv', action='store_true', help='Save to CSV')
        parser.add_argument('--save_db', action='store_true', help='Save to DB')
        parser.add_argument('--from-json', action='store_true', help='Load league IDs from leagues.json')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
        parser.add_argument('--delay', type=float, default=1.0, help='Delay between API requests in seconds (default: 1.0)')
        parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for API requests (default: 3)')
        parser.add_argument('--cache-ttl', type=int, default=24, help='Cache time-to-live in hours (default: 24)')
        parser.add_argument('--fetch-detailed', action='store_true', help='Fetch detailed match information including stadium, events, formations, and odds (significantly increases processing time)')
        args = parser.parse_args()
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        # Load league data from leagues.json if requested
        leagues_data = {}
        if args.from_json:
            logger.info('Loading league data from leagues.json')
            with open('leagues.json', 'r') as f:
                data = json.load(f)
                # Load all league data for reference
                for league in data['results']:
                    leagues_data[league['id']] = {
                        'name': league['name'],
                        'country': league['country']['name'].title() if 'country' in league else 'Unknown'
                    }
                
                # Only use all leagues from JSON if no specific leagues were provided
                if not args.leagues:
                    args.leagues = [league['id'] for league in data['results']]
                    logger.info(f'Loaded {len(args.leagues)} league IDs from leagues.json')
                else:
                    # Filter to only include leagues specified in command line arguments
                    specified_leagues = []
                    for league_id in args.leagues:
                        if league_id in leagues_data:
                            specified_leagues.append(league_id)
                            logger.info(f"Will process league ID {league_id} ({leagues_data[league_id]['name']}) from {leagues_data[league_id]['country']}")
                        else:
                            logger.warning(f"League ID {league_id} not found in leagues.json")
                    
                    # Update args.leagues to only include the specified leagues
                    args.leagues = specified_leagues
                    logger.info(f'Using {len(args.leagues)} league IDs from command line arguments with league data from leagues.json')
        elif not args.leagues:
            parser.error('No leagues provided and --from-json not set')
        logger.info(f'Parsed arguments: leagues={args.leagues}, config={args.config}, save_csv={args.save_csv}, save_db={args.save_db}')

        logger.info('Loading configuration')
        config = DataFetcherConfig.from_file(args.config)
        # Update config with command line parameters
        config.api.max_retries = args.max_retries
        config.processing.cache_ttl_hours = args.cache_ttl
        # Store delay for our custom retry logic
        request_delay = args.delay
        
        # Log configuration details if verbose
        if args.verbose:
            logger.debug(f"Configuration: delay={request_delay}s, max_retries={args.max_retries}, cache_ttl={args.cache_ttl}h")
            
        logger.info('Configuration loaded')
        load_config()  # Load global configuration

        logger.info('Initializing EnhancedDataFetcher')
        fetcher = EnhancedDataFetcher(config)
        logger.info('Fetcher initialized')

        # Fetch league names
        logger.info('Fetching all leagues')
        all_leagues = await fetcher.fetch_leagues()
        logger.info('Leagues fetched successfully')

        # Initialize counters and data structures for summary
        total_fixtures = 0
        leagues_processed = 0
        leagues_skipped = 0
        league_fixture_counts = {}
        countries = {}
        teams_by_country = {}
        features_computed = 0
        feature_computation_failures = 0
        
        # Initialize detailed fixture processing counters
        total_detailed_fixtures_processed = 0
        total_detailed_fixtures_failed = 0
        total_detailed_fixtures_skipped = 0
        
        for league_id in args.leagues:
            try:
                # Use league data from leagues.json if available
                if args.from_json and league_id in leagues_data:
                    league_name = leagues_data[league_id]['name']
                    country_name = leagues_data[league_id]['country']
                else:
                    league_name = all_leagues.get(league_id, f'League-{league_id}')
                    country_name = 'Unknown'
                
                logger.info(f'Processing league ID {league_id} ({league_name})')

                # Fetch upcoming fixtures for league with enhanced error handling
                logger.info(f'Fetching upcoming fixtures for {league_name}')
                try:
                    # Implement exponential backoff for API rate limiting
                    max_retries = args.max_retries
                    retry_count = 0
                    fixtures = []
                    
                    while retry_count <= max_retries:
                        try:
                            # Add configurable delay between API requests
                            if retry_count > 0:
                                backoff_delay = min(args.delay * (2 ** (retry_count - 1)), 60)  # Exponential backoff with max 60s
                                logger.info(f'Retry {retry_count}/{max_retries} for {league_name}, waiting {backoff_delay:.2f}s')
                                await asyncio.sleep(backoff_delay)
                            else:
                                # Regular delay between requests
                                await asyncio.sleep(request_delay)
                            
                            fixtures = await fetcher.fetch_upcoming_fixtures_for_league(league_id, league_name)
                            fixture_count = len(fixtures)
                            logger.info(f'Fetched {fixture_count} fixtures for {league_name}')
                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            retry_count += 1
                            if '429' in str(e) or 'Too Many Requests' in str(e):
                                logger.warning(f'Rate limit exceeded for {league_name}: {str(e)}')
                                if retry_count > max_retries:
                                    logger.error(f'Max retries exceeded for {league_name}')
                                    raise
                            else:
                                logger.error(f'Error fetching fixtures for {league_name}: {str(e)}')
                                raise
                except Exception as e:
                    logger.error(f'Failed to fetch fixtures for {league_name}: {str(e)}')
                    fixtures = []
                    fixture_count = 0
                
                # Update counters and data structures for summary
                total_fixtures += fixture_count
                leagues_processed += 1
                league_fixture_counts[league_name] = fixture_count
                
                # Track countries and teams
                if country_name not in teams_by_country:
                    teams_by_country[country_name] = set()
                    countries[country_name] = 0
                
                # Extract teams from fixtures and add to country's team set
                for fixture in fixtures:
                    # Access Match object attributes directly
                    teams_by_country[country_name].add(fixture.home_team_name)
                    teams_by_country[country_name].add(fixture.away_team_name)
                
                # Update team count for this country
                countries[country_name] = len(teams_by_country[country_name])

                if args.save_csv:
                    filename = f'upcoming_fixtures_{league_id}.csv'
                    logger.info(f'Saving fixtures to CSV: {filename}')
                    fetcher.save_matches_to_csv(fixtures, filename)
                    logger.info('Fixtures saved to CSV')

                if args.save_db:
                    logger.info(f'Saving fixtures to database for league {league_id}')
                    await fetcher.fetch_matches_to_db(fixtures, league_id, is_historical=False)
                    logger.info('Fixtures saved to database')
                    
                    # Fetch detailed match information for upcoming fixtures if requested
                    if args.fetch_detailed:
                        logger.info(f'Fetching detailed match information for {len(fixtures)} upcoming fixtures in {league_name}')
                        detailed_fixtures_processed = 0
                        detailed_fixtures_failed = 0
                        detailed_fixtures_skipped = 0
                        
                        for fixture in fixtures:
                            try:
                                # Only fetch detailed data if the fixture has a valid match ID
                                if hasattr(fixture, 'id') and fixture.id:
                                    # Add delay between detailed match requests
                                    await asyncio.sleep(request_delay)
                                    
                                    logger.debug(f'Fetching detailed info for upcoming fixture {fixture.id}: {fixture.home_team_name} vs {fixture.away_team_name}')
                                    detailed_match_data = await fetcher.fetch_detailed_match_info(fixture.id)
                                    
                                    if detailed_match_data:
                                        # Save detailed match data to database
                                        await fetcher.save_detailed_match_data(detailed_match_data)
                                        detailed_fixtures_processed += 1
                                        logger.debug(f'Successfully saved detailed data for upcoming fixture {fixture.id}')
                                    else:
                                        logger.warning(f'No detailed data available for upcoming fixture {fixture.id}')
                                        detailed_fixtures_failed += 1
                                else:
                                    logger.debug(f'Skipping detailed fetch for fixture without ID: {fixture.home_team_name} vs {fixture.away_team_name}')
                                    detailed_fixtures_skipped += 1
                                    
                            except Exception as e:
                                logger.error(f'Error fetching detailed data for upcoming fixture {getattr(fixture, "id", "unknown")}: {str(e)}')
                                detailed_fixtures_failed += 1
                                continue
                        
                        logger.info(f'Detailed fixture data processing complete for {league_name}: {detailed_fixtures_processed} successful, {detailed_fixtures_failed} failed, {detailed_fixtures_skipped} skipped')
                        
                        # Add to running totals
                        total_detailed_fixtures_processed += detailed_fixtures_processed
                        total_detailed_fixtures_failed += detailed_fixtures_failed
                        total_detailed_fixtures_skipped += detailed_fixtures_skipped
                    else:
                        logger.info(f'Skipping detailed fixture information fetch (use --fetch-detailed to enable)')
            except Exception as e:
                logger.error(f'Error processing league {league_id}: {str(e)}')
                leagues_skipped += 1
                # Continue to next league

        # Feature Pre-computation Phase
        if args.save_db and total_fixtures > 0:
            logger.info('='*80)
            logger.info('STARTING FEATURE PRE-COMPUTATION')
            logger.info('='*80)
            
            try:
                # Initialize feature precomputer
                with get_db_session() as db_session:
                    feature_precomputer = FeaturePrecomputer(db_session)
                    
                    # Get all fixtures that need feature computation
                    from formfinder.database import Fixture
                    
                    # Get recently added upcoming fixtures for feature computation
                    upcoming_fixtures = db_session.query(Fixture).filter(
                        Fixture.league_id.in_(args.leagues),
                        Fixture.match_date >= datetime.now().date()
                    ).all()
                    
                    if upcoming_fixtures:
                        logger.info(f'Computing features for {len(upcoming_fixtures)} upcoming fixtures')
                        
                        # Compute features for all fixtures
                        computation_results = await feature_precomputer.compute_all_features(
                            upcoming_fixtures,
                            batch_size=config.get('feature_computation', {}).get('batch_size', 10),
                            max_concurrent=config.get('feature_computation', {}).get('max_concurrent', 3)
                        )
                        
                        # Update summary with feature computation results
                        features_computed = computation_results.get('total_computed', 0)
                        feature_computation_failures = computation_results.get('total_failed', 0)
                        
                        logger.info(f'Feature computation completed: {features_computed} computed, {feature_computation_failures} failed')
                    else:
                        logger.info('No upcoming fixtures found for feature computation')
                        
            except Exception as e:
                logger.error(f'Error during feature pre-computation: {str(e)}')
                feature_computation_failures += 1
        else:
            logger.info('Skipping feature pre-computation (no fixtures or database save disabled)')

        # Display summary
        logger.info('================================================================================')
        logger.info('SUMMARY OF DATA FETCHED')
        logger.info('================================================================================')
        logger.info(f'Leagues processed: {leagues_processed}')
        logger.info(f'Leagues skipped: {leagues_skipped}')
        logger.info(f'Total upcoming fixtures fetched: {total_fixtures}')
        logger.info(f'Features computed: {features_computed}')
        logger.info(f'Feature computation failures: {feature_computation_failures}')
        logger.info(f'Countries represented: {len(countries)}')
        logger.info(f'Total teams: {sum(countries.values())}')
        
        # Display detailed fixture processing statistics only if detailed fetching was enabled
        if args.fetch_detailed:
            total_detailed_attempts = total_detailed_fixtures_processed + total_detailed_fixtures_failed
            if total_detailed_attempts > 0:
                logger.info(f'Detailed fixture data processed: {total_detailed_fixtures_processed}')
                logger.info(f'Detailed fixture data failed: {total_detailed_fixtures_failed}')
                logger.info(f'Detailed fixture data skipped: {total_detailed_fixtures_skipped}')
                success_rate = (total_detailed_fixtures_processed / total_detailed_attempts) * 100
                logger.info(f'Detailed fixture data success rate: {success_rate:.1f}%')
            else:
                logger.info('No detailed fixture data processing attempted')
        
        # Display leagues and fixture counts
        if league_fixture_counts:
            logger.info('')
            logger.info('Leagues and fixture counts:')
            for league_name, count in sorted(league_fixture_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f'  {league_name}: {count} fixtures')
        
        # Display countries and team counts
        if countries:
            logger.info('')
            logger.info('Countries and team counts:')
            for country_name, team_count in sorted(countries.items()):
                logger.info(f'  {country_name}: {team_count} teams')
                for team in sorted(teams_by_country[country_name]):
                    logger.info(f'    - {team}')
        
        # Save summary to JSON file
        try:
            # Prepare summary data for JSON serialization
            json_summary = {
                'timestamp': datetime.now().isoformat(),
                'script': 'upcoming_fetcher',
                'leagues_processed': leagues_processed,
                'leagues_skipped': leagues_skipped,
                'total_fixtures': total_fixtures,
                'features_computed': features_computed,
                'feature_computation_failures': feature_computation_failures,
                'countries': list(countries.keys()),
                'total_teams': sum(countries.values()),
                'league_fixture_counts': league_fixture_counts,
                'teams_by_country': {country: list(teams) for country, teams in teams_by_country.items()}
            }
            
            # Only include detailed fixture statistics if --fetch-detailed was used
            if args.fetch_detailed:
                json_summary.update({
                    'detailed_fixtures_processed': total_detailed_fixtures_processed,
                    'detailed_fixtures_failed': total_detailed_fixtures_failed,
                    'detailed_fixtures_skipped': total_detailed_fixtures_skipped
                })
            
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save to upcoming_fetch_summary.json
            summary_file = os.path.join(logs_dir, 'upcoming_fetch_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(json_summary, f, indent=2)
            logger.info(f"Summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving summary to file: {e}")
        
        logger.info('================================================================================')
        logger.info('Script completed successfully')
    except Exception as e:
        logger.error(f'Fatal error in main: {str(e)}')
        raise

if __name__ == '__main__':
    asyncio.run(main())