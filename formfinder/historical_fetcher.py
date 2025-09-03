import asyncio
import argparse
import logging
import json
import sys
import os
import time
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
    """Main function to fetch historical match data and standings for specified leagues and season.
    
    Features:
    - Fetches historical match data for specified leagues from the API
    - Fetches standings data for leagues with matches
    - Supports loading league IDs from leagues.json file
    - Supports saving data to CSV files and/or database
    - Implements exponential backoff for API rate limiting
    - Configurable delay between API requests
    - Caching mechanism for league data
    - Comprehensive summary reporting
    - Option to skip leagues with existing data
    
    Command-line arguments:
    - leagues: League IDs to fetch (space-separated, optional if --from-json is used)
    - --season: Season year (default: 2023-2024)
    - --config: Config file path (default: sdata_config.json)
    - --save_csv: Save matches and standings to CSV
    - --save-to-db/--save_db: Save matches and standings to DB
    - --from-json: Load league IDs from leagues.json
    - --verbose: Enable verbose logging
    - --skip-existing: Skip leagues that already have data in the database
    - --delay: Delay between API requests in seconds (default: 1)
    - --max-retries: Maximum number of retries for API requests (default: 3)
    - --cache-ttl: Cache time-to-live in hours (default: 24)
    """
    try:
        logger.info('Starting historical fetcher script')
        # Initialize summary data structures
        summary_data = {
            'leagues_processed': 0,
            'leagues_skipped': 0,
            'features_computed': 0,
            'feature_computation_failures': 0,
            'total_matches': 0,
            'new_matches': 0,
            'total_standings': 0,
            'countries': set(),
            'teams': set(),
            'leagues': [],
            'season': None,
            'leagues_with_matches': defaultdict(int),
            'leagues_with_standings': defaultdict(int),
            'teams_by_country': defaultdict(set)
        }
        parser = argparse.ArgumentParser(description='Fetch historical match data for specified leagues and season.')
        parser.add_argument('leagues', nargs='*', type=int, help='League IDs to fetch (space-separated, optional if --from-json is used)')
        parser.add_argument('--leagues-file', type=str, help='File containing league IDs to fetch')
        parser.add_argument('--season', default='2024-2025', help='Season year (default: 2024-2025)')
        parser.add_argument('--config', default='sdata_config.json', help='Config file path (default: sdata_config.json)')
        parser.add_argument('--save_csv', action='store_true', help='Save to CSV')
        parser.add_argument('--save-to-db', '--save_db', action='store_true', help='Save to DB')
        parser.add_argument('--from-json', action='store_true', help='Load league IDs from leagues.json')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
        parser.add_argument('--skip-existing', action='store_true', help='Skip leagues that already have data in the database for the specified season')
        parser.add_argument('--delay', type=float, default=1.0, help='Delay between API requests in seconds (default: 1.0)')
        parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for API requests (default: 3)')
        parser.add_argument('--cache-ttl', type=int, default=24, help='Cache time-to-live in hours (default: 24)')
        parser.add_argument('--fetch-detailed', action='store_true', help='Fetch detailed match information including stadium, events, formations, and odds (significantly increases processing time)')
        args = parser.parse_args()

        if args.leagues_file:
            with open(args.leagues_file, 'r') as f:
                args.leagues.extend([int(league_id) for league_id in f.read().split()])

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
            parser.error('No leagues provided and no league file or --from-json not set')
        logger.info(f'Parsed arguments: leagues={args.leagues}, season={args.season}, config={args.config}, save_csv={args.save_csv}, save_to_db={args.save_to_db}')

        logger.info('Loading configuration')
        config = DataFetcherConfig.from_file(args.config)
        config.processing.season_year = args.season
        
        # Update config with command line parameters
        config.api.max_retries = args.max_retries
        config.processing.cache_ttl_hours = args.cache_ttl
        # Store delay for our custom retry logic
        request_delay = args.delay
        
        # Log configuration details if verbose
        if args.verbose:
            logger.debug(f"Configuration: delay={request_delay}s, max_retries={args.max_retries}, cache_ttl={args.cache_ttl}h, season={args.season}")
            
        logger.info(f'Configuration loaded and season set to {args.season}')
        load_config()  # Load global configuration
        
        # Set season in summary data
        summary_data['season'] = args.season

        logger.info('Initializing EnhancedDataFetcher')
        fetcher = EnhancedDataFetcher(config)
        logger.info('Fetcher initialized')

        # Fetch league names
        logger.info('Fetching all leagues')
        all_leagues = await fetcher.fetch_leagues()
        logger.info('Leagues fetched successfully')

        # Check for existing data if skip-existing is enabled
        existing_leagues = set()
        if args.skip_existing and args.save_to_db:
            logger.info('Checking for existing league data in the database')
            from formfinder.database import get_db_session, Fixture, League
            from sqlalchemy import func
            
            with get_db_session() as session:
                # Query leagues that have fixtures in the database for the current season
                existing_data = session.query(League.id, func.count(Fixture.id).label('match_count')).\
                    join(Fixture, League.id == Fixture.league_id).\
                    filter(League.season == args.season).\
                    group_by(League.id).all()
                
                for league_id, match_count in existing_data:
                    if match_count > 0:
                        existing_leagues.add(league_id)
                        logger.info(f'League ID {league_id} already has {match_count} matches in the database for season {args.season}')
        
        for league_id in args.leagues:
            try:
                # Skip if league already has data and skip-existing is enabled
                if args.skip_existing and league_id in existing_leagues:
                    logger.info(f'Skipping league ID {league_id} as it already has data in the database')
                    summary_data['leagues_skipped'] += 1
                    continue
                    
                league_name = all_leagues.get(league_id, f'League-{league_id}')
                logger.info(f'Processing league ID {league_id} ({league_name})')
                
                # Check if CSV file already exists
                if args.save_csv and args.skip_existing:
                    filename = f'historical_matches_{league_id}_{args.season}.csv'
                    if os.path.exists(filename):
                        logger.info(f'CSV file {filename} already exists, skipping fetch for this league')
                        summary_data['leagues_skipped'] += 1
                        continue

                logger.info(f'Fetching historical matches for {league_name}')
                try:
                    # Implement exponential backoff for API rate limiting
                    max_retries = args.max_retries
                    retry_count = 0
                    matches = []
                    
                    while retry_count <= max_retries:
                        try:
                            # Add configurable delay between API requests
                            if retry_count > 0:
                                backoff_delay = min(request_delay * (2 ** (retry_count - 1)), 60)  # Exponential backoff with max 60s
                                logger.info(f'Retry {retry_count}/{max_retries} for {league_name}, waiting {backoff_delay:.2f}s')
                                await asyncio.sleep(backoff_delay)
                            else:
                                # Regular delay between requests
                                await asyncio.sleep(request_delay)
                            
                            matches = await fetcher.fetch_historical_matches(league_id, league_name)
                            logger.info(f'Fetched {len(matches)} matches for {league_name}')
                            
                            # Fetch standings if we have matches
                            if matches:
                                logger.info(f'Fetching standings for {league_name}')
                                try:
                                    # Implement exponential backoff for API rate limiting
                                    standings_max_retries = args.max_retries
                                    standings_retry_count = 0
                                    standings = []
                                    
                                    while standings_retry_count <= standings_max_retries:
                                        try:
                                            # Add configurable delay between API requests
                                            if standings_retry_count > 0:
                                                backoff_delay = min(request_delay * (2 ** (standings_retry_count - 1)), 60)  # Exponential backoff with max 60s
                                                logger.info(f'Retry {standings_retry_count}/{standings_max_retries} for standings of {league_name}, waiting {backoff_delay:.2f}s')
                                                await asyncio.sleep(backoff_delay)
                                            else:
                                                # Regular delay between requests
                                                await asyncio.sleep(request_delay)
                                            
                                            standings = await fetcher.fetch_standings_async(league_id, league_name)
                                            logger.info(f'Fetched {len(standings)} standings entries for {league_name}')
                                            
                                            # Update summary data for standings
                                            if standings:
                                                summary_data['total_standings'] += len(standings)
                                                summary_data['leagues_with_standings'][league_name] = len(standings)
                                            
                                            # Save standings to CSV if requested
                                            if args.save_csv and standings:
                                                standings_filename = f'standings_{league_id}_{args.season}.csv'
                                                logger.info(f'Saving standings to CSV: {standings_filename}')
                                                fetcher.save_standings_to_csv(standings, standings_filename)
                                                logger.info('Standings saved to CSV')
                                            
                                            # Save standings to database if requested
                                            if args.save_to_db and standings:
                                                logger.info(f'Saving standings to database for league {league_id}')
                                                await fetcher.fetch_standings_to_db(standings, league_id, args.season)
                                                logger.info('Standings saved to database')
                                                
                                            break  # Success, exit retry loop
                                            
                                        except Exception as e:
                                            standings_retry_count += 1
                                            if '429' in str(e) or 'Too Many Requests' in str(e):
                                                logger.warning(f'Rate limit exceeded for standings of {league_name}: {str(e)}')
                                                if standings_retry_count > standings_max_retries:
                                                    logger.error(f'Max retries exceeded for standings of {league_name}')
                                                    raise
                                            else:
                                                logger.error(f'Error fetching standings for {league_name}: {str(e)}')
                                                raise
                                except Exception as e:
                                    logger.error(f'Failed to fetch standings for {league_name}: {str(e)}')
                                    standings = []
                            
                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            retry_count += 1
                            if '429' in str(e) or 'Too Many Requests' in str(e):
                                logger.warning(f'Rate limit exceeded for {league_name}: {str(e)}')
                                if retry_count > max_retries:
                                    logger.error(f'Max retries exceeded for {league_name}')
                                    raise
                            else:
                                logger.error(f'Error fetching matches for {league_name}: {str(e)}')
                                raise
                except Exception as e:
                    logger.error(f'Failed to fetch matches for {league_name}: {str(e)}')
                    matches = []
                
                # Update summary data
                summary_data['leagues_processed'] += 1
                summary_data['total_matches'] += len(matches)
                
                # Get country information
                country = 'Unknown'
                if args.from_json and league_id in leagues_data:
                    country = leagues_data[league_id]['country']
                
                summary_data['countries'].add(country)
                summary_data['leagues'].append((league_id, league_name, country))
                summary_data['leagues_with_matches'][league_name] = len(matches)
                
                # Extract teams from matches
                for match in matches:
                    summary_data['teams'].add(match.home_team_name)
                    summary_data['teams'].add(match.away_team_name)
                    summary_data['teams_by_country'][country].add(match.home_team_name)
                    summary_data['teams_by_country'][country].add(match.away_team_name)
                
                if not matches:
                    logger.warning(f'No matches found for league {league_id} ({league_name})')
                    continue

                if args.save_csv:
                    filename = f'historical_matches_{league_id}_{args.season}.csv'
                    logger.info(f'Saving matches to CSV: {filename}')
                    fetcher.save_matches_to_csv(matches, filename)
                    logger.info('Matches saved to CSV')

                if args.save_to_db:
                    logger.info(f'Saving matches to database for league {league_id}')
                    # Check for duplicates in the database before saving
                    from formfinder.database import get_db_session, Fixture
                    
                    with get_db_session() as session:
                        # Get existing match dates for this league to avoid duplicates
                        existing_matches = set()
                        db_matches = session.query(Fixture.home_team_id, Fixture.away_team_id, Fixture.match_date).\
                            filter(Fixture.league_id == league_id).all()
                        
                        for home_id, away_id, match_date in db_matches:
                            match_key = f"{home_id}_{away_id}_{match_date.strftime('%Y-%m-%d')}"
                            existing_matches.add(match_key)
                        
                        # Filter out matches that already exist in the database
                        new_matches = []
                        for match in matches:
                            match_date = datetime.strptime(match.date, '%Y-%m-%d')
                            match_key = f"{match.home_team_id}_{match.away_team_id}_{match.date}"
                            
                            if match_key not in existing_matches:
                                new_matches.append(match)
                            else:
                                logger.debug(f'Skipping duplicate match: {match.home_team_name} vs {match.away_team_name} on {match.date}')
                        
                        if not new_matches:
                            logger.info(f'No new matches to save for league {league_id}')
                            continue
                            
                        logger.info(f'Saving {len(new_matches)} new matches to database (filtered {len(matches) - len(new_matches)} duplicates)')
                        await fetcher.fetch_matches_to_db(new_matches, league_id, is_historical=True)
                        logger.info('New matches saved to database')
                        
                        # Update summary with new matches count
                        summary_data['new_matches'] += len(new_matches)
                        
                        # Fetch detailed match information if requested
                        if args.fetch_detailed:
                            logger.info(f'Fetching detailed match information for {len(matches)} matches in {league_name}')
                            detailed_matches_processed = 0
                            detailed_matches_failed = 0
                            
                            for match in matches:
                                try:
                                    # Add delay between detailed match requests
                                    await asyncio.sleep(request_delay)
                                    
                                    logger.debug(f'Fetching detailed info for match {match.id}: {match.home_team_name} vs {match.away_team_name}')
                                    detailed_match_data = await fetcher.fetch_detailed_match_info(match.id)
                                    
                                    if detailed_match_data:
                                        # Save detailed match data to database
                                        await fetcher.save_detailed_match_data(detailed_match_data)
                                        detailed_matches_processed += 1
                                        logger.debug(f'Successfully saved detailed data for match {match.id}')
                                    else:
                                        logger.warning(f'No detailed data available for match {match.id}')
                                        detailed_matches_failed += 1
                                        
                                except Exception as e:
                                    logger.error(f'Error fetching detailed data for match {match.id}: {str(e)}')
                                    detailed_matches_failed += 1
                                    continue
                            
                            logger.info(f'Detailed match data processing complete for {league_name}: {detailed_matches_processed} successful, {detailed_matches_failed} failed')
                            
                            # Update summary data with detailed match processing stats
                            if 'detailed_matches_processed' not in summary_data:
                                summary_data['detailed_matches_processed'] = 0
                                summary_data['detailed_matches_failed'] = 0
                            summary_data['detailed_matches_processed'] += detailed_matches_processed
                            summary_data['detailed_matches_failed'] += detailed_matches_failed
                        else:
                            logger.info(f'Skipping detailed match information fetch (use --fetch-detailed to enable)')
            except Exception as e:
                logger.error(f'Error processing league {league_id}: {str(e)}')
                # Continue to next league

        # Feature Pre-computation Phase
        if args.save_to_db and summary_data['new_matches'] > 0:
            logger.info('='*80)
            logger.info('STARTING FEATURE PRE-COMPUTATION')
            logger.info('='*80)
            
            try:
                # Initialize feature precomputer
                from formfinder.database import get_db_session
                with get_db_session() as db_session:
                    feature_precomputer = FeaturePrecomputer(db_session)
                    
                    # Get all fixtures that need feature computation
                    from formfinder.database import Fixture
                    
                    # Get recently added fixtures for feature computation
                    recent_fixtures = db_session.query(Fixture).filter(
                        Fixture.league_id.in_([int(lid) for lid in args.leagues])
                    ).all()
                    
                    if recent_fixtures:
                        logger.info(f'Computing features for {len(recent_fixtures)} fixtures')
                        
                        # Compute features for all fixtures
                        computation_results = await feature_precomputer.compute_all_features(
                            recent_fixtures,
                            batch_size=config.get('feature_computation', {}).get('batch_size', 10),
                            max_concurrent=config.get('feature_computation', {}).get('max_concurrent', 3)
                        )
                        
                        # Update summary with feature computation results
                        summary_data['features_computed'] = computation_results.get('total_computed', 0)
                        summary_data['feature_computation_failures'] = computation_results.get('total_failed', 0)
                        
                        logger.info(f'Feature computation completed: {summary_data["features_computed"]} computed, {summary_data["feature_computation_failures"]} failed')
                    else:
                        logger.info('No fixtures found for feature computation')
                        
            except Exception as e:
                logger.error(f'Error during feature pre-computation: {str(e)}')
                summary_data['feature_computation_failures'] = summary_data.get('feature_computation_failures', 0) + 1
        else:
            logger.info('Skipping feature pre-computation (no new matches or database save disabled)')

        # Display summary information
        logger.info('='*80)
        logger.info('SUMMARY OF DATA FETCHED')
        logger.info('='*80)
        logger.info(f"Season: {summary_data['season']}")
        logger.info(f"Leagues processed: {summary_data['leagues_processed']}")
        logger.info(f"Leagues skipped: {summary_data['leagues_skipped']}")
        logger.info(f"Total matches fetched: {summary_data['total_matches']}")
        logger.info(f"New matches added to database: {summary_data['new_matches']}")
        logger.info(f"Total standings entries fetched: {summary_data['total_standings']}")
        logger.info(f"Features computed: {summary_data['features_computed']}")
        logger.info(f"Feature computation failures: {summary_data['feature_computation_failures']}")
        
        # Display detailed match processing statistics only if detailed fetching was enabled
        if args.fetch_detailed and 'detailed_matches_processed' in summary_data:
            logger.info(f"Detailed match data processed: {summary_data['detailed_matches_processed']}")
            logger.info(f"Detailed match data failed: {summary_data['detailed_matches_failed']}")
            total_detailed_attempts = summary_data['detailed_matches_processed'] + summary_data['detailed_matches_failed']
            if total_detailed_attempts > 0:
                success_rate = (summary_data['detailed_matches_processed'] / total_detailed_attempts) * 100
                logger.info(f"Detailed match data success rate: {success_rate:.1f}%")
        logger.info(f"Countries represented: {len(summary_data['countries'])}")
        logger.info(f"Total teams: {len(summary_data['teams'])}")
        
        # Display leagues and match counts
        if summary_data['leagues_with_matches']:
            logger.info('\nLeagues and match counts:')
            for league_name, match_count in sorted(summary_data['leagues_with_matches'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {league_name}: {match_count} matches")
        
        # Display leagues and standings counts
        if summary_data['leagues_with_standings']:
            logger.info('\nLeagues and standings counts:')
            for league_name, standings_count in sorted(summary_data['leagues_with_standings'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {league_name}: {standings_count} standings entries")
        
        # Display countries and team counts
        if summary_data['teams_by_country']:
            logger.info('\nCountries and team counts:')
            for country, teams in sorted(summary_data['teams_by_country'].items(), key=lambda x: len(x[1]), reverse=True):
                logger.info(f"  {country}: {len(teams)} teams")
                if args.verbose:
                    for team in sorted(teams):
                        logger.info(f"    - {team}")
        
        # Save summary to JSON file
        try:
            # Convert sets to lists for JSON serialization
            json_summary = {
                'timestamp': datetime.now().isoformat(),
                'script': 'historical_fetcher',
                'season': summary_data['season'],
                'leagues_processed': summary_data['leagues_processed'],
                'leagues_skipped': summary_data['leagues_skipped'],
                'total_matches': summary_data['total_matches'],
                'new_matches': summary_data['new_matches'],
                'total_standings': summary_data['total_standings'],
                'features_computed': summary_data['features_computed'],
                'feature_computation_failures': summary_data['feature_computation_failures'],
                'countries': list(summary_data['countries']),
                'teams_count': len(summary_data['teams']),
                'leagues_with_matches': dict(summary_data['leagues_with_matches']),
                'leagues_with_standings': dict(summary_data['leagues_with_standings']),
                'teams_by_country': {country: list(teams) for country, teams in summary_data['teams_by_country'].items()}
            }
            
            # Only include detailed match statistics if --fetch-detailed was used
            if args.fetch_detailed:
                json_summary.update({
                    'detailed_matches_processed': summary_data.get('detailed_matches_processed', 0),
                    'detailed_matches_failed': summary_data.get('detailed_matches_failed', 0)
                })
            
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save to historical_fetch_summary.json
            summary_file = os.path.join(logs_dir, 'historical_fetch_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(json_summary, f, indent=2)
            logger.info(f"Summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving summary to file: {e}")
                
        logger.info('='*80)
        logger.info('Script completed successfully')
    except Exception as e:
        logger.error(f'Fatal error in main: {str(e)}')
        raise

if __name__ == '__main__':
    asyncio.run(main())