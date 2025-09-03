import sys
import os
import json
import logging
from collections import defaultdict

# Set up logging to both console and file
log_file = 'test_historical_fetcher.log'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def main():
    # Initialize summary data structures
    summary_data = {
        'leagues_processed': 0,
        'leagues_skipped': 0,
        'total_matches': 0,
        'new_matches': 0,
        'countries': set(),
        'teams': set(),
        'leagues': [],
        'season': '2023-2024',
        'leagues_with_matches': defaultdict(int),
        'teams_by_country': defaultdict(set)
    }
    
    # Specify league IDs to process
    league_ids = [203, 204]  # Superliga and Ligue 1
    
    # Load league data from leagues.json
    leagues_data = {}
    logger.info('Loading league data from leagues.json')
    with open('leagues.json', 'r') as f:
        data = json.load(f)
        for league in data['results']:
            leagues_data[league['id']] = {
                'name': league['name'],
                'country': league['country']['name'].title() if 'country' in league else 'Unknown'
            }
    
    # Process specified leagues
    for league_id in league_ids:
        if league_id in leagues_data:
            league_name = leagues_data[league_id]['name']
            country = leagues_data[league_id]['country']
            logger.info(f'Processing league ID {league_id} ({league_name}) from {country}')
            
            # Simulate fetching matches
            matches = 180  # Simulated number of matches
            
            # Update summary data
            summary_data['leagues_processed'] += 1
            summary_data['total_matches'] += matches
            summary_data['countries'].add(country)
            summary_data['leagues'].append((league_id, league_name, country))
            summary_data['leagues_with_matches'][league_name] = matches
            
            # Simulate team data
            if league_id == 203:  # Superliga
                teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
            else:  # Ligue 1
                teams = ['Team F', 'Team G', 'Team H', 'Team I', 'Team J']
            
            for team in teams:
                summary_data['teams'].add(team)
                summary_data['teams_by_country'][country].add(team)
    
    # Display summary information
    logger.info('='*80)
    logger.info('SUMMARY OF DATA FETCHED')
    logger.info('='*80)
    logger.info(f"Season: {summary_data['season']}")
    logger.info(f"Leagues processed: {summary_data['leagues_processed']}")
    logger.info(f"Leagues skipped: {summary_data['leagues_skipped']}")
    logger.info(f"Total matches fetched: {summary_data['total_matches']}")
    logger.info(f"New matches added to database: {summary_data['new_matches']}")
    logger.info(f"Countries represented: {len(summary_data['countries'])}")
    logger.info(f"Total teams: {len(summary_data['teams'])}")
    
    # Display leagues and match counts
    if summary_data['leagues_with_matches']:
        logger.info('\nLeagues and match counts:')
        for league_name, match_count in sorted(summary_data['leagues_with_matches'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {league_name}: {match_count} matches")
    
    # Display countries and team counts
    if summary_data['teams_by_country']:
        logger.info('\nCountries and team counts:')
        for country, teams in sorted(summary_data['teams_by_country'].items(), key=lambda x: len(x[1]), reverse=True):
            logger.info(f"  {country}: {len(teams)} teams")
            for team in sorted(teams):
                logger.info(f"    - {team}")
    
    logger.info('='*80)
    logger.info('Script completed successfully')

if __name__ == '__main__':
    main()