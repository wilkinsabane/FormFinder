#!/usr/bin/env python3
"""
Script to add new seasons to existing leagues without resetting the database.
This preserves all existing data and only adds new league entries for specified seasons.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import League
from formfinder.database import get_db_session


def add_seasons(new_seasons=None):
    """Add new seasons to existing leagues without data loss."""
    
    if new_seasons is None:
        new_seasons = ["2022-2023", "2026-2027"]  # Default new seasons
    
    # Load configuration
    load_config()
    
    # Load leagues from JSON
    leagues_file = Path("leagues.json")
    if not leagues_file.exists():
        print("ERROR: leagues.json not found!")
        return
    
    try:
        with open(leagues_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        leagues_data = {}
        for league in data.get('results', []):
            leagues_data[league['id']] = {
                'name': league['name'],
                'country': league['country']['name']
            }
        
        print(f"Loaded {len(leagues_data)} leagues from leagues.json")
        
        with get_db_session() as session:
            total_created = 0
            
            for season in new_seasons:
                created_count = 0
                
                for league_id, league_info in leagues_data.items():
                    # Check if league already exists for this season
                    existing = session.query(League).filter_by(
                        id=league_id, 
                        season=season
                    ).first()
                    
                    if not existing:
                        league = League(
                            id=league_id,
                            name=league_info['name'],
                            country=league_info['country'],
                            season=season
                        )
                        session.add(league)
                        created_count += 1
                        print(f"Adding league: {league_info['name']} ({league_info['country']}) - {season}")
                
                total_created += created_count
                print(f"Added {created_count} new leagues for season {season}")
            
            # Commit the transaction
            session.commit()
            print(f"Successfully added {total_created} new league entries across {len(new_seasons)} seasons")
            print("All existing data has been preserved!")
            
    except Exception as e:
        print(f"ERROR: Failed to add seasons: {e}")
        raise


def get_available_seasons():
    """Show which seasons are already in the database."""
    from formfinder.config import load_config
    load_config()
    
    with get_db_session() as session:
        existing_seasons = session.query(League.season).distinct().all()
        existing_seasons = [s[0] for s in sorted(existing_seasons)]
        
        print("=== CURRENT DATABASE STATUS ===")
        print(f"Existing seasons: {existing_seasons}")
        
        total_leagues = session.query(League).count()
        print(f"Total league entries: {total_leagues}")
        
        for season in existing_seasons:
            count = session.query(League).filter(League.season == season).count()
            print(f"  {season}: {count} leagues")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Add new seasons to existing leagues')
    parser.add_argument('--seasons', nargs='+', help='List of seasons to add (e.g., 2022-2023 2026-2027)')
    parser.add_argument('--list', action='store_true', help='Show existing seasons')
    
    args = parser.parse_args()
    
    if args.list:
        get_available_seasons()
    else:
        seasons_to_add = args.seasons if args.seasons else ["2022-2023", "2026-2027"]
        add_seasons(seasons_to_add)