#!/usr/bin/env python3
"""
Script to populate leagues table from leagues.json
This must be run before historical_fetcher.py --save-to-db
"""

import asyncio
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from formfinder.database import init_database
from formfinder.config import load_config
from formfinder.database import League
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


async def populate_leagues():
    """Populate leagues table from leagues.json for 2023-2024 season."""
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize database
    init_database(config.get_database_url())
    
    # Create engine and session
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    
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
        
        session = Session()
        
        # Create leagues for multiple seasons
        seasons = ["2023-2024", "2024-2025", "2025-2026"]
        total_created = 0
        
        for season in seasons:
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
                    print(f"Creating league: {league_info['name']} ({league_info['country']}) - {season}")
            
            total_created += created_count
            print(f"Created {created_count} leagues for season {season}")
        
        session.commit()
        session.close()
        
        print(f"Successfully created {total_created} leagues across all seasons")
        
    except Exception as e:
        print(f"ERROR: Failed to populate leagues: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(populate_leagues())