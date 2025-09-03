#!/usr/bin/env python3
"""Script to check available teams in the database."""

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text

def main():
    """Check available teams in the database."""
    load_config()
    config = get_config()
    
    engine = create_engine(config.get_database_url())
    
    with engine.connect() as conn:
        # Check if teams table exists (PostgreSQL syntax)
        result = conn.execute(text("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'teams'
        """))
        
        if not result.fetchone():
            print("❌ Teams table not found in database")
            return
        
        # Get available teams
        result = conn.execute(text("SELECT id, name FROM teams LIMIT 20"))
        teams = result.fetchall()
        
        if teams:
            print("✅ Available Teams:")
            print("=" * 40)
            for team in teams:
                print(f"ID: {team.id:4d} | Name: {team.name}")
            print("=" * 40)
            print(f"Total teams shown: {len(teams)}")
        else:
            print("❌ No teams found in database")
            
        # Also check fixtures for team IDs
        print("\n🔍 Checking fixtures for team IDs...")
        result = conn.execute(text("""
            SELECT DISTINCT home_team_id, away_team_id 
            FROM fixtures 
            WHERE home_team_id IS NOT NULL AND away_team_id IS NOT NULL
            LIMIT 10
        """))
        
        fixture_teams = result.fetchall()
        if fixture_teams:
            print("Team IDs found in fixtures:")
            for fixture in fixture_teams:
                print(f"Home: {fixture.home_team_id}, Away: {fixture.away_team_id}")
        else:
            print("No team IDs found in fixtures")

if __name__ == "__main__":
    main()