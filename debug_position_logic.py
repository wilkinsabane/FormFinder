#!/usr/bin/env python3
"""
Debug script to test the position logic and identify the date filtering issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime

def debug_position_logic():
    """Debug the position logic for fixture 20819."""
    load_config()
    
    with get_db_session() as db_session:
        # Get fixture details
        fixture_query = text("""
            SELECT home_team_id, away_team_id, league_id, match_date
            FROM fixtures
            WHERE id = 20819
        """)
        
        fixture = db_session.execute(fixture_query).fetchone()
        if not fixture:
            print("Fixture 20819 not found")
            return
            
        home_team_id, away_team_id, league_id, match_date = fixture
        print(f"Fixture 20819: Home team {home_team_id}, Away team {away_team_id}, League {league_id}")
        print(f"Match date: {match_date}")
        print()
        
        # Test the current logic (with date filter)
        print("=== Current Logic (with date filter) ===")
        for team_id, team_name in [(home_team_id, "Home"), (away_team_id, "Away")]:
            query = text("""
                SELECT position, updated_at
                FROM standings
                WHERE team_id = :team_id
                    AND league_id = :league_id
                    AND updated_at <= :match_date
                ORDER BY updated_at DESC
                LIMIT 1
            """)
            
            result = db_session.execute(query, {
                'team_id': team_id,
                'league_id': league_id,
                'match_date': match_date
            }).fetchone()
            
            if result:
                print(f"{team_name} team {team_id}: Position {result[0]}, Updated: {result[1]}")
            else:
                print(f"{team_name} team {team_id}: No position data found with date filter")
                
                # Try fallback query
                fallback_query = text("""
                    SELECT position, updated_at
                    FROM standings
                    WHERE team_id = :team_id
                        AND league_id = :league_id
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                
                fallback_result = db_session.execute(fallback_query, {
                    'team_id': team_id,
                    'league_id': league_id
                }).fetchone()
                
                if fallback_result:
                    print(f"  Fallback: Position {fallback_result[0]}, Updated: {fallback_result[1]}")
                    print(f"  Using default position 10 because updated_at > match_date")
                else:
                    print(f"  No position data at all - using default 10")
        
        print()
        print("=== Proposed Fix (use most recent data) ===")
        for team_id, team_name in [(home_team_id, "Home"), (away_team_id, "Away")]:
            query = text("""
                SELECT position, updated_at
                FROM standings
                WHERE team_id = :team_id
                    AND league_id = :league_id
                ORDER BY updated_at DESC
                LIMIT 1
            """)
            
            result = db_session.execute(query, {
                'team_id': team_id,
                'league_id': league_id
            }).fetchone()
            
            if result:
                print(f"{team_name} team {team_id}: Position {result[0]}, Updated: {result[1]}")
            else:
                print(f"{team_name} team {team_id}: No position data found - using default 10")

if __name__ == "__main__":
    debug_position_logic()