#!/usr/bin/env python3
"""
Simple script to test populating team performance states
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.markov_state_classifier import MarkovStateClassifier
from formfinder.database import get_db_session
from sqlalchemy import text

def test_populate_states():
    """Test populating team performance states for league 39"""
    try:
        # Load configuration
        load_config()
        print("Configuration loaded successfully")
        
        # Initialize classifier
        classifier = MarkovStateClassifier()
        print("MarkovStateClassifier initialized")
        
        # Get teams for league 207 (Premier League)
        league_id = 207
        teams_query = """
        SELECT DISTINCT team_id 
        FROM (
            SELECT home_team_id as team_id FROM fixtures WHERE league_id = :league_id
            UNION
            SELECT away_team_id as team_id FROM fixtures WHERE league_id = :league_id
        ) teams
        LIMIT 5
        """
        
        with get_db_session() as session:
            teams = session.execute(
                text(teams_query),
                {'league_id': league_id}
            ).fetchall()
        
        print(f"Found {len(teams)} teams for league {league_id}")
        
        # Process states for first few teams
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        total_states = 0
        
        for i, team_row in enumerate(teams[:3]):  # Process only first 3 teams
            team_id = team_row[0]
            print(f"Processing team {team_id} ({i+1}/{min(3, len(teams))})...")
            
            try:
                states = classifier.process_team_states(
                    team_id=team_id,
                    league_id=league_id,
                    start_date=start_date,
                    end_date=end_date
                )
                total_states += len(states)
                print(f"  Created {len(states)} states for team {team_id}")
                
            except Exception as e:
                print(f"  Error processing team {team_id}: {e}")
                continue
        
        print(f"\nTotal states created: {total_states}")
        
        # Check final count
        with get_db_session() as session:
            count = session.execute(
                text("SELECT COUNT(*) FROM team_performance_states")
            ).scalar()
            print(f"Total records in team_performance_states: {count}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_populate_states()