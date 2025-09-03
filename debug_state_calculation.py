#!/usr/bin/env python3
"""
Debug script to investigate Markov state calculation issues.
Checks if team performance states are being calculated and stored properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from formfinder.config import load_config
from formfinder.database import get_db_session, TeamPerformanceState
from formfinder.markov_state_classifier import MarkovStateClassifier
from sqlalchemy import text
import json

def main():
    """Debug Markov state calculation issues."""
    load_config()
    
    with get_db_session() as session:
        # Check total team performance states
        total_states = session.query(TeamPerformanceState).count()
        print(f"Total team performance states in database: {total_states}")
        
        # Check contexts available
        contexts_query = """
            SELECT DISTINCT home_away_context, COUNT(*) as count
            FROM team_performance_states
            GROUP BY home_away_context
            ORDER BY count DESC
        """
        contexts = session.execute(text(contexts_query)).fetchall()
        print("\nContexts available:")
        for context, count in contexts:
            print(f"  {context}: {count} records")
        
        # Check state distribution
        states_query = """
            SELECT performance_state, COUNT(*) as count
            FROM team_performance_states
            GROUP BY performance_state
            ORDER BY count DESC
        """
        states = session.execute(text(states_query)).fetchall()
        print("\nState distribution:")
        for state, count in states:
            print(f"  {state}: {count} records")
        
        # Check recent states for teams in pre_computed_features
        recent_states_query = """
            SELECT DISTINCT tps.team_id, tps.performance_state, tps.state_score, 
                   tps.state_date, tps.home_away_context
            FROM team_performance_states tps
            INNER JOIN (
                SELECT DISTINCT home_team_id as team_id FROM pre_computed_features
                UNION
                SELECT DISTINCT away_team_id as team_id FROM pre_computed_features
            ) pcf ON tps.team_id = pcf.team_id
            ORDER BY tps.team_id, tps.state_date DESC
            LIMIT 20
        """
        recent_states = session.execute(text(recent_states_query)).fetchall()
        print("\nRecent states for teams in pre_computed_features:")
        for team_id, state, score, date, context in recent_states:
            print(f"  Team {team_id}: {state} (score: {score:.3f}) on {date} [{context}]")
        
        # Test state calculation for a specific team
        print("\n" + "="*50)
        print("Testing state calculation for a specific team...")
        
        # Get a team from pre_computed_features
        team_query = "SELECT DISTINCT home_team_id, league_id FROM pre_computed_features LIMIT 1"
        team_result = session.execute(text(team_query)).fetchone()
        
        if team_result:
            team_id, league_id = team_result
            print(f"Testing team {team_id} in league {league_id}")
            
            # Initialize classifier
            classifier = MarkovStateClassifier()
            
            # Calculate performance score
            reference_date = datetime.now()
            metrics = classifier.calculate_performance_score(
                team_id, league_id, reference_date, 'overall'
            )
            
            print(f"\nCalculated metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            
            # Check what state this would be classified as
            performance_score = metrics.get('performance_score', 0.4)
            classified_state = classifier.classify_state(performance_score)
            print(f"\nClassified state: {classified_state}")
            print(f"Performance score: {performance_score:.3f}")
            
            # Check thresholds
            print(f"\nState thresholds:")
            for state, threshold in classifier.thresholds.items():
                print(f"  {state}: >= {threshold['min_score']}")

if __name__ == "__main__":
    main()