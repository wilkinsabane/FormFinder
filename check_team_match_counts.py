#!/usr/bin/env python3
"""
Check how many matches teams have played to find teams with sufficient history.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

# Load configuration
load_config()

def check_team_match_counts():
    """Check match counts for teams in pre_computed_features."""
    
    with get_db_session() as session:
        # Get teams from pre_computed_features and their match counts
        query = """
            WITH team_matches AS (
                SELECT 
                    t.team_id,
                    t.league_id,
                    COUNT(*) as match_count
                FROM (
                    SELECT home_team_id as team_id, league_id FROM fixtures WHERE status = 'finished'
                    UNION ALL
                    SELECT away_team_id as team_id, league_id FROM fixtures WHERE status = 'finished'
                ) t
                GROUP BY t.team_id, t.league_id
            ),
            precomputed_teams AS (
                SELECT DISTINCT home_team_id as team_id, league_id FROM pre_computed_features
                UNION
                SELECT DISTINCT away_team_id as team_id, league_id FROM pre_computed_features
            )
            SELECT 
                pt.team_id,
                pt.league_id,
                COALESCE(tm.match_count, 0) as match_count
            FROM precomputed_teams pt
            LEFT JOIN team_matches tm ON pt.team_id = tm.team_id AND pt.league_id = tm.league_id
            ORDER BY match_count DESC
            LIMIT 20
        """
        
        results = session.execute(text(query)).fetchall()
        
        print("Team match counts (top 20):")
        print("Team ID | League ID | Match Count")
        print("-" * 35)
        
        teams_with_enough_matches = []
        
        for result in results:
            team_id = result[0]
            league_id = result[1]
            match_count = result[2]
            
            print(f"{team_id:7} | {league_id:9} | {match_count:11}")
            
            if match_count >= 3:
                teams_with_enough_matches.append((team_id, league_id, match_count))
        
        print(f"\nTeams with >= 3 matches: {len(teams_with_enough_matches)}")
        
        if teams_with_enough_matches:
            print("\nTesting state calculation for teams with sufficient matches:")
            
            # Test a few teams with enough matches
            from formfinder.markov_state_classifier import MarkovStateClassifier
            from datetime import datetime
            
            state_classifier = MarkovStateClassifier()
            
            for team_id, league_id, match_count in teams_with_enough_matches[:3]:
                print(f"\n--- Team {team_id} in league {league_id} ({match_count} matches) ---")
                
                try:
                    # Get a recent date for this team
                    date_query = """
                        SELECT MAX(match_date) 
                        FROM fixtures 
                        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
                          AND league_id = :league_id
                          AND status = 'finished'
                    """
                    
                    date_result = session.execute(text(date_query), {
                        'team_id': team_id,
                        'league_id': league_id
                    }).fetchone()
                    
                    if date_result and date_result[0]:
                        reference_date = date_result[0]
                        
                        # Test overall context
                        metrics = state_classifier.calculate_performance_score(
                            team_id, league_id, reference_date, 'overall'
                        )
                        
                        print(f"Metrics: {metrics}")
                        
                        if metrics['matches_analyzed'] >= 3:
                            calculated_state = state_classifier.classify_state(metrics['performance_score'])
                            print(f"Calculated state: {calculated_state}")
                            print(f"Performance score: {metrics['performance_score']:.3f}")
                        else:
                            print(f"Still not enough matches analyzed: {metrics['matches_analyzed']}")
                    
                except Exception as e:
                    print(f"Error testing team {team_id}: {e}")

if __name__ == "__main__":
    check_team_match_counts()