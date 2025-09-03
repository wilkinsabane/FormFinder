#!/usr/bin/env python3
"""
Debug why pre_computed_features shows 'average' states when team_performance_states has data.
"""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime

def main():
    load_config()
    
    with get_db_session() as session:
        # Get a sample fixture from pre_computed_features
        fixture_query = text("""
            SELECT pcf.fixture_id, f.home_team_id, f.away_team_id, f.league_id, f.match_date,
                   pcf.markov_home_current_state, pcf.markov_away_current_state
            FROM pre_computed_features pcf
            JOIN fixtures f ON pcf.fixture_id = f.id
            LIMIT 5
        """)
        
        fixtures = session.execute(fixture_query).fetchall()
        
        print("Sample fixtures from pre_computed_features:")
        print("Fixture ID | Home Team | Away Team | League | Match Date | Home State | Away State")
        print("-" * 90)
        
        for fixture in fixtures:
            fixture_id, home_team, away_team, league_id, match_date, home_state, away_state = fixture
            print(f"{fixture_id:10} | {home_team:9} | {away_team:9} | {league_id:6} | {match_date} | {home_state:10} | {away_state}")
            
            # Check if these teams have states in team_performance_states
            for team_id, context in [(home_team, 'home'), (away_team, 'away')]:
                state_query = text("""
                    SELECT performance_state, state_score, state_date
                    FROM team_performance_states
                    WHERE team_id = :team_id
                      AND league_id = :league_id
                      AND home_away_context = :context
                      AND state_date <= :reference_date
                    ORDER BY state_date DESC
                    LIMIT 1
                """)
                
                state_result = session.execute(state_query, {
                    'team_id': team_id,
                    'league_id': league_id,
                    'context': context,
                    'reference_date': match_date
                }).fetchone()
                
                if state_result:
                    print(f"  Team {team_id} ({context}): Found state '{state_result[0]}' with score {state_result[1]} on {state_result[2]}")
                else:
                    print(f"  Team {team_id} ({context}): No state found for this team/league/context/date combination")
                    
                    # Check if team exists in any context
                    any_state_query = text("""
                        SELECT performance_state, state_score, state_date, home_away_context
                        FROM team_performance_states
                        WHERE team_id = :team_id
                        ORDER BY state_date DESC
                        LIMIT 3
                    """)
                    
                    any_states = session.execute(any_state_query, {'team_id': team_id}).fetchall()
                    if any_states:
                        print(f"    But team {team_id} has these states:")
                        for state in any_states:
                            print(f"      {state[0]} (score: {state[1]}, date: {state[2]}, context: {state[3]})")
                    else:
                        print(f"    Team {team_id} has no states at all in team_performance_states")
            
            print()

if __name__ == '__main__':
    main()