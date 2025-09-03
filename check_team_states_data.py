#!/usr/bin/env python3

import sys
from pathlib import Path
from sqlalchemy import text

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.config import load_config
from formfinder.database import get_db_session

def check_team_states_data():
    """Check the quality of data in team_performance_states table."""
    
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Check total count
        count_query = text("SELECT COUNT(*) FROM team_performance_states")
        total_count = session.execute(count_query).scalar()
        print(f"Total team performance states: {total_count}")
        
        # Check recent records
        recent_query = text("""
            SELECT team_id, fixture_id, performance_state, state_score, 
                   goals_scored, goals_conceded, matches_considered, 
                   win_rate, points_per_game, form_streak, created_at
            FROM team_performance_states 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        recent_records = session.execute(recent_query).fetchall()
        print("\nRecent records:")
        print("-" * 120)
        print(f"{'Team':<6} {'Fixture':<8} {'State':<10} {'Score':<6} {'Goals':<8} {'Matches':<8} {'WinRate':<8} {'PPG':<6} {'Streak':<10} {'Created':<20}")
        print("-" * 120)
        
        for r in recent_records:
            try:
                team_id = r.team_id if r.team_id is not None else 0
                fixture_id = r.fixture_id if r.fixture_id is not None else 0
                goals_str = f"{r.goals_scored or 0}/{r.goals_conceded or 0}"
                created_str = r.created_at.strftime('%Y-%m-%d %H:%M') if r.created_at else 'N/A'
                perf_state = r.performance_state if r.performance_state is not None else 'N/A'
                state_score = r.state_score if r.state_score is not None else 0.0
                win_rate = r.win_rate if r.win_rate is not None else 0.0
                ppg = r.points_per_game if r.points_per_game is not None else 0.0
                form_streak = r.form_streak if r.form_streak is not None else 'N/A'
                matches = r.matches_considered if r.matches_considered is not None else 0
                print(f"{team_id:<6} {fixture_id:<8} {perf_state:<10} {state_score:<6.3f} {goals_str:<8} {matches:<8} {win_rate:<8.3f} {ppg:<6.2f} {form_streak:<10} {created_str:<20}")
            except Exception as e:
                print(f"Error formatting row: {e}")
                print(f"Raw row data: {dict(r._mapping)}")
        
        # Check for potential data quality issues
        print("\nData Quality Checks:")
        print("-" * 50)
        
        # Check for records with 0 matches considered
        zero_matches_query = text("SELECT COUNT(*) FROM team_performance_states WHERE matches_considered = 0")
        zero_matches = session.execute(zero_matches_query).scalar()
        print(f"Records with 0 matches considered: {zero_matches}")
        
        # Check for records with null performance states
        null_state_query = text("SELECT COUNT(*) FROM team_performance_states WHERE performance_state IS NULL")
        null_states = session.execute(null_state_query).scalar()
        print(f"Records with null performance state: {null_states}")
        
        # Check distribution of performance states
        state_dist_query = text("""
            SELECT performance_state, COUNT(*) as count 
            FROM team_performance_states 
            GROUP BY performance_state 
            ORDER BY count DESC
        """)
        
        state_distribution = session.execute(state_dist_query).fetchall()
        print("\nPerformance state distribution:")
        for state, count in state_distribution:
            print(f"  {state}: {count}")
        
        # Check average values
        avg_query = text("""
            SELECT 
                AVG(state_score) as avg_score,
                AVG(goals_scored) as avg_goals_scored,
                AVG(goals_conceded) as avg_goals_conceded,
                AVG(matches_considered) as avg_matches,
                AVG(win_rate) as avg_win_rate,
                AVG(points_per_game) as avg_ppg
            FROM team_performance_states
            WHERE matches_considered > 0
        """)
        
        avg_result = session.execute(avg_query).fetchone()
        print("\nAverage values (excluding 0-match records):")
        print(f"  Average score: {avg_result.avg_score:.3f}")
        print(f"  Average goals scored: {avg_result.avg_goals_scored:.2f}")
        print(f"  Average goals conceded: {avg_result.avg_goals_conceded:.2f}")
        print(f"  Average matches considered: {avg_result.avg_matches:.1f}")
        print(f"  Average win rate: {avg_result.avg_win_rate:.3f}")
        print(f"  Average points per game: {avg_result.avg_ppg:.2f}")

if __name__ == '__main__':
    check_team_states_data()