#!/usr/bin/env python3
"""
Populate team_performance_states for remaining teams without coverage.

This script identifies teams that don't have performance states and populates them
by analyzing their recent fixtures and calculating performance metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_teams_without_states():
    """Get list of teams that don't have performance states."""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT t.id, t.name, t.league_id, l.name as league_name
            FROM teams t
            JOIN leagues l ON t.league_id = l.league_pk
            WHERE t.id NOT IN (SELECT DISTINCT team_id FROM team_performance_states)
            ORDER BY t.id
        """))
        return result.fetchall()

def get_team_recent_fixtures(team_id, limit=10):
    """Get recent fixtures for a team to calculate performance state (no date restriction for remaining teams)."""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                f.id as fixture_id,
                f.match_date,
                f.home_team_id,
                f.away_team_id,
                f.home_score,
                f.away_score,
                f.status,
                ht.name as home_team_name,
                at.name as away_team_name
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE (f.home_team_id = :team_id OR f.away_team_id = :team_id)
                AND f.status = 'finished'
                AND f.home_score IS NOT NULL
                AND f.away_score IS NOT NULL
            ORDER BY f.match_date DESC
            LIMIT :limit
        """), {"team_id": team_id, "limit": limit})
        return result.fetchall()

def calculate_performance_state(team_id, fixtures):
    """Calculate performance state based on recent fixtures."""
    if not fixtures:
        return {
            'performance_state': 'average',
            'state_score': 0.5,
            'goals_scored': 0.0,
            'goals_conceded': 0.0,
            'goal_difference': 0.0,
            'win_rate': 0.0,
            'points_per_game': 0.0,
            'form_streak': '',
            'matches_considered': 0
        }
    
    total_goals_scored = 0
    total_goals_conceded = 0
    total_points = 0
    form_results = []
    
    for fixture in fixtures:
        is_home = fixture.home_team_id == team_id
        
        if is_home:
            goals_scored = fixture.home_score
            goals_conceded = fixture.away_score
        else:
            goals_scored = fixture.away_score
            goals_conceded = fixture.home_score
        
        total_goals_scored += goals_scored
        total_goals_conceded += goals_conceded
        
        # Calculate points (3 for win, 1 for draw, 0 for loss)
        if goals_scored > goals_conceded:
            points = 3
            form_results.append('W')
        elif goals_scored == goals_conceded:
            points = 1
            form_results.append('D')
        else:
            points = 0
            form_results.append('L')
        
        total_points += points
    
    num_matches = len(fixtures)
    avg_goals_scored = total_goals_scored / num_matches
    avg_goals_conceded = total_goals_conceded / num_matches
    goal_difference = avg_goals_scored - avg_goals_conceded
    win_rate = len([r for r in form_results if r == 'W']) / num_matches
    points_per_game = total_points / num_matches
    form_streak = ''.join(form_results[:5])  # Last 5 matches
    
    # Calculate state score (0.0 to 1.0)
    # Based on points per game (0-3), goal difference, and win rate
    state_score = (
        (points_per_game / 3.0) * 0.5 +  # 50% weight on points
        min(max((goal_difference + 2) / 4, 0), 1) * 0.3 +  # 30% weight on goal diff
        win_rate * 0.2  # 20% weight on win rate
    )
    
    # Classify performance state
    if state_score >= 0.8:
        performance_state = 'excellent'
    elif state_score >= 0.6:
        performance_state = 'good'
    elif state_score >= 0.4:
        performance_state = 'average'
    elif state_score >= 0.2:
        performance_state = 'poor'
    else:
        performance_state = 'terrible'
    
    return {
        'performance_state': performance_state,
        'state_score': state_score,
        'goals_scored': avg_goals_scored,
        'goals_conceded': avg_goals_conceded,
        'goal_difference': goal_difference,
        'win_rate': win_rate,
        'points_per_game': points_per_game,
        'form_streak': form_streak,
        'matches_considered': num_matches
    }

def insert_performance_state(team_id, league_id, state_data):
    """Insert performance state into database."""
    with get_db_session() as session:
        session.execute(text("""
            INSERT INTO team_performance_states (
                team_id, league_id, state_date, performance_state, state_score,
                goals_scored, goals_conceded, goal_difference, win_rate,
                points_per_game, form_streak, matches_considered, home_away_context
            ) VALUES (
                :team_id, :league_id, :state_date, :performance_state, :state_score,
                :goals_scored, :goals_conceded, :goal_difference, :win_rate,
                :points_per_game, :form_streak, :matches_considered, 'overall'
            )
        """), {
            'team_id': team_id,
            'league_id': league_id,
            'state_date': datetime.utcnow(),
            'performance_state': state_data['performance_state'],
            'state_score': state_data['state_score'],
            'goals_scored': state_data['goals_scored'],
            'goals_conceded': state_data['goals_conceded'],
            'goal_difference': state_data['goal_difference'],
            'win_rate': state_data['win_rate'],
            'points_per_game': state_data['points_per_game'],
            'form_streak': state_data['form_streak'],
            'matches_considered': state_data['matches_considered']
        })
        session.commit()

def main():
    """Main function to populate remaining team performance states."""
    logger.info("Starting population of remaining team performance states...")
    
    # Load configuration
    load_config()
    
    # Get teams without performance states
    teams_without_states = get_teams_without_states()
    logger.info(f"Found {len(teams_without_states)} teams without performance states")
    
    if not teams_without_states:
        logger.info("All teams already have performance states!")
        return
    
    processed_count = 0
    skipped_count = 0
    
    for team in teams_without_states:
        team_id, team_name, league_id, league_name = team
        
        try:
            # Get recent fixtures for this team
            fixtures = get_team_recent_fixtures(team_id)
            
            if not fixtures:
                logger.warning(f"No fixtures found for team {team_name} (ID: {team_id})")
                skipped_count += 1
                continue
            
            # Calculate performance state
            state_data = calculate_performance_state(team_id, fixtures)
            
            # Insert into database
            insert_performance_state(team_id, league_id, state_data)
            
            logger.info(f"Processed team {team_name} (ID: {team_id}) - State: {state_data['performance_state']} (Score: {state_data['state_score']:.3f})")
            processed_count += 1
            
            # Progress update every 50 teams
            if processed_count % 50 == 0:
                logger.info(f"Progress: {processed_count}/{len(teams_without_states)} teams processed")
                
        except Exception as e:
            logger.error(f"Error processing team {team_name} (ID: {team_id}): {e}")
            skipped_count += 1
            continue
    
    logger.info(f"Completed! Processed: {processed_count}, Skipped: {skipped_count}")
    
    # Final coverage check
    with get_db_session() as session:
        total_teams_result = session.execute(text("SELECT COUNT(*) FROM teams"))
        total_teams = total_teams_result.scalar()
        
        teams_with_states_result = session.execute(text("""
            SELECT COUNT(DISTINCT team_id) FROM team_performance_states
        """))
        teams_with_states = teams_with_states_result.scalar()
        
        coverage_pct = (teams_with_states / total_teams) * 100
        logger.info(f"Final coverage: {teams_with_states}/{total_teams} teams ({coverage_pct:.1f}%)")

if __name__ == "__main__":
    main()