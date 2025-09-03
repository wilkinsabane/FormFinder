#!/usr/bin/env python3
"""
Populate team_performance_states table with historical data.

This script analyzes historical fixtures to calculate team performance states
(momentum, form, sentiment) and populates the team_performance_states table.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.config import load_config
from formfinder.database import get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('populate_team_states.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def calculate_team_performance_state(team_id: int, match_date: datetime, session) -> Dict[str, any]:
    """
    Calculate team performance state based on recent form and results.
    
    Args:
        team_id: ID of the team
        match_date: Date to calculate state for
        session: Database session
        
    Returns:
        Dictionary with performance state information
    """
    # Look back 30 days for recent form
    lookback_date = match_date - timedelta(days=30)
    
    # Get recent matches for the team
    query = text("""
        SELECT 
            CASE WHEN home_team_id = :team_id THEN 'home' ELSE 'away' END as venue,
            CASE WHEN home_team_id = :team_id THEN home_score ELSE away_score END as goals_for,
            CASE WHEN home_team_id = :team_id THEN away_score ELSE home_score END as goals_against,
            match_date,
            CASE 
                WHEN (home_team_id = :team_id AND home_score > away_score) OR 
                     (away_team_id = :team_id AND away_score > home_score) THEN 'W'
                WHEN home_score = away_score THEN 'D'
                ELSE 'L'
            END as result,
            (CASE WHEN home_team_id = :team_id THEN home_score ELSE away_score END) +
            (CASE WHEN home_team_id = :team_id THEN away_score ELSE home_score END) as total_goals
        FROM fixtures
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
            AND match_date < :match_date
            AND match_date >= :lookback_date
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
            AND status = 'finished'
        ORDER BY match_date DESC
        LIMIT 10
    """)
    
    result = session.execute(query, {
        'team_id': team_id,
        'match_date': match_date,
        'lookback_date': lookback_date
    })
    
    matches = result.fetchall()
    
    if not matches:
        # No recent matches, return neutral state
        return {
            'momentum_score': 0.0,
            'form_trend': 'stable',
            'performance_state': 'average',
            'confidence_level': 0.5,
            'matches_analyzed': 0,
            'goals_for': 0,
            'goals_against': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0
        }
    
    # Calculate momentum based on recent results
    momentum_points = 0
    total_goals_for = 0
    total_goals_against = 0
    wins = 0
    draws = 0
    losses = 0
    
    # Weight recent matches more heavily
    for i, match in enumerate(matches):
        weight = 1.0 - (i * 0.1)  # Decrease weight for older matches
        
        if match.result == 'W':
            momentum_points += 3 * weight
            wins += 1
        elif match.result == 'D':
            momentum_points += 1 * weight
            draws += 1
        else:
            losses += 1
        
        total_goals_for += match.goals_for
        total_goals_against += match.goals_against
    
    # Normalize momentum score
    max_possible_points = sum(3 * (1.0 - i * 0.1) for i in range(len(matches)))
    momentum_score = momentum_points / max_possible_points if max_possible_points > 0 else 0.0
    
    # Calculate form trend
    if len(matches) >= 3:
        recent_results = [m.result for m in matches[:3]]
        recent_wins = recent_results.count('W')
        if recent_wins >= 2:
            form_trend = 'improving'
        elif recent_wins == 0:
            form_trend = 'declining'
        else:
            form_trend = 'stable'
    else:
        form_trend = 'stable'
    
    # Determine performance state
    if momentum_score >= 0.7:
        performance_state = 'excellent'
    elif momentum_score >= 0.5:
        performance_state = 'good'
    elif momentum_score >= 0.3:
        performance_state = 'average'
    else:
        performance_state = 'poor'
    
    # Calculate confidence level based on number of matches
    confidence_level = min(1.0, len(matches) / 5.0)
    
    return {
        'momentum_score': round(momentum_score, 3),
        'form_trend': form_trend,
        'performance_state': performance_state,
        'confidence_level': round(confidence_level, 3),
        'matches_analyzed': len(matches),
        'goals_for': total_goals_for,
        'goals_against': total_goals_against,
        'wins': wins,
        'draws': draws,
        'losses': losses
    }


def get_teams_and_fixtures(session, limit_teams: Optional[int] = None) -> List[Dict]:
    """
    Get all teams and their fixtures for state calculation.
    
    Args:
        session: Database session
        limit_teams: Optional limit on number of teams to process
        
    Returns:
        List of team-fixture combinations
    """
    # Get all teams that have played matches
    base_query = """
        SELECT DISTINCT 
            t.id as team_id,
            t.name as team_name,
            f.id as fixture_id,
            f.match_date,
            f.league_id,
            CASE WHEN f.home_team_id = t.id THEN 'home' ELSE 'away' END as venue
        FROM teams t
        JOIN fixtures f ON (f.home_team_id = t.id OR f.away_team_id = t.id)
        WHERE f.home_score IS NOT NULL 
            AND f.away_score IS NOT NULL
            AND f.status = 'finished'
            AND f.match_date >= '2020-09-01'  -- Focus on recent seasons
        ORDER BY t.id, f.match_date
    """
    
    if limit_teams:
        query = text(base_query + f" LIMIT {limit_teams * 50}")
    else:
        query = text(base_query)
    
    result = session.execute(query)
    return [{
        'team_id': row.team_id,
        'team_name': row.team_name,
        'fixture_id': row.fixture_id,
        'match_date': row.match_date,
        'league_id': row.league_id,
        'venue': row.venue
    } for row in result.fetchall()]


def populate_team_performance_states(session, limit_teams: Optional[int] = None, force: bool = False):
    """
    Populate team_performance_states table with historical data.
    
    Args:
        session: Database session
        limit_teams: Optional limit on number of teams to process
        force: If True, regenerate existing records
    """
    logger.info("Starting team performance states population")
    
    # Get teams and fixtures
    team_fixtures = get_teams_and_fixtures(session, limit_teams)
    logger.info(f"Found {len(team_fixtures)} team-fixture combinations to process")
    
    if not team_fixtures:
        logger.warning("No team fixtures found for processing")
        return
    
    # Group by team for processing
    teams_data = {}
    for tf in team_fixtures:
        team_id = tf['team_id']
        if team_id not in teams_data:
            teams_data[team_id] = {
                'name': tf['team_name'],
                'fixtures': []
            }
        teams_data[team_id]['fixtures'].append(tf)
    
    logger.info(f"Processing {len(teams_data)} unique teams")
    
    total_inserted = 0
    total_errors = 0
    
    for team_id, team_data in teams_data.items():
        try:
            logger.info(f"Processing team {team_id} ({team_data['name']}) with {len(team_data['fixtures'])} fixtures")
            
            # Process fixtures for this team (sample every 3rd fixture to get more data)
            fixtures_to_process = team_data['fixtures'][::3]  # Every 3rd fixture
            
            for fixture_data in fixtures_to_process:
                try:
                    # Calculate performance state for this fixture date
                    perf_state = calculate_team_performance_state(
                        team_id, 
                        fixture_data['match_date'], 
                        session
                    )
                    
                    # Check if record already exists (unless force mode)
                    if not force:
                        check_query = text("""
                            SELECT id FROM team_performance_states 
                            WHERE team_id = :team_id 
                            AND fixture_id = :fixture_id
                        """)
                        
                        existing = session.execute(check_query, {
                            'team_id': team_id,
                            'fixture_id': fixture_data['fixture_id']
                        }).fetchone()
                        
                        if existing:
                            continue  # Skip if already exists
                    
                    # If force mode, delete existing record first
                    if force:
                        delete_query = text("""
                            DELETE FROM team_performance_states 
                            WHERE team_id = :team_id 
                            AND fixture_id = :fixture_id
                        """)
                        session.execute(delete_query, {
                            'team_id': team_id,
                            'fixture_id': fixture_data['fixture_id']
                        })
                    
                    # Insert new performance state record
                    insert_query = text("""
                        INSERT INTO team_performance_states (
                            team_id, fixture_id, league_id, state_date, home_away_context,
                            state_score, performance_state, goals_scored, goals_conceded,
                            goal_difference, win_rate, points_per_game, form_streak,
                            matches_considered, created_at, updated_at
                        ) VALUES (
                            :team_id, :fixture_id, :league_id, :state_date, :home_away_context,
                            :state_score, :performance_state, :goals_scored, :goals_conceded,
                            :goal_difference, :win_rate, :points_per_game, :form_streak,
                            :matches_considered, NOW(), NOW()
                        )
                    """)
                    
                    # Calculate additional metrics for the new schema
                    points = (perf_state['wins'] * 3) + perf_state['draws']
                    matches_played = perf_state['matches_analyzed']
                    points_per_game = points / matches_played if matches_played > 0 else 0.0
                    win_rate = perf_state['wins'] / matches_played if matches_played > 0 else 0.0
                    goal_difference = perf_state['goals_for'] - perf_state['goals_against']
                    
                    session.execute(insert_query, {
                        'team_id': team_id,
                        'fixture_id': fixture_data['fixture_id'],
                        'league_id': fixture_data['league_id'],
                        'state_date': fixture_data['match_date'],
                        'home_away_context': fixture_data['venue'],
                        'state_score': perf_state['momentum_score'],
                        'performance_state': perf_state['performance_state'],
                        'goals_scored': perf_state['goals_for'],
                        'goals_conceded': perf_state['goals_against'],
                        'goal_difference': goal_difference,
                        'win_rate': round(win_rate, 3),
                        'points_per_game': round(points_per_game, 3),
                        'form_streak': perf_state['form_trend'],
                        'matches_considered': perf_state['matches_analyzed']
                    })
                    
                    total_inserted += 1
                    
                    if total_inserted % 100 == 0:
                        session.commit()
                        logger.info(f"Inserted {total_inserted} performance state records")
                
                except Exception as e:
                    logger.warning(f"Error processing fixture {fixture_data['fixture_id']} for team {team_id}: {e}")
                    session.rollback()  # Rollback failed transaction
                    total_errors += 1
                    continue
        
        except Exception as e:
            logger.error(f"Error processing team {team_id}: {e}")
            session.rollback()  # Rollback failed transaction
            total_errors += 1
            continue
    
    # Final commit
    session.commit()
    
    logger.info("=" * 50)
    logger.info("Team performance states population completed")
    logger.info(f"Total records inserted: {total_inserted}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("=" * 50)


def main():
    """Main function to populate team performance states."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate team performance states')
    parser.add_argument('--limit', type=int, default=200, help='Limit number of teams to process (default: 200)')
    parser.add_argument('--force', action='store_true', help='Force regeneration of existing records')
    args = parser.parse_args()
    
    try:
        # Load configuration
        load_config()
        logger.info("Configuration loaded")
        
        # Use database session
        with get_db_session() as session:
            logger.info("Database connection established")
            
            # Check current state count
            count_query = text("SELECT COUNT(*) FROM team_performance_states")
            current_count = session.execute(count_query).scalar()
            logger.info(f"Current team performance states count: {current_count}")
            
            # Populate team performance states
            populate_team_performance_states(session, limit_teams=args.limit, force=args.force)
            
            # Check final count
            final_count = session.execute(count_query).scalar()
            logger.info(f"Final team performance states count: {final_count}")
            logger.info(f"Added {final_count - current_count} new records")
    
    except Exception as e:
        logger.error(f"Population failed: {e}")
        raise


if __name__ == '__main__':
    main()