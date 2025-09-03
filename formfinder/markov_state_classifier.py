"""Markov Chain State Classification System

This module implements the state classification system for Markov chain analysis.
It classifies team performance into discrete states based on recent form and performance metrics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import (
    get_db_session, TeamPerformanceState, Team, League, Fixture,
    MatchEvent, Standing
)
from .logger import get_logger

logger = get_logger(__name__)

class MarkovStateClassifier:
    """Classifies team performance into discrete states for Markov chain analysis."""
    
    # Performance state definitions
    STATES = ['excellent', 'good', 'average', 'poor', 'terrible']
    
    # Default thresholds for state classification
    DEFAULT_THRESHOLDS = {
        'excellent': {'min_score': 0.8, 'color': '#2E8B57'},
        'good': {'min_score': 0.6, 'color': '#32CD32'},
        'average': {'min_score': 0.4, 'color': '#FFD700'},
        'poor': {'min_score': 0.2, 'color': '#FF6347'},
        'terrible': {'min_score': 0.0, 'color': '#DC143C'}
    }
    
    def __init__(self, lookback_window: int = 5, min_matches: int = 3):
        """
        Initialize the state classifier.
        
        Args:
            lookback_window: Number of recent matches to consider for state calculation
            min_matches: Minimum number of matches required for state classification
        """
        self.lookback_window = lookback_window
        self.min_matches = min_matches
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        
    def calculate_performance_score(
        self, 
        team_id: int, 
        league_id: int, 
        reference_date: datetime,
        context: str = 'overall'
    ) -> Dict[str, Union[float, int, str]]:
        """
        Calculate comprehensive performance score for a team.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            reference_date: Date to calculate performance from
            context: 'home', 'away', or 'overall'
            
        Returns:
            Dictionary containing performance metrics and calculated score
        """
        with get_db_session() as session:
            return self._calculate_performance_score_with_session(
                session, team_id, league_id, reference_date, context
            )
    
    def _calculate_performance_score_with_session(
        self, 
        session,
        team_id: int, 
        league_id: int, 
        reference_date: datetime,
        context: str = 'overall'
    ) -> Dict[str, Union[float, int, str]]:
        """
        Calculate comprehensive performance score for a team using provided session.
        
        Args:
            session: Database session to use
            team_id: Team identifier
            league_id: League identifier
            reference_date: Date to calculate performance from
            context: 'home', 'away', or 'overall'
            
        Returns:
            Dictionary containing performance metrics and calculated score
        """
        try:
            # Get recent fixtures for the team
            fixtures_query = """
                SELECT f.id, f.match_date, f.home_team_id, f.away_team_id,
                       f.home_score, f.away_score, f.status
                FROM fixtures f
                WHERE (f.home_team_id = :team_id OR f.away_team_id = :team_id)
                  AND f.league_id = :league_id
                  AND f.match_date <= :reference_date
                  AND f.status = 'finished'
                  AND f.home_score IS NOT NULL
                  AND f.away_score IS NOT NULL
            """
            
            # Add context filter if specified
            if context == 'home':
                fixtures_query += " AND f.home_team_id = :team_id"
            elif context == 'away':
                fixtures_query += " AND f.away_team_id = :team_id"
                
            fixtures_query += " ORDER BY f.match_date DESC LIMIT :limit"
            
            fixtures = session.execute(
                text(fixtures_query),
                {
                    'team_id': team_id,
                    'league_id': league_id,
                    'reference_date': reference_date,
                    'limit': self.lookback_window
                }
            ).fetchall()
            
            if len(fixtures) < self.min_matches:
                logger.warning(
                    f"Insufficient matches for team {team_id} in league {league_id}. "
                    f"Found {len(fixtures)}, need {self.min_matches}"
                )
                return self._default_performance_metrics()
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(fixtures, team_id)
            
            # Calculate composite performance score
            performance_score = self._calculate_composite_score(metrics)
            
            # Add metadata
            metrics.update({
                'performance_score': performance_score,
                'matches_analyzed': len(fixtures),
                'context': context,
                'calculation_date': reference_date
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return self._default_performance_metrics()
    
    def _calculate_metrics(self, fixtures: List, team_id: int) -> Dict[str, float]:
        """
        Calculate detailed performance metrics from fixtures.
        
        Args:
            fixtures: List of fixture records
            team_id: Team identifier
            
        Returns:
            Dictionary of calculated metrics
        """
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        points = 0
        form_streak = ""
        recent_results = []
        
        for fixture in fixtures:
            is_home = fixture.home_team_id == team_id
            team_goals = fixture.home_score if is_home else fixture.away_score
            opponent_goals = fixture.away_score if is_home else fixture.home_score
            
            goals_scored += team_goals
            goals_conceded += opponent_goals
            
            # Determine result
            if team_goals > opponent_goals:
                wins += 1
                points += 3
                recent_results.append('W')
            elif team_goals == opponent_goals:
                draws += 1
                points += 1
                recent_results.append('D')
            else:
                losses += 1
                recent_results.append('L')
        
        total_matches = len(fixtures)
        form_streak = ''.join(recent_results[:5])  # Last 5 results
        
        # Calculate derived metrics
        win_rate = wins / total_matches if total_matches > 0 else 0
        points_per_game = points / total_matches if total_matches > 0 else 0
        goal_difference = goals_scored - goals_conceded
        goals_per_game = goals_scored / total_matches if total_matches > 0 else 0
        goals_conceded_per_game = goals_conceded / total_matches if total_matches > 0 else 0
        
        # Calculate momentum (recent form vs overall form)
        recent_form = self._calculate_recent_momentum(recent_results)
        
        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goal_difference,
            'points': points,
            'win_rate': win_rate,
            'points_per_game': points_per_game,
            'goals_per_game': goals_per_game,
            'goals_conceded_per_game': goals_conceded_per_game,
            'form_streak': form_streak,
            'recent_momentum': recent_form,
            'total_matches': total_matches
        }
    
    def _calculate_recent_momentum(self, results: List[str]) -> float:
        """
        Calculate momentum based on recent results pattern.
        
        Args:
            results: List of recent results ('W', 'D', 'L')
            
        Returns:
            Momentum score between -1 and 1
        """
        if not results:
            return 0.0
        
        # Convert results to numeric values
        result_values = []
        for result in results:
            if result == 'W':
                result_values.append(1.0)
            elif result == 'D':
                result_values.append(0.0)
            else:  # 'L'
                result_values.append(-1.0)
        
        if not result_values:
            return 0.0
        
        # Weight recent results more heavily - extend weights if needed or limit results
        base_weights = [0.4, 0.3, 0.2, 0.1]
        if len(result_values) <= len(base_weights):
            weights = np.array(base_weights[:len(result_values)])
        else:
            # If we have more results than weights, take only the most recent ones
            result_values = result_values[:len(base_weights)]
            weights = np.array(base_weights)
        
        weights = weights / weights.sum()  # Normalize
        
        # Calculate weighted momentum
        try:
            momentum = np.average(result_values, weights=weights)
            return momentum
        except Exception as e:
            logger.error(f"Error in momentum calculation: {e}, result_values: {result_values}, weights: {weights}")
            return 0.0
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite performance score from individual metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Normalized performance score between 0 and 1
        """
        # Define weights for different metrics
        weights = {
            'win_rate': 0.25,
            'points_per_game': 0.20,
            'goal_difference': 0.15,
            'goals_per_game': 0.15,
            'goals_conceded_per_game': 0.10,  # Inverted
            'recent_momentum': 0.15
        }
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {}
        
        # Win rate is already normalized
        normalized_metrics['win_rate'] = metrics['win_rate']
        
        # Points per game (max 3)
        normalized_metrics['points_per_game'] = min(metrics['points_per_game'] / 3.0, 1.0)
        
        # Goal difference (normalize to -5 to +5 range)
        goal_diff_normalized = (metrics['goal_difference'] + 5) / 10.0
        normalized_metrics['goal_difference'] = max(0, min(goal_diff_normalized, 1.0))
        
        # Goals per game (normalize to 0-4 range)
        normalized_metrics['goals_per_game'] = min(metrics['goals_per_game'] / 4.0, 1.0)
        
        # Goals conceded per game (inverted, normalize to 0-4 range)
        goals_conceded_norm = 1.0 - min(metrics['goals_conceded_per_game'] / 4.0, 1.0)
        normalized_metrics['goals_conceded_per_game'] = goals_conceded_norm
        
        # Recent momentum (already -1 to 1, convert to 0-1)
        normalized_metrics['recent_momentum'] = (metrics['recent_momentum'] + 1) / 2.0
        
        # Calculate weighted composite score
        composite_score = sum(
            normalized_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return max(0.0, min(composite_score, 1.0))
    
    def classify_state(self, performance_score: float) -> str:
        """
        Classify performance score into discrete state.
        
        Args:
            performance_score: Normalized performance score (0-1)
            
        Returns:
            Performance state string
        """
        for state in ['excellent', 'good', 'average', 'poor']:
            if performance_score >= self.thresholds[state]['min_score']:
                return state
        return 'terrible'
    
    def _default_performance_metrics(self) -> Dict[str, Union[float, int, str]]:
        """
        Return default metrics when calculation fails.
        
        Returns:
            Dictionary with default values
        """
        return {
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'goal_difference': 0,
            'points': 0,
            'win_rate': 0.0,
            'points_per_game': 0.0,
            'goals_per_game': 0.0,
            'goals_conceded_per_game': 0.0,
            'form_streak': '',
            'recent_momentum': 0.0,
            'total_matches': 0,
            'performance_score': 0.4,  # Default to 'average'
            'matches_analyzed': 0,
            'context': 'overall',
            'calculation_date': datetime.now()
        }
    
    def process_team_states(
        self, 
        team_id: int, 
        league_id: int, 
        start_date: datetime, 
        end_date: datetime,
        contexts: List[str] = None
    ) -> List[TeamPerformanceState]:
        """
        Process and store team performance states for a date range.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            start_date: Start date for processing
            end_date: End date for processing
            contexts: List of contexts to process ('home', 'away', 'overall')
            
        Returns:
            List of created TeamPerformanceState objects
        """
        if contexts is None:
            contexts = ['overall', 'home', 'away']
        
        created_states = []
        
        with get_db_session() as session:
            try:
                # Get all fixture dates in the range
                fixture_dates_query = """
                    SELECT DISTINCT f.match_date
                    FROM fixtures f
                    WHERE (f.home_team_id = :team_id OR f.away_team_id = :team_id)
                      AND f.league_id = :league_id
                      AND f.match_date BETWEEN :start_date AND :end_date
                      AND f.status = 'finished'
                    ORDER BY f.match_date
                """
                
                fixture_dates = session.execute(
                    text(fixture_dates_query),
                    {
                        'team_id': team_id,
                        'league_id': league_id,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                ).fetchall()
                
                for date_row in fixture_dates:
                    fixture_date = date_row[0]
                    
                    for context in contexts:
                        # Check if state already exists
                        existing_state = session.query(TeamPerformanceState).filter_by(
                            team_id=team_id,
                            league_id=league_id,
                            state_date=fixture_date,
                            home_away_context=context
                        ).first()
                        
                        if existing_state:
                            logger.debug(
                                f"State already exists for team {team_id}, "
                                f"date {fixture_date}, context {context}"
                            )
                            continue
                        
                        # Calculate performance metrics
                        metrics = self._calculate_performance_score_with_session(
                            session, team_id, league_id, fixture_date, context
                        )
                        
                        if metrics['matches_analyzed'] < self.min_matches:
                            continue
                        
                        # Classify state
                        performance_state = self.classify_state(metrics['performance_score'])
                        
                        # Create state record
                        state_record = TeamPerformanceState(
                            team_id=team_id,
                            league_id=league_id,
                            state_date=fixture_date,
                            performance_state=performance_state,
                            state_score=metrics['performance_score'],
                            goals_scored=metrics['goals_scored'],
                            goals_conceded=metrics['goals_conceded'],
                            goal_difference=metrics['goal_difference'],
                            win_rate=metrics['win_rate'],
                            points_per_game=metrics['points_per_game'],
                            form_streak=metrics['form_streak'],
                            matches_considered=metrics['matches_analyzed'],
                            home_away_context=context
                        )
                        
                        session.add(state_record)
                        created_states.append(state_record)
                
                session.commit()
                logger.info(
                    f"Created {len(created_states)} performance states for team {team_id} "
                    f"in league {league_id}"
                )
                
                return created_states
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error processing team states: {e}")
                raise
    
    def get_current_state(
        self, 
        team_id: int, 
        league_id: int, 
        reference_date: datetime = None,
        context: str = 'overall'
    ) -> Optional[str]:
        """
        Get the current performance state for a team.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            reference_date: Reference date (defaults to now)
            context: Context for state lookup
            
        Returns:
            Current performance state or None if not found
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        with get_db_session() as session:
            try:
                state = session.query(TeamPerformanceState).filter_by(
                    team_id=team_id,
                    league_id=league_id,
                    home_away_context=context
                ).filter(
                    TeamPerformanceState.state_date <= reference_date
                ).order_by(
                    TeamPerformanceState.state_date.desc()
                ).first()
                
                return state.performance_state if state else None
                
            except Exception as e:
                logger.error(f"Error getting current state: {e}")
                return None
    
    def update_thresholds(self, new_thresholds: Dict[str, Dict[str, float]]) -> None:
        """
        Update state classification thresholds.
        
        Args:
            new_thresholds: Dictionary of new threshold values
        """
        for state, values in new_thresholds.items():
            if state in self.thresholds:
                self.thresholds[state].update(values)
        
        logger.info("Updated state classification thresholds")
    
    def get_state_distribution(
        self, 
        league_id: int = None, 
        context: str = 'overall',
        days_back: int = 30
    ) -> Dict[str, int]:
        """
        Get distribution of states across teams.
        
        Args:
            league_id: Optional league filter
            context: Context for state lookup
            days_back: Number of days to look back
            
        Returns:
            Dictionary with state counts
        """
        with get_db_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                
                query = session.query(TeamPerformanceState).filter(
                    TeamPerformanceState.state_date >= cutoff_date,
                    TeamPerformanceState.home_away_context == context
                )
                
                if league_id:
                    query = query.filter(TeamPerformanceState.league_id == league_id)
                
                states = query.all()
                
                distribution = {state: 0 for state in self.STATES}
                for state_record in states:
                    distribution[state_record.performance_state] += 1
                
                return distribution
                
            except Exception as e:
                logger.error(f"Error getting state distribution: {e}")
                return {state: 0 for state in self.STATES}