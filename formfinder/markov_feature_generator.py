"""Markov Chain Feature Generator

This module generates advanced features from Markov chain analysis for use in
machine learning models. It computes momentum, entropy, volatility, and other
Markov-based features.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import (
    get_db_session, TeamPerformanceState, MarkovTransitionMatrix,
    MarkovFeatures, Team, League
)
from .markov_state_classifier import MarkovStateClassifier
from .markov_transition_calculator import MarkovTransitionCalculator
from .logger import get_logger

logger = get_logger(__name__)

class MarkovFeatureGenerator:
    """Generates advanced Markov chain features for machine learning models."""
    
    # Performance states
    STATES = ['excellent', 'good', 'average', 'poor', 'terrible']
    
    # State value mapping for numerical calculations
    STATE_VALUES = {
        'excellent': 1.0,
        'good': 0.75,
        'average': 0.5,
        'poor': 0.25,
        'terrible': 0.0
    }
    
    def __init__(self, db_session: Session = None, lookback_window: int = 10):
        """
        Initialize the feature generator.
        
        Args:
            db_session: Database session for queries (optional, will use get_db_session if not provided)
            lookback_window: Number of recent states to consider for features
        """
        self.db_session = db_session
        self.lookback_window = lookback_window
        self.state_classifier = MarkovStateClassifier()
        self.transition_calculator = MarkovTransitionCalculator()
    
    def _get_session(self):
        """Get database session - use provided session or create new one."""
        if self.db_session:
            return self.db_session
        else:
            return get_db_session()
    
    def calculate_momentum_score(
        self, 
        team_id: int, 
        league_id: int, 
        reference_date: datetime,
        context: str = 'overall'
    ) -> float:
        """
        Calculate momentum score based on recent state transitions.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            reference_date: Reference date for calculation
            context: Context for analysis
            
        Returns:
            Momentum score (-1 to 1, where 1 is strong positive momentum)
        """
        with get_db_session() as session:
            try:
                # Get recent states
                states_query = """
                    SELECT performance_state, state_date, state_score
                    FROM team_performance_states
                    WHERE team_id = :team_id
                      AND league_id = :league_id
                      AND home_away_context = :context
                      AND state_date <= :reference_date
                    ORDER BY state_date DESC
                    LIMIT :limit
                """
                
                states = session.execute(
                    text(states_query),
                    {
                        'team_id': team_id,
                        'league_id': league_id,
                        'context': context,
                        'reference_date': reference_date,
                        'limit': self.lookback_window
                    }
                ).fetchall()
                
                if len(states) < 2:
                    return 0.0
                
                # Calculate momentum using state values and trends
                state_values = [self.STATE_VALUES[state[0]] for state in states]
                state_scores = [state[2] for state in states]
                
                # Calculate trend using linear regression slope
                x = np.arange(len(state_values))
                
                # Weight recent states more heavily
                weights = np.exp(-0.1 * x)  # Exponential decay
                
                # Calculate weighted trend for state values
                if len(state_values) > 1:
                    state_trend = np.polyfit(x, state_values, 1, w=weights)[0]
                else:
                    state_trend = 0.0
                
                # Calculate weighted trend for state scores
                if len(state_scores) > 1:
                    score_trend = np.polyfit(x, state_scores, 1, w=weights)[0]
                else:
                    score_trend = 0.0
                
                # Combine trends with recent performance
                recent_performance = np.mean(state_values[:3]) if len(state_values) >= 3 else np.mean(state_values)
                
                # Calculate momentum score
                momentum = (
                    0.4 * state_trend +
                    0.4 * score_trend +
                    0.2 * (recent_performance - 0.5)  # Deviation from average
                )
                
                # Normalize to [-1, 1] range
                momentum = np.clip(momentum * 2, -1.0, 1.0)
                
                return float(momentum)
                
            except Exception as e:
                logger.error(f"Error calculating momentum score: {e}")
                return 0.0
    
    def calculate_state_stability(
        self, 
        team_id: int, 
        league_id: int, 
        reference_date: datetime,
        context: str = 'overall'
    ) -> float:
        """
        Calculate state stability (consistency of performance).
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            reference_date: Reference date for calculation
            context: Context for analysis
            
        Returns:
            Stability score (0 to 1, where 1 is very stable)
        """
        with get_db_session() as session:
            try:
                # Get recent states
                states_query = """
                    SELECT performance_state, state_score
                    FROM team_performance_states
                    WHERE team_id = :team_id
                      AND league_id = :league_id
                      AND home_away_context = :context
                      AND state_date <= :reference_date
                    ORDER BY state_date DESC
                    LIMIT :limit
                """
                
                states = session.execute(
                    text(states_query),
                    {
                        'team_id': team_id,
                        'league_id': league_id,
                        'context': context,
                        'reference_date': reference_date,
                        'limit': self.lookback_window
                    }
                ).fetchall()
                
                if len(states) < 2:
                    return 0.5  # Neutral stability
                
                # Calculate stability based on state consistency
                state_values = [self.STATE_VALUES[state[0]] for state in states]
                state_scores = [state[1] for state in states]
                
                # Calculate coefficient of variation (lower = more stable)
                state_cv = np.std(state_values) / (np.mean(state_values) + 1e-6)
                score_cv = np.std(state_scores) / (np.mean(state_scores) + 1e-6)
                
                # Calculate state change frequency
                state_changes = sum(1 for i in range(len(states) - 1) 
                                  if states[i][0] != states[i + 1][0])
                change_rate = state_changes / (len(states) - 1) if len(states) > 1 else 0
                
                # Combine metrics (lower values = higher stability)
                combined_cv = (state_cv + score_cv) / 2
                
                # Convert to stability score (0-1, higher = more stable)
                stability = 1.0 / (1.0 + combined_cv + change_rate)
                
                return float(np.clip(stability, 0.0, 1.0))
                
            except Exception as e:
                logger.error(f"Error calculating state stability: {e}")
                return 0.5
    
    def calculate_transition_entropy(
        self, 
        team_id: int, 
        league_id: int, 
        context: str = 'overall'
    ) -> float:
        """
        Calculate transition entropy (unpredictability of state changes).
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            context: Context for analysis
            
        Returns:
            Entropy score (0 to log2(5), higher = more unpredictable)
        """
        try:
            # Get transition matrix
            transition_matrix = self.transition_calculator.get_transition_matrix(
                team_id, league_id, context
            )
            
            if not transition_matrix:
                return np.log2(len(self.STATES)) / 2  # Medium entropy
            
            # Calculate average entropy across all states
            total_entropy = 0.0
            valid_states = 0
            
            for from_state in self.STATES:
                if from_state in transition_matrix:
                    entropy = self.transition_calculator.get_transition_entropy(
                        transition_matrix, from_state
                    )
                    total_entropy += entropy
                    valid_states += 1
            
            if valid_states == 0:
                return np.log2(len(self.STATES)) / 2
            
            average_entropy = total_entropy / valid_states
            return float(average_entropy)
            
        except Exception as e:
            logger.error(f"Error calculating transition entropy: {e}")
            return np.log2(len(self.STATES)) / 2
    
    def calculate_performance_volatility(
        self, 
        team_id: int, 
        league_id: int, 
        reference_date: datetime,
        context: str = 'overall'
    ) -> float:
        """
        Calculate performance volatility (variance in performance levels).
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            reference_date: Reference date for calculation
            context: Context for analysis
            
        Returns:
            Volatility score (0 to 1, higher = more volatile)
        """
        with get_db_session() as session:
            try:
                # Get recent state scores
                states_query = """
                    SELECT state_score
                    FROM team_performance_states
                    WHERE team_id = :team_id
                      AND league_id = :league_id
                      AND home_away_context = :context
                      AND state_date <= :reference_date
                    ORDER BY state_date DESC
                    LIMIT :limit
                """
                
                states = session.execute(
                    text(states_query),
                    {
                        'team_id': team_id,
                        'league_id': league_id,
                        'context': context,
                        'reference_date': reference_date,
                        'limit': self.lookback_window
                    }
                ).fetchall()
                
                if len(states) < 2:
                    return 0.5  # Medium volatility
                
                scores = [state[0] for state in states]
                
                # Calculate volatility as normalized standard deviation
                volatility = np.std(scores) / (np.mean(scores) + 1e-6)
                
                # Normalize to 0-1 range (assuming max reasonable volatility is 2.0)
                normalized_volatility = min(volatility / 2.0, 1.0)
                
                return float(normalized_volatility)
                
            except Exception as e:
                logger.error(f"Error calculating performance volatility: {e}")
                return 0.5
    
    def get_current_state_info(
        self, 
        team_id: int, 
        league_id: int, 
        reference_date: datetime,
        context: str = 'overall'
    ) -> Dict[str, Union[str, int, float]]:
        """
        Get current state information for a team.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            reference_date: Reference date
            context: Context for analysis
            
        Returns:
            Dictionary with current state information
        """
        with get_db_session() as session:
            try:
                # Get most recent state
                state_query = """
                    SELECT performance_state, state_score, state_date
                    FROM team_performance_states
                    WHERE team_id = :team_id
                      AND league_id = :league_id
                      AND home_away_context = :context
                      AND state_date <= :reference_date
                    ORDER BY state_date DESC
                    LIMIT 1
                """
                
                current_state = session.execute(
                    text(state_query),
                    {
                        'team_id': team_id,
                        'league_id': league_id,
                        'context': context,
                        'reference_date': reference_date
                    }
                ).fetchone()
                
                # If no data before reference date, get the most recent available state
                if not current_state:
                    fallback_query = """
                        SELECT performance_state, state_score, state_date
                        FROM team_performance_states
                        WHERE team_id = :team_id
                          AND league_id = :league_id
                          AND home_away_context = :context
                        ORDER BY state_date DESC
                        LIMIT 1
                    """
                    
                    current_state = session.execute(
                        text(fallback_query),
                        {
                            'team_id': team_id,
                            'league_id': league_id,
                            'context': context
                        }
                    ).fetchone()
                
                if not current_state:
                    # Calculate current state using state classifier
                    try:
                        metrics = self.state_classifier._calculate_performance_score_with_session(
                            session, team_id, league_id, reference_date, context
                        )
                        
                        if metrics['matches_analyzed'] >= self.state_classifier.min_matches:
                            calculated_state = self.state_classifier.classify_state(metrics['performance_score'])
                            return {
                                'current_state': calculated_state,
                                'state_duration': 1,
                                'state_confidence': metrics['performance_score']
                            }
                    except Exception as e:
                        logger.warning(f"Error calculating current state: {e}")
                    
                    return {
                        'current_state': 'average',
                        'state_duration': 0,
                        'state_confidence': 0.5
                    }
                
                # Calculate state duration
                duration_query = """
                    SELECT COUNT(*)
                    FROM team_performance_states
                    WHERE team_id = :team_id
                      AND league_id = :league_id
                      AND home_away_context = :context
                      AND performance_state = :current_state
                      AND state_date <= :reference_date
                      AND state_date > (
                          SELECT COALESCE(MAX(state_date), '1900-01-01')
                          FROM team_performance_states
                          WHERE team_id = :team_id
                            AND league_id = :league_id
                            AND home_away_context = :context
                            AND performance_state != :current_state
                            AND state_date < :reference_date
                      )
                """
                
                duration_result = session.execute(
                    text(duration_query),
                    {
                        'team_id': team_id,
                        'league_id': league_id,
                        'context': context,
                        'current_state': current_state[0],
                        'reference_date': reference_date
                    }
                ).fetchone()
                
                state_duration = duration_result[0] if duration_result else 1
                
                # Calculate state confidence based on consistency
                confidence = min(current_state[1] + (state_duration - 1) * 0.1, 1.0)
                
                return {
                    'current_state': current_state[0],
                    'state_duration': state_duration,
                    'state_confidence': float(confidence)
                }
                
            except Exception as e:
                logger.error(f"Error getting current state info: {e}")
                return {
                    'current_state': 'average',
                    'state_duration': 0,
                    'state_confidence': 0.5
                }
    
    def predict_next_state_info(
        self, 
        team_id: int, 
        league_id: int, 
        current_state: str,
        context: str = 'overall'
    ) -> Dict[str, Union[str, float]]:
        """
        Predict next state using transition matrix.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            current_state: Current performance state
            context: Context for analysis
            
        Returns:
            Dictionary with next state prediction
        """
        try:
            # Get transition matrix
            transition_matrix = self.transition_calculator.get_transition_matrix(
                team_id, league_id, context
            )
            
            if not transition_matrix or current_state not in transition_matrix:
                return {
                    'expected_next_state': current_state,
                    'next_state_probability': 0.5
                }
            
            # Predict next state
            next_state, probability = self.transition_calculator.predict_next_state(
                current_state, transition_matrix
            )
            
            return {
                'expected_next_state': next_state,
                'next_state_probability': float(probability)
            }
            
        except Exception as e:
            logger.error(f"Error predicting next state: {e}")
            return {
                'expected_next_state': current_state,
                'next_state_probability': 0.5
            }
    
    def calculate_advanced_features(
        self, 
        team_id: int, 
        league_id: int, 
        context: str = 'overall'
    ) -> Dict[str, float]:
        """
        Calculate advanced Markov features.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            context: Context for analysis
            
        Returns:
            Dictionary of advanced features
        """
        try:
            # Get transition matrix
            transition_matrix = self.transition_calculator.get_transition_matrix(
                team_id, league_id, context
            )
            
            if not transition_matrix:
                return {
                    'mean_return_time': 5.0,
                    'steady_state_probability': 0.2,
                    'absorption_probability': 0.0
                }
            
            # Calculate steady state probabilities
            steady_state_probs = self.transition_calculator.calculate_steady_state_probabilities(
                transition_matrix
            )
            
            # Get current state
            current_state_info = self.get_current_state_info(
                team_id, league_id, datetime.now(), context
            )
            current_state = current_state_info['current_state']
            
            # Calculate mean return time to current state
            mean_return_time = self.transition_calculator.calculate_mean_return_time(
                transition_matrix, current_state
            )
            
            # Get steady state probability for current state
            steady_state_prob = steady_state_probs.get(current_state, 0.2)
            
            # Calculate absorption probability (probability of staying in excellent state)
            absorption_prob = 0.0
            if 'excellent' in transition_matrix and 'excellent' in transition_matrix['excellent']:
                absorption_prob = transition_matrix['excellent']['excellent']
            
            return {
                'mean_return_time': float(mean_return_time) if mean_return_time != float('inf') else 10.0,
                'steady_state_probability': float(steady_state_prob),
                'absorption_probability': float(absorption_prob)
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced features: {e}")
            return {
                'mean_return_time': 5.0,
                'steady_state_probability': 0.2,
                'absorption_probability': 0.0
            }
    
    def generate_team_features(
        self, 
        team_id: int, 
        league_id: int, 
        reference_date: datetime = None,
        context: str = 'overall'
    ) -> Dict[str, Union[str, int, float]]:
        """
        Generate comprehensive Markov features for a team.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            reference_date: Reference date (defaults to now)
            context: Context for analysis
            
        Returns:
            Dictionary of all Markov features
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        try:
            # Get current state information
            current_state_info = self.get_current_state_info(
                team_id, league_id, reference_date, context
            )
            
            # Calculate momentum and stability features
            momentum = self.calculate_momentum_score(
                team_id, league_id, reference_date, context
            )
            
            stability = self.calculate_state_stability(
                team_id, league_id, reference_date, context
            )
            
            entropy = self.calculate_transition_entropy(
                team_id, league_id, context
            )
            
            volatility = self.calculate_performance_volatility(
                team_id, league_id, reference_date, context
            )
            
            # Predict next state
            next_state_info = self.predict_next_state_info(
                team_id, league_id, current_state_info['current_state'], context
            )
            
            # Calculate advanced features
            advanced_features = self.calculate_advanced_features(
                team_id, league_id, context
            )
            
            # Determine trend direction
            trend_direction = 'stable'
            if momentum > 0.2:
                trend_direction = 'improving'
            elif momentum < -0.2:
                trend_direction = 'declining'
            
            # Combine all features
            features = {
                # Current state information
                'current_state': current_state_info['current_state'],
                'state_duration': current_state_info['state_duration'],
                'state_confidence': current_state_info['state_confidence'],
                
                # Momentum and trend features
                'momentum_score': momentum,
                'trend_direction': trend_direction,
                
                # Stability and volatility features
                'state_stability': stability,
                'transition_entropy': entropy,
                'performance_volatility': volatility,
                
                # Prediction features
                'expected_next_state': next_state_info['expected_next_state'],
                'next_state_probability': next_state_info['next_state_probability'],
                
                # Advanced features
                'mean_return_time': advanced_features['mean_return_time'],
                'steady_state_probability': advanced_features['steady_state_probability'],
                'absorption_probability': advanced_features['absorption_probability'],
                
                # Metadata
                'feature_date': reference_date,
                'home_away_context': context,
                'lookback_window': self.lookback_window
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating team features: {e}")
            return self._default_features(reference_date, context)
    
    def _default_features(self, reference_date: datetime, context: str) -> Dict[str, Union[str, int, float]]:
        """
        Return default feature values when calculation fails.
        
        Args:
            reference_date: Reference date
            context: Context
            
        Returns:
            Dictionary with default feature values
        """
        return {
            'current_state': 'average',
            'state_duration': 1,
            'state_confidence': 0.5,
            'momentum_score': 0.0,
            'trend_direction': 'stable',
            'state_stability': 0.5,
            'transition_entropy': np.log2(len(self.STATES)) / 2,
            'performance_volatility': 0.5,
            'expected_next_state': 'average',
            'next_state_probability': 0.2,
            'mean_return_time': 5.0,
            'steady_state_probability': 0.2,
            'absorption_probability': 0.0,
            'feature_date': reference_date,
            'home_away_context': context,
            'lookback_window': self.lookback_window
        }
    
    def store_team_features(
        self, 
        team_id: int, 
        league_id: int, 
        features: Dict[str, Union[str, int, float]],
        fixture_id: int = None
    ) -> MarkovFeatures:
        """
        Store generated features in the database.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            features: Generated features dictionary
            fixture_id: Optional fixture identifier
            
        Returns:
            Created MarkovFeatures object
        """
        with get_db_session() as session:
            try:
                # Check if features already exist
                existing_features = session.query(MarkovFeatures).filter_by(
                    team_id=team_id,
                    league_id=league_id,
                    feature_date=features['feature_date'],
                    home_away_context=features['home_away_context']
                ).first()
                
                if existing_features:
                    # Update existing features
                    for key, value in features.items():
                        if hasattr(existing_features, key) and key not in ['feature_date', 'home_away_context']:
                            setattr(existing_features, key, value)
                    
                    existing_features.updated_at = datetime.now()
                    feature_record = existing_features
                else:
                    # Create new features record
                    feature_record = MarkovFeatures(
                        team_id=team_id,
                        league_id=league_id,
                        fixture_id=fixture_id,
                        feature_date=features['feature_date'],
                        current_state=features['current_state'],
                        state_duration=features['state_duration'],
                        momentum_score=features['momentum_score'],
                        trend_direction=features['trend_direction'],
                        state_stability=features['state_stability'],
                        transition_entropy=features['transition_entropy'],
                        performance_volatility=features['performance_volatility'],
                        expected_next_state=features['expected_next_state'],
                        next_state_probability=features['next_state_probability'],
                        state_confidence=features['state_confidence'],
                        mean_return_time=features['mean_return_time'],
                        steady_state_probability=features['steady_state_probability'],
                        absorption_probability=features['absorption_probability'],
                        home_away_context=features['home_away_context'],
                        lookback_window=features['lookback_window']
                    )
                    
                    session.add(feature_record)
                
                session.commit()
                logger.info(f"Stored Markov features for team {team_id} in league {league_id}")
                
                return feature_record
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error storing team features: {e}")
                raise
    
    def process_team_features(
        self, 
        team_id: int, 
        league_id: int, 
        start_date: datetime = None,
        end_date: datetime = None,
        contexts: List[str] = None
    ) -> Dict[str, List[MarkovFeatures]]:
        """
        Process and store Markov features for all contexts and dates.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            start_date: Start date for processing
            end_date: End date for processing
            contexts: List of contexts to process
            
        Returns:
            Dictionary mapping contexts to created feature records
        """
        if contexts is None:
            contexts = ['overall', 'home', 'away']
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        results = {}
        
        for context in contexts:
            try:
                # Get fixture dates in the range
                with get_db_session() as session:
                    fixture_dates_query = """
                        SELECT DISTINCT f.match_date, f.id
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
                
                created_features = []
                
                for date_row in fixture_dates:
                    fixture_date, fixture_id = date_row
                    
                    # Generate features for this date
                    features = self.generate_team_features(
                        team_id, league_id, fixture_date, context
                    )
                    
                    # Store features
                    feature_record = self.store_team_features(
                        team_id, league_id, features, fixture_id
                    )
                    
                    created_features.append(feature_record)
                
                results[context] = created_features
                
                logger.info(
                    f"Processed Markov features for team {team_id}, context {context}: "
                    f"{len(created_features)} records created/updated"
                )
                
            except Exception as e:
                logger.error(
                    f"Error processing features for team {team_id}, context {context}: {e}"
                )
                results[context] = []
        
        return results
    
    def generate_features(
        self,
        home_team_id: int,
        away_team_id: int,
        match_date: datetime,
        league_id: int,
        context: str = 'overall'
    ) -> Dict[str, Union[str, int, float]]:
        """
        Generate Markov features for both home and away teams for a match.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            match_date: Match date for reference
            league_id: League identifier
            context: Context for analysis
            
        Returns:
            Dictionary of combined Markov features with home_ and away_ prefixes
        """
        combined_features = {}
        
        try:
            # Generate features for home team (use 'home' context)
            home_features = self.generate_team_features(
                team_id=home_team_id,
                league_id=league_id,
                reference_date=match_date,
                context='home'
            )
            
            # Add home team features with prefix
            for key, value in home_features.items():
                if key not in ['feature_date', 'home_away_context', 'lookback_window']:
                    combined_features[f'home_{key}'] = value
                    
            # Generate features for away team (use 'away' context)
            away_features = self.generate_team_features(
                team_id=away_team_id,
                league_id=league_id,
                reference_date=match_date,
                context='away'
            )
            
            # Add away team features with prefix
            for key, value in away_features.items():
                if key not in ['feature_date', 'home_away_context', 'lookback_window']:
                    combined_features[f'away_{key}'] = value
                    
            # Calculate relative features
            if 'home_momentum_score' in combined_features and 'away_momentum_score' in combined_features:
                combined_features['momentum_diff'] = combined_features['home_momentum_score'] - combined_features['away_momentum_score']
                
            if 'home_performance_volatility' in combined_features and 'away_performance_volatility' in combined_features:
                combined_features['volatility_diff'] = combined_features['home_performance_volatility'] - combined_features['away_performance_volatility']
                
            if 'home_transition_entropy' in combined_features and 'away_transition_entropy' in combined_features:
                combined_features['entropy_diff'] = combined_features['home_transition_entropy'] - combined_features['away_transition_entropy']
                
            # Add match-level prediction features
            home_state = home_features.get('current_state', 'average')
            away_state = away_features.get('current_state', 'average')
            
            # Calculate match prediction confidence based on state certainty
            home_confidence = home_features.get('state_confidence', 0.5)
            away_confidence = away_features.get('state_confidence', 0.5)
            combined_features['match_prediction_confidence'] = (home_confidence + away_confidence) / 2
            
            # Generate outcome probabilities based on team states
            home_strength = self.STATE_VALUES.get(home_state, 0.5)
            away_strength = self.STATE_VALUES.get(away_state, 0.5)
            
            # Simple outcome probability calculation
            total_strength = home_strength + away_strength + 0.1  # Add small constant to avoid division by zero
            home_win_prob = home_strength / total_strength * 0.6 + 0.2  # Home advantage
            away_win_prob = away_strength / total_strength * 0.6 + 0.1
            draw_prob = 1.0 - home_win_prob - away_win_prob
            
            combined_features['outcome_probabilities'] = {
                'home_win': round(home_win_prob, 3),
                'draw': round(max(draw_prob, 0.1), 3),  # Ensure minimum draw probability
                'away_win': round(away_win_prob, 3)
            }
                
            logger.debug(f"Generated {len(combined_features)} combined Markov features for match {home_team_id} vs {away_team_id}")
            
        except Exception as e:
            logger.error(f"Error generating combined Markov features: {e}")
            
        return combined_features