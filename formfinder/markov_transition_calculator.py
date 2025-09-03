"""Markov Chain Transition Matrix Calculator

This module calculates transition probabilities between performance states
and maintains transition matrices for Markov chain analysis.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import (
    get_db_session, TeamPerformanceState, MarkovTransitionMatrix,
    Team, League
)
from .logger import get_logger

logger = get_logger(__name__)

class MarkovTransitionCalculator:
    """Calculates and manages Markov chain transition matrices."""
    
    # Performance states
    STATES = ['excellent', 'good', 'average', 'poor', 'terrible']
    
    def __init__(self, smoothing_alpha: float = 1.0, min_transitions: int = 5):
        """
        Initialize the transition calculator.
        
        Args:
            smoothing_alpha: Laplace smoothing parameter
            min_transitions: Minimum transitions required for reliable calculation
        """
        self.smoothing_alpha = smoothing_alpha
        self.min_transitions = min_transitions
        
    def calculate_transition_matrix(
        self,
        team_id: int,
        league_id: int,
        start_date: datetime = None,
        end_date: datetime = None,
        context: str = 'overall'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate transition matrix for a specific team.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            start_date: Start date for analysis (defaults to 1 year ago)
            end_date: End date for analysis (defaults to now)
            context: Context for transitions ('home', 'away', 'overall')
            
        Returns:
            Nested dictionary representing transition matrix
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        with get_db_session() as session:
            try:
                # Get ordered performance states for the team
                states_query = """
                    SELECT performance_state, state_date
                    FROM team_performance_states
                    WHERE team_id = :team_id
                      AND league_id = :league_id
                      AND home_away_context = :context
                      AND state_date BETWEEN :start_date AND :end_date
                    ORDER BY state_date ASC
                """
                
                states = session.execute(
                    text(states_query),
                    {
                        'team_id': team_id,
                        'league_id': league_id,
                        'context': context,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                ).fetchall()
                
                if len(states) < 2:
                    logger.warning(
                        f"Insufficient state data for team {team_id}. "
                        f"Found {len(states)} states, need at least 2"
                    )
                    return self._default_transition_matrix()
                
                # Count transitions
                transition_counts = self._count_transitions(states)
                
                # Calculate probabilities with smoothing
                transition_matrix = self._calculate_probabilities(
                    transition_counts, self.smoothing_alpha
                )
                
                return transition_matrix
                
            except Exception as e:
                logger.error(f"Error calculating transition matrix: {e}")
                return self._default_transition_matrix()
    
    def _count_transitions(self, states: List[Tuple]) -> Dict[str, Dict[str, int]]:
        """
        Count transitions between consecutive states.
        
        Args:
            states: List of (state, date) tuples ordered by date
            
        Returns:
            Nested dictionary with transition counts
        """
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(states) - 1):
            from_state = states[i][0]
            to_state = states[i + 1][0]
            transition_counts[from_state][to_state] += 1
        
        # Convert to regular dict for consistency
        return {from_state: dict(to_states) for from_state, to_states in transition_counts.items()}
    
    def _calculate_probabilities(
        self, 
        transition_counts: Dict[str, Dict[str, int]], 
        alpha: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate transition probabilities with Laplace smoothing.
        
        Args:
            transition_counts: Raw transition counts
            alpha: Smoothing parameter
            
        Returns:
            Transition probability matrix
        """
        transition_matrix = {}
        
        for from_state in self.STATES:
            transition_matrix[from_state] = {}
            
            # Get counts for this state
            counts = transition_counts.get(from_state, {})
            
            # Calculate total transitions from this state (with smoothing)
            total_transitions = sum(counts.values()) + alpha * len(self.STATES)
            
            # Calculate probabilities for each target state
            for to_state in self.STATES:
                count = counts.get(to_state, 0)
                probability = (count + alpha) / total_transitions
                transition_matrix[from_state][to_state] = probability
        
        return transition_matrix
    
    def _default_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Return default uniform transition matrix.
        
        Returns:
            Uniform transition matrix
        """
        uniform_prob = 1.0 / len(self.STATES)
        return {
            from_state: {to_state: uniform_prob for to_state in self.STATES}
            for from_state in self.STATES
        }
    
    def store_transition_matrix(
        self,
        team_id: int,
        league_id: int,
        transition_matrix: Dict[str, Dict[str, float]],
        transition_counts: Dict[str, Dict[str, int]] = None,
        context: str = 'overall',
        data_window_start: datetime = None,
        data_window_end: datetime = None
    ) -> List[MarkovTransitionMatrix]:
        """
        Store transition matrix in the database.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            transition_matrix: Calculated transition probabilities
            transition_counts: Raw transition counts
            context: Context for transitions
            data_window_start: Start of data window
            data_window_end: End of data window
            
        Returns:
            List of created MarkovTransitionMatrix objects
        """
        with get_db_session() as session:
            created_records = []
            
            try:
                # Calculate total transitions for metadata
                total_transitions = 0
                if transition_counts:
                    total_transitions = sum(
                        sum(to_counts.values()) for to_counts in transition_counts.values()
                    )
                
                calculation_date = datetime.now()
                
                for from_state in self.STATES:
                    for to_state in self.STATES:
                        # Check if record already exists
                        existing_record = session.query(MarkovTransitionMatrix).filter_by(
                            team_id=team_id,
                            league_id=league_id,
                            from_state=from_state,
                            to_state=to_state,
                            home_away_context=context
                        ).first()
                        
                        # Get counts and probabilities
                        count = 0
                        if transition_counts and from_state in transition_counts:
                            count = transition_counts[from_state].get(to_state, 0)
                        
                        probability = transition_matrix[from_state][to_state]
                        smoothed_probability = probability  # Already smoothed in calculation
                        
                        if existing_record:
                            # Update existing record
                            existing_record.transition_count = count
                            existing_record.transition_probability = probability
                            existing_record.smoothed_probability = smoothed_probability
                            existing_record.calculation_date = calculation_date
                            existing_record.data_window_start = data_window_start
                            existing_record.data_window_end = data_window_end
                            existing_record.total_transitions = total_transitions
                            existing_record.smoothing_alpha = self.smoothing_alpha
                            existing_record.updated_at = calculation_date
                            
                            created_records.append(existing_record)
                        else:
                            # Create new record
                            record = MarkovTransitionMatrix(
                                team_id=team_id,
                                league_id=league_id,
                                from_state=from_state,
                                to_state=to_state,
                                transition_count=count,
                                transition_probability=probability,
                                smoothed_probability=smoothed_probability,
                                home_away_context=context,
                                calculation_date=calculation_date,
                                data_window_start=data_window_start,
                                data_window_end=data_window_end,
                                total_transitions=total_transitions,
                                smoothing_alpha=self.smoothing_alpha
                            )
                            
                            session.add(record)
                            created_records.append(record)
                
                session.commit()
                logger.info(
                    f"Stored transition matrix for team {team_id} in league {league_id} "
                    f"(context: {context})"
                )
                
                return created_records
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error storing transition matrix: {e}")
                raise
    
    def get_transition_matrix(
        self,
        team_id: int,
        league_id: int,
        context: str = 'overall'
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Retrieve stored transition matrix for a team.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            context: Context for transitions
            
        Returns:
            Transition matrix or None if not found
        """
        with get_db_session() as session:
            try:
                records = session.query(MarkovTransitionMatrix).filter_by(
                    team_id=team_id,
                    league_id=league_id,
                    home_away_context=context
                ).all()
                
                if not records:
                    return None
                
                # Reconstruct matrix
                matrix = defaultdict(dict)
                for record in records:
                    matrix[record.from_state][record.to_state] = record.smoothed_probability
                
                return dict(matrix)
                
            except Exception as e:
                logger.error(f"Error retrieving transition matrix: {e}")
                return None
    
    def calculate_steady_state_probabilities(
        self, 
        transition_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate steady-state probabilities for the transition matrix.
        
        Args:
            transition_matrix: Transition probability matrix
            
        Returns:
            Steady-state probability distribution
        """
        try:
            # Convert to numpy matrix
            matrix = np.array([
                [transition_matrix[from_state][to_state] for to_state in self.STATES]
                for from_state in self.STATES
            ])
            
            # Find eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
            
            # Find the eigenvector corresponding to eigenvalue 1
            steady_state_index = np.argmin(np.abs(eigenvalues - 1.0))
            steady_state_vector = np.real(eigenvectors[:, steady_state_index])
            
            # Normalize to get probabilities
            steady_state_vector = steady_state_vector / steady_state_vector.sum()
            
            # Ensure non-negative probabilities
            steady_state_vector = np.abs(steady_state_vector)
            steady_state_vector = steady_state_vector / steady_state_vector.sum()
            
            return dict(zip(self.STATES, steady_state_vector))
            
        except Exception as e:
            logger.error(f"Error calculating steady state probabilities: {e}")
            # Return uniform distribution as fallback
            uniform_prob = 1.0 / len(self.STATES)
            return {state: uniform_prob for state in self.STATES}
    
    def calculate_mean_return_time(
        self, 
        transition_matrix: Dict[str, Dict[str, float]], 
        target_state: str
    ) -> float:
        """
        Calculate mean return time to a specific state.
        
        Args:
            transition_matrix: Transition probability matrix
            target_state: State to calculate return time for
            
        Returns:
            Mean return time (in number of transitions)
        """
        try:
            steady_state_probs = self.calculate_steady_state_probabilities(transition_matrix)
            return 1.0 / steady_state_probs[target_state]
        except Exception as e:
            logger.error(f"Error calculating mean return time: {e}")
            return float('inf')
    
    def predict_next_state(
        self, 
        current_state: str, 
        transition_matrix: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float]:
        """
        Predict the most likely next state.
        
        Args:
            current_state: Current performance state
            transition_matrix: Transition probability matrix
            
        Returns:
            Tuple of (predicted_state, probability)
        """
        try:
            if current_state not in transition_matrix:
                # Return uniform prediction
                uniform_prob = 1.0 / len(self.STATES)
                return self.STATES[0], uniform_prob
            
            transitions = transition_matrix[current_state]
            predicted_state = max(transitions.keys(), key=lambda k: transitions[k])
            probability = transitions[predicted_state]
            
            return predicted_state, probability
            
        except Exception as e:
            logger.error(f"Error predicting next state: {e}")
            return current_state, 0.0
    
    def calculate_n_step_probabilities(
        self, 
        transition_matrix: Dict[str, Dict[str, float]], 
        n: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate n-step transition probabilities.
        
        Args:
            transition_matrix: Base transition matrix
            n: Number of steps
            
        Returns:
            N-step transition matrix
        """
        try:
            # Convert to numpy matrix
            matrix = np.array([
                [transition_matrix[from_state][to_state] for to_state in self.STATES]
                for from_state in self.STATES
            ])
            
            # Calculate matrix power
            n_step_matrix = np.linalg.matrix_power(matrix, n)
            
            # Convert back to dictionary format
            result = {}
            for i, from_state in enumerate(self.STATES):
                result[from_state] = {}
                for j, to_state in enumerate(self.STATES):
                    result[from_state][to_state] = n_step_matrix[i, j]
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating n-step probabilities: {e}")
            return transition_matrix
    
    def process_team_transitions(
        self,
        team_id: int,
        league_id: int,
        start_date: datetime = None,
        end_date: datetime = None,
        contexts: List[str] = None
    ) -> Dict[str, List[MarkovTransitionMatrix]]:
        """
        Process and store transition matrices for all contexts.
        
        Args:
            team_id: Team identifier
            league_id: League identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            contexts: List of contexts to process
            
        Returns:
            Dictionary mapping contexts to created records
        """
        if contexts is None:
            contexts = ['overall', 'home', 'away']
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        results = {}
        
        for context in contexts:
            try:
                # Calculate transition matrix
                transition_matrix = self.calculate_transition_matrix(
                    team_id, league_id, start_date, end_date, context
                )
                
                # Get raw counts for storage
                with get_db_session() as session:
                    states_query = """
                        SELECT performance_state, state_date
                        FROM team_performance_states
                        WHERE team_id = :team_id
                          AND league_id = :league_id
                          AND home_away_context = :context
                          AND state_date BETWEEN :start_date AND :end_date
                        ORDER BY state_date ASC
                    """
                    
                    states = session.execute(
                        text(states_query),
                        {
                            'team_id': team_id,
                            'league_id': league_id,
                            'context': context,
                            'start_date': start_date,
                            'end_date': end_date
                        }
                    ).fetchall()
                    
                    transition_counts = self._count_transitions(states)
                
                # Store transition matrix
                records = self.store_transition_matrix(
                    team_id, league_id, transition_matrix, transition_counts,
                    context, start_date, end_date
                )
                
                results[context] = records
                
                logger.info(
                    f"Processed transitions for team {team_id}, context {context}: "
                    f"{len(records)} records created/updated"
                )
                
            except Exception as e:
                logger.error(
                    f"Error processing transitions for team {team_id}, context {context}: {e}"
                )
                results[context] = []
        
        return results
    
    def get_transition_entropy(
        self, 
        transition_matrix: Dict[str, Dict[str, float]], 
        from_state: str
    ) -> float:
        """
        Calculate entropy of transitions from a specific state.
        
        Args:
            transition_matrix: Transition probability matrix
            from_state: Source state
            
        Returns:
            Entropy value (higher = more unpredictable)
        """
        try:
            if from_state not in transition_matrix:
                return 0.0
            
            probabilities = list(transition_matrix[from_state].values())
            probabilities = [p for p in probabilities if p > 0]  # Remove zero probabilities
            
            if not probabilities:
                return 0.0
            
            entropy = -sum(p * np.log2(p) for p in probabilities)
            return entropy
            
        except Exception as e:
            logger.error(f"Error calculating transition entropy: {e}")
            return 0.0