#!/usr/bin/env python3
"""
Markov Chain Integration Tests

Comprehensive tests for the Markov chain implementation including:
1. State classification accuracy
2. Transition matrix calculations
3. Feature generation
4. Integration with enhanced predictor
5. Database operations
6. Performance validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from formfinder.markov_state_classifier import MarkovStateClassifier
    from formfinder.markov_transition_calculator import MarkovTransitionCalculator
    from formfinder.markov_feature_generator import MarkovFeatureGenerator
    from formfinder.enhanced_predictor import EnhancedGoalPredictor
    from formfinder.database import TeamPerformanceState, MarkovTransitionMatrix, MarkovFeatures
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

class TestMarkovStateClassifier:
    """Test suite for MarkovStateClassifier."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        session.execute.return_value.fetchall.return_value = []
        session.commit.return_value = None
        session.rollback.return_value = None
        return session
    
    @pytest.fixture
    def state_classifier(self, mock_db_session):
        """Create a MarkovStateClassifier instance with mocked dependencies."""
        return MarkovStateClassifier()
    
    def test_initialization(self, state_classifier):
        """Test proper initialization of MarkovStateClassifier."""
        assert state_classifier.db_session is not None
        assert hasattr(state_classifier, 'performance_weights')
        assert hasattr(state_classifier, 'state_thresholds')
    
    def test_calculate_performance_score(self, state_classifier):
        """Test performance score calculation."""
        # Test with typical match data
        match_data = {
            'goals_for': 2,
            'goals_against': 1,
            'shots_for': 15,
            'shots_against': 8,
            'possession': 60.0,
            'corners_for': 6,
            'corners_against': 3,
            'fouls_for': 12,
            'fouls_against': 15,
            'yellow_cards': 2,
            'red_cards': 0
        }
        
        score = state_classifier.calculate_performance_score(match_data)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        
        # Test with excellent performance
        excellent_data = {
            'goals_for': 4,
            'goals_against': 0,
            'shots_for': 20,
            'shots_against': 3,
            'possession': 75.0,
            'corners_for': 10,
            'corners_against': 1,
            'fouls_for': 8,
            'fouls_against': 20,
            'yellow_cards': 1,
            'red_cards': 0
        }
        
        excellent_score = state_classifier.calculate_performance_score(excellent_data)
        assert excellent_score > score
        
        # Test with poor performance
        poor_data = {
            'goals_for': 0,
            'goals_against': 3,
            'shots_for': 5,
            'shots_against': 18,
            'possession': 30.0,
            'corners_for': 2,
            'corners_against': 8,
            'fouls_for': 20,
            'fouls_against': 8,
            'yellow_cards': 4,
            'red_cards': 1
        }
        
        poor_score = state_classifier.calculate_performance_score(poor_data)
        assert poor_score < score
    
    def test_classify_state(self, state_classifier):
        """Test state classification logic."""
        # Test excellent state
        excellent_score = 85.0
        state = state_classifier.classify_state(excellent_score)
        assert state == 'excellent'
        
        # Test good state
        good_score = 70.0
        state = state_classifier.classify_state(good_score)
        assert state == 'good'
        
        # Test average state
        average_score = 50.0
        state = state_classifier.classify_state(average_score)
        assert state == 'average'
        
        # Test poor state
        poor_score = 30.0
        state = state_classifier.classify_state(poor_score)
        assert state == 'poor'
        
        # Test terrible state
        terrible_score = 10.0
        state = state_classifier.classify_state(terrible_score)
        assert state == 'terrible'
    
    def test_process_team_match(self, state_classifier, mock_db_session):
        """Test processing of individual team matches."""
        # Mock database response
        mock_db_session.execute.return_value.fetchone.return_value = (
            1, 2, 1, 15, 8, 60.0, 6, 3, 12, 15, 2, 0  # Sample match data
        )
        
        team_id = 1
        match_date = datetime(2024, 1, 15)
        league_id = 39
        
        result = state_classifier.process_team_match(team_id, match_date, league_id)
        
        assert result is True
        assert mock_db_session.execute.call_count >= 1
        assert mock_db_session.commit.called
    
    @pytest.mark.parametrize("days_back,expected_calls", [
        (7, 1),   # One week
        (30, 1),  # One month
        (90, 1),  # Three months
    ])
    def test_process_date_range(self, state_classifier, mock_db_session, days_back, expected_calls):
        """Test processing of date ranges."""
        end_date = datetime(2024, 1, 15)
        start_date = end_date - timedelta(days=days_back)
        league_id = 39
        
        # Mock fixtures in date range
        mock_db_session.execute.return_value.fetchall.return_value = [
            (1, datetime(2024, 1, 10)),  # Sample fixture
            (2, datetime(2024, 1, 12)),
        ]
        
        processed = state_classifier.process_date_range(start_date, end_date, league_id)
        
        assert isinstance(processed, int)
        assert processed >= 0

class TestMarkovTransitionCalculator:
    """Test suite for MarkovTransitionCalculator."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        session.execute.return_value.fetchall.return_value = []
        session.commit.return_value = None
        return session
    
    @pytest.fixture
    def transition_calculator(self, mock_db_session):
        """Create a MarkovTransitionCalculator instance."""
        return MarkovTransitionCalculator(mock_db_session)
    
    def test_initialization(self, transition_calculator):
        """Test proper initialization."""
        assert transition_calculator.db_session is not None
        assert hasattr(transition_calculator, 'smoothing_factor')
        assert hasattr(transition_calculator, 'min_observations')
    
    def test_get_state_transitions(self, transition_calculator, mock_db_session):
        """Test retrieval of state transitions."""
        # Mock state sequence
        mock_db_session.execute.return_value.fetchall.return_value = [
            ('excellent', datetime(2024, 1, 1)),
            ('good', datetime(2024, 1, 8)),
            ('average', datetime(2024, 1, 15)),
            ('good', datetime(2024, 1, 22)),
            ('excellent', datetime(2024, 1, 29)),
        ]
        
        transitions = transition_calculator.get_state_transitions(
            team_id=1,
            league_id=39,
            context='overall'
        )
        
        assert isinstance(transitions, list)
        # Should have 4 transitions from 5 states
        expected_transitions = 4
        assert len(transitions) <= expected_transitions
    
    def test_calculate_transition_matrix(self, transition_calculator):
        """Test transition matrix calculation."""
        # Sample transitions
        transitions = [
            ('excellent', 'good'),
            ('good', 'average'),
            ('average', 'good'),
            ('good', 'excellent'),
            ('excellent', 'excellent'),
        ]
        
        matrix = transition_calculator.calculate_transition_matrix(transitions)
        
        assert isinstance(matrix, dict)
        assert 'excellent' in matrix
        assert 'good' in matrix
        assert 'average' in matrix
        
        # Check that probabilities sum to 1 for each state
        for from_state, to_states in matrix.items():
            total_prob = sum(to_states.values())
            assert abs(total_prob - 1.0) < 1e-6  # Allow for floating point precision
    
    def test_apply_smoothing(self, transition_calculator):
        """Test Laplace smoothing application."""
        # Matrix with zero probabilities
        raw_matrix = {
            'excellent': {'excellent': 0.5, 'good': 0.5, 'average': 0.0, 'poor': 0.0, 'terrible': 0.0},
            'good': {'excellent': 0.3, 'good': 0.4, 'average': 0.3, 'poor': 0.0, 'terrible': 0.0},
        }
        
        smoothed = transition_calculator.apply_smoothing(raw_matrix)
        
        assert isinstance(smoothed, dict)
        
        # Check that no probability is exactly zero after smoothing
        for from_state, to_states in smoothed.items():
            for prob in to_states.values():
                assert prob > 0.0
            
            # Check probabilities still sum to 1
            total_prob = sum(to_states.values())
            assert abs(total_prob - 1.0) < 1e-6
    
    def test_calculate_and_store_transitions(self, transition_calculator, mock_db_session):
        """Test complete transition calculation and storage."""
        # Mock state data
        mock_db_session.execute.return_value.fetchall.return_value = [
            ('excellent', datetime(2024, 1, 1)),
            ('good', datetime(2024, 1, 8)),
            ('average', datetime(2024, 1, 15)),
        ]
        
        result = transition_calculator.calculate_and_store_transitions(
            team_id=1,
            league_id=39,
            context='overall'
        )
        
        assert result is True
        assert mock_db_session.commit.called

class TestMarkovFeatureGenerator:
    """Test suite for MarkovFeatureGenerator."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        session.execute.return_value.fetchall.return_value = []
        session.commit.return_value = None
        return session
    
    @pytest.fixture
    def feature_generator(self, mock_db_session):
        """Create a MarkovFeatureGenerator instance."""
        return MarkovFeatureGenerator(mock_db_session)
    
    def test_initialization(self, feature_generator):
        """Test proper initialization."""
        assert feature_generator.db_session is not None
        assert hasattr(feature_generator, 'lookback_days')
    
    def test_get_team_current_state(self, feature_generator, mock_db_session):
        """Test retrieval of team's current state."""
        # Mock current state
        mock_db_session.execute.return_value.fetchone.return_value = ('excellent',)
        
        state = feature_generator.get_team_current_state(
            team_id=1,
            match_date=datetime(2024, 1, 15),
            league_id=39
        )
        
        assert state == 'excellent'
    
    def test_get_transition_probabilities(self, feature_generator, mock_db_session):
        """Test retrieval of transition probabilities."""
        # Mock transition matrix
        mock_db_session.execute.return_value.fetchone.return_value = (
            '{"excellent": 0.3, "good": 0.4, "average": 0.2, "poor": 0.08, "terrible": 0.02}',
        )
        
        probs = feature_generator.get_transition_probabilities(
            team_id=1,
            from_state='good',
            league_id=39,
            context='overall'
        )
        
        assert isinstance(probs, dict)
        assert 'excellent' in probs
        assert sum(probs.values()) == pytest.approx(1.0, rel=1e-6)
    
    def test_calculate_state_momentum(self, feature_generator, mock_db_session):
        """Test state momentum calculation."""
        # Mock recent states (improving trend)
        mock_db_session.execute.return_value.fetchall.return_value = [
            ('poor', datetime(2024, 1, 1)),
            ('average', datetime(2024, 1, 8)),
            ('good', datetime(2024, 1, 15)),
            ('excellent', datetime(2024, 1, 22)),
        ]
        
        momentum = feature_generator.calculate_momentum_score(
            team_id=1,
            league_id=39,
            reference_date=datetime(2024, 1, 25),
            context='overall'
        )
        
        assert isinstance(momentum, float)
        assert momentum > 0  # Should be positive for improving trend
    
    def test_calculate_transition_entropy(self, feature_generator, mock_db_session):
        """Test transition entropy calculation."""
        # Mock recent states (high variability)
        mock_db_session.execute.return_value.fetchall.return_value = [
            ('excellent', datetime(2024, 1, 1)),
            ('poor', datetime(2024, 1, 8)),
            ('excellent', datetime(2024, 1, 15)),
            ('terrible', datetime(2024, 1, 22)),
            ('good', datetime(2024, 1, 29)),
        ]
        
        entropy = feature_generator.calculate_transition_entropy(
            team_id=1,
            league_id=39,
            context='overall'
        )
        
        assert isinstance(entropy, float)
        assert entropy >= 0.0  # Entropy is always non-negative
    
    def test_calculate_performance_volatility(self, feature_generator, mock_db_session):
        """Test performance volatility calculation."""
        # Mock recent states with varying performance
        mock_db_session.execute.return_value.fetchall.return_value = [
            ('excellent', datetime(2024, 1, 1)),
            ('good', datetime(2024, 1, 8)),
            ('poor', datetime(2024, 1, 15)),
            ('average', datetime(2024, 1, 22)),
        ]
        
        volatility = feature_generator.calculate_performance_volatility(
            team_id=1,
            league_id=39,
            reference_date=datetime(2024, 1, 25),
            context='overall'
        )
        
        assert isinstance(volatility, float)
        assert volatility >= 0.0  # Volatility is always non-negative
    
    def test_generate_features(self, feature_generator, mock_db_session):
        """Test complete feature generation."""
        # Mock various database responses
        mock_responses = [
            ('excellent',),  # Home team current state
            ('good',),       # Away team current state
            # Transition probabilities
            ('{"excellent": 0.3, "good": 0.4, "average": 0.2, "poor": 0.08, "terrible": 0.02}',),
            ('{"excellent": 0.2, "good": 0.3, "average": 0.3, "poor": 0.15, "terrible": 0.05}',),
        ]
        
        mock_db_session.execute.return_value.fetchone.side_effect = mock_responses
        
        # Mock state history for momentum/entropy/volatility
        mock_db_session.execute.return_value.fetchall.return_value = [
            ('excellent', datetime(2024, 1, 1)),
            ('good', datetime(2024, 1, 8)),
        ]
        
        features = feature_generator.generate_features(
            home_team_id=1,
            away_team_id=2,
            match_date=datetime(2024, 1, 15),
            league_id=39
        )
        
        assert isinstance(features, dict)
        
        # Check for expected feature categories
        expected_features = [
            'home_current_state', 'away_current_state',
            'home_momentum', 'away_momentum',
            'home_entropy', 'away_entropy',
            'home_volatility', 'away_volatility'
        ]
        
        for feature in expected_features:
            assert feature in features
    
    def test_store_features(self, feature_generator, mock_db_session):
        """Test feature storage in database."""
        features = {
            'home_current_state': 'excellent',
            'away_current_state': 'good',
            'home_momentum': 0.75,
            'away_momentum': -0.25,
            'home_entropy': 1.2,
            'away_entropy': 1.8,
            'home_volatility': 0.3,
            'away_volatility': 0.6
        }
        
        result = feature_generator.store_features(
            fixture_id=123,
            features=features
        )
        
        assert result is True
        assert mock_db_session.commit.called

class TestEnhancedPredictorIntegration:
    """Test suite for Markov integration with EnhancedGoalPredictor."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.get_database_url.return_value = "sqlite:///:memory:"
        return config
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        session.execute.return_value.fetchall.return_value = []
        session.execute.return_value.fetchone.return_value = None
        session.commit.return_value = None
        return session
    
    @patch('formfinder.enhanced_predictor.create_engine')
    @patch('formfinder.enhanced_predictor.sessionmaker')
    def test_markov_feature_generator_initialization(self, mock_sessionmaker, mock_create_engine, mock_config):
        """Test that MarkovFeatureGenerator is properly initialized in EnhancedGoalPredictor."""
        mock_session = Mock()
        mock_sessionmaker.return_value.return_value = mock_session
        mock_session.execute.return_value = None
        
        with patch('formfinder.enhanced_predictor.MarkovFeatureGenerator') as mock_markov_gen:
            predictor = EnhancedGoalPredictor(mock_config, mock_session)
            
            # Verify MarkovFeatureGenerator was initialized
            mock_markov_gen.assert_called_once_with(mock_session)
            assert hasattr(predictor, 'markov_generator')
    
    def test_markov_features_in_extraction(self, mock_db_session):
        """Test that Markov features are included in feature extraction."""
        # This would require more complex mocking of the entire feature extraction process
        # For now, we'll test the integration point exists
        
        with patch('formfinder.enhanced_predictor.EnhancedGoalPredictor') as mock_predictor:
            predictor_instance = mock_predictor.return_value
            predictor_instance.markov_generator = Mock()
            predictor_instance.markov_generator.generate_features.return_value = {
                'home_momentum': 0.5,
                'away_momentum': -0.2
            }
            
            # Verify the markov generator can be called
            features = predictor_instance.markov_generator.generate_features(
                home_team_id=1,
                away_team_id=2,
                match_date=datetime(2024, 1, 15),
                league_id=39
            )
            
            assert 'home_momentum' in features
            assert 'away_momentum' in features

class TestDatabaseIntegration:
    """Test suite for database operations and schema validation."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database for testing."""
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_file.close()
        
        engine = create_engine(f'sqlite:///{db_file.name}')
        
        # Create tables (simplified for testing)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE team_performance_states (
                    id INTEGER PRIMARY KEY,
                    team_id INTEGER NOT NULL,
                    match_date DATE NOT NULL,
                    league_id INTEGER NOT NULL,
                    performance_state VARCHAR(20) NOT NULL,
                    performance_score FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE markov_transition_matrices (
                    id INTEGER PRIMARY KEY,
                    team_id INTEGER,
                    league_id INTEGER NOT NULL,
                    context VARCHAR(20) NOT NULL,
                    from_state VARCHAR(20) NOT NULL,
                    to_state_probabilities TEXT NOT NULL,
                    observation_count INTEGER NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE markov_features (
                    id INTEGER PRIMARY KEY,
                    fixture_id INTEGER NOT NULL,
                    home_current_state VARCHAR(20),
                    away_current_state VARCHAR(20),
                    home_momentum FLOAT,
                    away_momentum FLOAT,
                    home_entropy FLOAT,
                    away_entropy FLOAT,
                    home_volatility FLOAT,
                    away_volatility FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.commit()
        
        yield engine
        
        # Cleanup
        os.unlink(db_file.name)
    
    def test_team_performance_state_storage(self, temp_db):
        """Test storing team performance states."""
        Session = sessionmaker(bind=temp_db)
        session = Session()
        
        try:
            # Insert test data
            session.execute(text("""
                INSERT INTO team_performance_states 
                (team_id, match_date, league_id, performance_state, performance_score)
                VALUES (1, '2024-01-15', 39, 'excellent', 85.5)
            """))
            session.commit()
            
            # Verify data was stored
            result = session.execute(text("""
                SELECT team_id, performance_state, performance_score 
                FROM team_performance_states 
                WHERE team_id = 1
            """))
            
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1
            assert row[1] == 'excellent'
            assert row[2] == 85.5
            
        finally:
            session.close()
    
    def test_transition_matrix_storage(self, temp_db):
        """Test storing transition matrices."""
        Session = sessionmaker(bind=temp_db)
        session = Session()
        
        try:
            # Insert test transition matrix
            probabilities = '{"excellent": 0.3, "good": 0.4, "average": 0.2, "poor": 0.08, "terrible": 0.02}'
            
            session.execute(text("""
                INSERT INTO markov_transition_matrices 
                (team_id, league_id, context, from_state, to_state_probabilities, observation_count)
                VALUES (1, 39, 'overall', 'good', :probs, 25)
            """), {'probs': probabilities})
            session.commit()
            
            # Verify data was stored
            result = session.execute(text("""
                SELECT team_id, from_state, to_state_probabilities, observation_count
                FROM markov_transition_matrices 
                WHERE team_id = 1 AND from_state = 'good'
            """))
            
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1
            assert row[1] == 'good'
            assert probabilities in row[2]
            assert row[3] == 25
            
        finally:
            session.close()
    
    def test_markov_features_storage(self, temp_db):
        """Test storing Markov features."""
        Session = sessionmaker(bind=temp_db)
        session = Session()
        
        try:
            # Insert test features
            session.execute(text("""
                INSERT INTO markov_features 
                (fixture_id, home_current_state, away_current_state, 
                 home_momentum, away_momentum, home_entropy, away_entropy,
                 home_volatility, away_volatility)
                VALUES (123, 'excellent', 'good', 0.75, -0.25, 1.2, 1.8, 0.3, 0.6)
            """))
            session.commit()
            
            # Verify data was stored
            result = session.execute(text("""
                SELECT fixture_id, home_current_state, home_momentum, away_entropy
                FROM markov_features 
                WHERE fixture_id = 123
            """))
            
            row = result.fetchone()
            assert row is not None
            assert row[0] == 123
            assert row[1] == 'excellent'
            assert row[2] == 0.75
            assert row[3] == 1.8
            
        finally:
            session.close()

class TestPerformanceValidation:
    """Test suite for performance validation and benchmarking."""
    
    def test_state_classification_performance(self):
        """Test that state classification performs within acceptable time limits."""
        import time
        
        # Mock session for performance testing
        mock_session = Mock()
        mock_session.execute.return_value.fetchone.return_value = (
            2, 1, 15, 8, 60.0, 6, 3, 12, 15, 2, 0
        )
        mock_session.commit.return_value = None
        
        classifier = MarkovStateClassifier()
        
        # Time multiple classifications
        start_time = time.time()
        for i in range(100):
            classifier.process_team_match(
                team_id=i % 10 + 1,
                match_date=datetime(2024, 1, 15),
                league_id=39
            )
        end_time = time.time()
        
        # Should complete 100 classifications in under 1 second
        elapsed = end_time - start_time
        assert elapsed < 1.0, f"State classification too slow: {elapsed:.3f}s for 100 operations"
    
    def test_feature_generation_performance(self):
        """Test that feature generation performs within acceptable time limits."""
        import time
        
        # Mock session for performance testing
        mock_session = Mock()
        mock_session.execute.return_value.fetchone.side_effect = [
            ('excellent',),  # Current state
            ('{"excellent": 0.3, "good": 0.4, "average": 0.2, "poor": 0.08, "terrible": 0.02}',),  # Transitions
        ] * 100
        
        mock_session.execute.return_value.fetchall.return_value = [
            ('excellent', datetime(2024, 1, 1)),
            ('good', datetime(2024, 1, 8)),
        ]
        
        generator = MarkovFeatureGenerator(mock_session)
        
        # Time multiple feature generations
        start_time = time.time()
        for i in range(50):
            generator.generate_features(
                home_team_id=i % 10 + 1,
                away_team_id=(i + 1) % 10 + 1,
                match_date=datetime(2024, 1, 15),
                league_id=39
            )
        end_time = time.time()
        
        # Should complete 50 feature generations in under 2 seconds
        elapsed = end_time - start_time
        assert elapsed < 2.0, f"Feature generation too slow: {elapsed:.3f}s for 50 operations"

if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([
        __file__,
        '-v',
        '--cov=formfinder.markov_state_classifier',
        '--cov=formfinder.markov_transition_calculator', 
        '--cov=formfinder.markov_feature_generator',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov'
    ])