"""Database models and setup for FormFinder."""

import logging
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, Index, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

from .config import get_config

logger = logging.getLogger(__name__)

Base = declarative_base()


class League(Base):
    """League information."""
    __tablename__ = 'leagues'
    
    # Surrogate primary key to maintain foreign key compatibility
    league_pk = Column(Integer, primary_key=True, autoincrement=True)
    
    # League identifier from API (can repeat for different seasons)
    id = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    country = Column(String(100))
    season = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    teams = relationship("Team", back_populates="league")
    fixtures = relationship("Fixture", back_populates="league")
    standings = relationship("Standing", back_populates="league")
    form_analysis = relationship("TeamFormAnalysis", back_populates="league")
    combined_analysis = relationship("CombinedAnalysis", back_populates="league")
    
    __table_args__ = (
        UniqueConstraint('id', 'season', name='uq_league_season'),
    )


class Team(Base):
    """Team information."""
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    name = Column(String(255), nullable=False)
    short_name = Column(String(50))
    logo_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    league = relationship("League", back_populates="teams")
    home_fixtures = relationship("Fixture", foreign_keys="Fixture.home_team_id", back_populates="home_team")
    away_fixtures = relationship("Fixture", foreign_keys="Fixture.away_team_id", back_populates="away_team")
    standings = relationship("Standing", back_populates="team")
    
    # Sentiment analysis relationships
    sentiment_data = relationship("SentimentData", back_populates="team")
    sentiment_scores = relationship("TeamSentimentScore", back_populates="team")
    form_analysis = relationship("TeamFormAnalysis", back_populates="team")
    combined_analysis = relationship("CombinedAnalysis", back_populates="team")
    
    __table_args__ = (
        Index('ix_team_league', 'league_id'),
    )


class Fixture(Base):
    """Match fixtures and results."""
    __tablename__ = 'fixtures'
    
    id = Column(Integer, primary_key=True)
    api_fixture_id = Column(String(50), unique=True, nullable=True)  # API fixture ID for detailed data lookup
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    home_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    match_date = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False)  # scheduled, live, finished, postponed, cancelled
    round_number = Column(Integer)
    
    # Match result
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_score_ht = Column(Integer)  # Half-time score
    away_score_ht = Column(Integer)
    home_score_et = Column(Integer)  # Extra time score
    away_score_et = Column(Integer)
    home_score_pen = Column(Integer)  # Penalty score
    away_score_pen = Column(Integer)
    
    # Match details
    referee = Column(String(255))
    venue = Column(String(255))
    attendance = Column(Integer)
    winner = Column(String(10))  # 'home', 'away', 'draw'
    has_extra_time = Column(Boolean, default=False)
    has_penalties = Column(Boolean, default=False)
    minute = Column(Integer)  # Current minute for live matches
    
    # Stadium information
    stadium_id = Column(Integer)
    stadium_name = Column(String(255))
    stadium_city = Column(String(255))
    
    # Formation information
    home_formation = Column(String(20))
    away_formation = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    league = relationship("League", back_populates="fixtures")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_fixtures")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_fixtures")
    predictions = relationship("Prediction", back_populates="fixture")
    events = relationship("MatchEvent", back_populates="fixture")
    odds = relationship("MatchOdds", back_populates="fixture")
    lineups = relationship("MatchLineup", back_populates="fixture")
    weather_data = relationship("WeatherData", back_populates="fixture")
    weather_forecasts = relationship("WeatherForecast", back_populates="fixture")
    
    __table_args__ = (
        Index('ix_fixture_league_date', 'league_id', 'match_date'),
        Index('ix_fixture_teams', 'home_team_id', 'away_team_id'),
        Index('ix_fixture_status', 'status'),
        Index('ix_fixture_stadium', 'stadium_id'),
    )


class MatchEvent(Base):
    """Match events (goals, cards, substitutions, etc.)."""
    __tablename__ = 'match_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    event_type = Column(String(50), nullable=False)  # 'goal', 'yellow_card', 'red_card', 'substitution', etc.
    event_minute = Column(String(10))  # Can be '45+2', '90+4', etc.
    team = Column(String(10))  # 'home' or 'away'
    player_id = Column(Integer)
    player_name = Column(String(255))
    assist_player_id = Column(Integer)
    assist_player_name = Column(String(255))
    description = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    fixture = relationship("Fixture", back_populates="events")
    
    __table_args__ = (
        Index('ix_event_fixture', 'fixture_id'),
        Index('ix_event_type', 'event_type'),
        Index('ix_event_minute', 'event_minute'),
    )


class MatchOdds(Base):
    """Match betting odds."""
    __tablename__ = 'match_odds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    
    # Match winner odds
    home_win_odds = Column(Float)
    draw_odds = Column(Float)
    away_win_odds = Column(Float)
    
    # Over/Under odds
    over_under_total = Column(Float)  # Total goals line (e.g., 2.5)
    over_odds = Column(Float)
    under_odds = Column(Float)
    
    # Handicap odds
    handicap_market = Column(Float)  # Handicap line (e.g., -0.5)
    handicap_home_odds = Column(Float)
    handicap_away_odds = Column(Float)
    
    # Metadata
    last_modified_timestamp = Column(Integer)  # Unix timestamp
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fixture = relationship("Fixture", back_populates="odds")
    
    __table_args__ = (
        Index('ix_odds_fixture', 'fixture_id'),
    )


class MatchLineup(Base):
    """Match lineups and player positions."""
    __tablename__ = 'match_lineups'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    player_id = Column(Integer)
    player_name = Column(String(255), nullable=False)
    team = Column(String(10), nullable=False)  # 'home' or 'away'
    position = Column(String(10))  # 'G', 'D', 'M', 'F'
    lineup_type = Column(String(20))  # 'starting', 'bench', 'sidelined'
    status = Column(String(20))  # For sidelined players: 'out', 'doubtful', etc.
    status_description = Column(String(255))  # 'Injury', 'Suspension', etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    fixture = relationship("Fixture", back_populates="lineups")
    
    __table_args__ = (
        Index('ix_lineup_fixture', 'fixture_id'),
        Index('ix_lineup_player', 'player_id'),
        Index('ix_lineup_team', 'team'),
        Index('ix_lineup_type', 'lineup_type'),
    )


class Standing(Base):
    """League standings/table."""
    __tablename__ = 'standings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    season = Column(String(20), nullable=False)  # Season identifier (e.g., '2024-2025')
    position = Column(Integer, nullable=False)
    played = Column(Integer, default=0)
    won = Column(Integer, default=0)
    drawn = Column(Integer, default=0)
    lost = Column(Integer, default=0)
    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    goal_difference = Column(Integer, default=0)
    points = Column(Integer, default=0)
    
    # Form and streaks
    form = Column(String(10))  # Last 5 matches: WWDLL
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    league = relationship("League", back_populates="standings")
    team = relationship("Team", back_populates="standings")
    
    __table_args__ = (
        UniqueConstraint('league_id', 'team_id', 'season', name='uq_standing_league_team_season'),
        Index('ix_standing_league_position', 'league_id', 'position'),
        Index('ix_standing_league_season', 'league_id', 'season'),
    )


class HighFormTeam(Base):
    """High-form teams identified by standalone_data_processor."""
    __tablename__ = 'high_form_teams'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    
    # Team information
    team_name = Column(String(255), nullable=False)  # Team name for easy identification
    
    # Form data
    win_rate = Column(Float, nullable=False)  # Win rate as a decimal (0.0-1.0)
    total_matches = Column(Integer, nullable=False)  # Total matches considered
    wins = Column(Integer, nullable=False)  # Number of wins
    
    # Analysis period
    analysis_start_date = Column(DateTime, nullable=False)  # Start of analysis period
    analysis_end_date = Column(DateTime, nullable=False)  # End of analysis period
    
    # Metadata
    algorithm_version = Column(String(50))
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team")
    league = relationship("League")
    
    __table_args__ = (
        Index('ix_high_form_team_league', 'league_id'),
        Index('ix_high_form_team_team', 'team_id'),
        Index('ix_high_form_team_analysis_date', 'analysis_date'),
        Index('ix_high_form_team_win_rate', 'win_rate'),
    )


class Prediction(Base):
    """Match predictions."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    
    # Prediction data
    predicted_total_goals = Column(Float)
    over_2_5_probability = Column(Float)
    predicted_home_score = Column(Float)
    predicted_away_score = Column(Float)
    confidence_score = Column(Float)  # 0-1 scale
    
    # Form analysis
    home_team_form_score = Column(Float)
    away_team_form_score = Column(Float)
    home_team_recent_wins = Column(Integer)
    away_team_recent_wins = Column(Integer)
    
    # Sentiment analysis
    home_team_sentiment = Column(Float, nullable=True)
    away_team_sentiment = Column(Float, nullable=True)
    sentiment_articles_analyzed = Column(Integer, nullable=True)
    
    # Head-to-head data
    h2h_home_wins = Column(Integer)
    h2h_draws = Column(Integer)
    h2h_away_wins = Column(Integer)
    h2h_total_matches = Column(Integer)
    
    # Prediction metadata
    algorithm_version = Column(String(50))
    features_used = Column(Text)  # JSON string of features
    prediction_date = Column(DateTime, default=datetime.utcnow)
    
    # Result tracking
    actual_result = Column(String(10))  # 'home_win', 'draw', 'away_win'
    prediction_correct = Column(Boolean)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fixture = relationship("Fixture", back_populates="predictions")
    
    __table_args__ = (
        Index('ix_prediction_fixture', 'fixture_id'),
        Index('ix_prediction_date', 'prediction_date'),
        Index('ix_prediction_confidence', 'confidence_score'),
    )


class DataFetchLog(Base):
    """Log of data fetching operations."""
    __tablename__ = 'data_fetch_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    league_id = Column(Integer, nullable=False)
    data_type = Column(String(50), nullable=False)  # 'fixtures', 'standings', 'teams'
    fetch_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), nullable=False)  # 'success', 'error', 'partial'
    records_fetched = Column(Integer, default=0)
    error_message = Column(Text)
    duration_seconds = Column(Float)
    
    __table_args__ = (
        Index('ix_fetch_log_league_type', 'league_id', 'data_type'),
        Index('ix_fetch_log_date', 'fetch_date'),
    )


class SentimentSource(Base):
    """Sources for sentiment data."""
    __tablename__ = 'sentiment_sources'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String(100), unique=True, nullable=False)
    source_type = Column(String(50), nullable=False)  # 'news', 'twitter', 'reddit'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sentiment_data = relationship("SentimentData", back_populates="source")


class SentimentData(Base):
    """Raw sentiment data from various sources."""
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    source_id = Column(Integer, ForeignKey('sentiment_sources.id'))
    content = Column(Text, nullable=False)
    sentiment_score = Column(Float)  # -1.0 to 1.0
    confidence_score = Column(Float)  # 0.0 to 1.0
    sentiment_label = Column(String(20))  # 'positive', 'negative', 'neutral'
    author = Column(String(255))
    post_date = Column(DateTime)
    url = Column(String(500))
    raw_data = Column(Text)  # JSON string for additional metadata
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="sentiment_data")
    source = relationship("SentimentSource", back_populates="sentiment_data")
    
    __table_args__ = (
        Index('ix_sentiment_team', 'team_id'),
        Index('ix_sentiment_source', 'source_id'),
        Index('ix_sentiment_post_date', 'post_date'),
    )


class TeamSentimentScore(Base):
    """Aggregated sentiment scores for teams."""
    __tablename__ = 'team_sentiment_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    date = Column(DateTime, nullable=False)
    avg_sentiment_score = Column(Float)  # -1.0 to 1.0
    sentiment_volume = Column(Integer, default=0)
    positive_ratio = Column(Float)  # 0.0 to 1.0
    negative_ratio = Column(Float)  # 0.0 to 1.0
    neutral_ratio = Column(Float)  # 0.0 to 1.0
    data_points = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="sentiment_scores")
    
    __table_args__ = (
        Index('ix_team_sentiment_team', 'team_id'),
        Index('ix_team_sentiment_date', 'date'),
    )


class TeamFormAnalysis(Base):
    """Team form analysis data."""
    __tablename__ = 'team_form_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    league_id = Column(Integer, ForeignKey('leagues.league_pk'))
    analysis_date = Column(DateTime, nullable=False)
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    win_rate = Column(Float)  # 0.0 to 1.0
    form_score = Column(Float)  # Calculated form score
    streak = Column(String(10))  # Current form streak (e.g., "WWDWL")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="form_analysis")
    league = relationship("League", back_populates="form_analysis")
    
    __table_args__ = (
        Index('ix_team_form_team', 'team_id'),
        Index('ix_team_form_league', 'league_id'),
        Index('ix_team_form_date', 'analysis_date'),
    )


class CombinedAnalysis(Base):
    """Combined sentiment and form analysis."""
    __tablename__ = 'combined_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    league_id = Column(Integer, ForeignKey('leagues.league_pk'))
    analysis_date = Column(DateTime, nullable=False)
    sentiment_score = Column(Float)  # -1.0 to 1.0
    form_score = Column(Float)  # 0.0 to 1.0
    combined_score = Column(Float)  # Weighted combination
    sentiment_weight = Column(Float, default=0.3)  # Configurable weight
    form_weight = Column(Float, default=0.7)  # Configurable weight
    prediction_confidence = Column(Float)  # Confidence in prediction
    trend_direction = Column(String(20))  # 'improving', 'declining', 'stable'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="combined_analysis")
    league = relationship("League", back_populates="combined_analysis")
    
    __table_args__ = (
        Index('ix_combined_team', 'team_id'),
        Index('ix_combined_league', 'league_id'),
        Index('ix_combined_date', 'analysis_date'),
    )


class AlertThreshold(Base):
    """Alert thresholds for monitoring."""
    __tablename__ = 'alert_thresholds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    threshold_type = Column(String(50), nullable=False)  # 'sentiment', 'form', 'combined'
    min_value = Column(Float)
    max_value = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_alert_team', 'team_id'),
        Index('ix_alert_threshold_type', 'threshold_type'),
    )


class NotificationLog(Base):
    """Log of notifications sent."""
    __tablename__ = 'notification_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    threshold_id = Column(Integer, ForeignKey('alert_thresholds.id'))
    notification_type = Column(String(50))  # 'email', 'webhook', 'sms'
    message = Column(Text)
    sent_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='sent')  # 'sent', 'failed', 'pending'
    
    __table_args__ = (
        Index('ix_notification_team', 'team_id'),
        Index('ix_notification_sent', 'sent_at'),
    )


class ProcessingStatus(Base):
    """Status tracking for background processes."""
    __tablename__ = 'processing_status'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    process_name = Column(String(100), unique=True, nullable=False)
    last_run = Column(DateTime)
    status = Column(String(20))  # 'running', 'success', 'failed', 'idle'
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_processing_name', 'process_name'),
    )


class PreComputedFeatures(Base):
    """Pre-computed features for training and prediction."""
    __tablename__ = 'pre_computed_features'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    home_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    match_date = Column(DateTime, nullable=False)
    
    # Team form features (actual database columns)
    home_avg_goals_scored = Column(Float)
    home_avg_goals_conceded = Column(Float)
    home_avg_goals_scored_home = Column(Float)
    home_avg_goals_conceded_home = Column(Float)
    home_form_last_5_games = Column(String)
    home_wins_last_5 = Column(Integer)
    home_draws_last_5 = Column(Integer)
    home_losses_last_5 = Column(Integer)
    home_goals_for_last_5 = Column(Integer)
    home_goals_against_last_5 = Column(Integer)
    
    away_avg_goals_scored = Column(Float)
    away_avg_goals_conceded = Column(Float)
    away_avg_goals_scored_away = Column(Float)
    away_avg_goals_conceded_away = Column(Float)
    away_form_last_5_games = Column(String)
    away_wins_last_5 = Column(Integer)
    away_draws_last_5 = Column(Integer)
    away_losses_last_5 = Column(Integer)
    away_goals_for_last_5 = Column(Integer)
    away_goals_against_last_5 = Column(Integer)
    
    # Head-to-head features (actual database columns)
    h2h_last_updated = Column(DateTime)
    
    # League position features - removed as they don't exist in actual database
    
    # xG features
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_avg_goals_scored = Column(Float)
    home_avg_goals_conceded = Column(Float)
    away_avg_goals_scored = Column(Float)
    away_avg_goals_conceded = Column(Float)
    
    # Team strength and momentum features
    home_team_strength = Column(Float)
    away_team_strength = Column(Float)
    home_team_momentum = Column(Float)
    away_team_momentum = Column(Float)
    
    # Sentiment features
    home_team_sentiment = Column(Float)
    away_team_sentiment = Column(Float)
    
    # Markov features
    home_team_markov_momentum = Column(Float)
    away_team_markov_momentum = Column(Float)
    home_team_state_stability = Column(Float)
    away_team_state_stability = Column(Float)
    home_team_transition_entropy = Column(Float)
    away_team_transition_entropy = Column(Float)
    home_team_performance_volatility = Column(Float)
    away_team_performance_volatility = Column(Float)
    home_team_current_state = Column(String(20))
    away_team_current_state = Column(String(20))
    home_team_state_duration = Column(Integer)
    away_team_state_duration = Column(Integer)
    home_team_expected_next_state = Column(String(20))
    away_team_expected_next_state = Column(String(20))
    home_team_state_confidence = Column(Float)
    away_team_state_confidence = Column(Float)
    markov_match_prediction_confidence = Column(Float)
    markov_outcome_probabilities = Column(Text)
    
    # Weather and excitement features
    excitement_rating = Column(Float)
    weather_temp_c = Column(Float)
    weather_temp_f = Column(Float)
    weather_humidity = Column(Float)
    weather_wind_speed = Column(Float)
    weather_precipitation = Column(Float)
    weather_condition = Column(String(50))
    
    # Metadata
    features_computed_at = Column(DateTime, default=datetime.utcnow)
    data_quality_score = Column(Float)
    computation_source = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_precomputed_fixture', 'fixture_id'),
        Index('ix_precomputed_league', 'league_id'),
        Index('ix_precomputed_teams', 'home_team_id', 'away_team_id'),
    )


class TeamPerformanceState(Base):
    """Team performance states for Markov chain analysis."""
    __tablename__ = 'team_performance_states'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=True)  # Null for aggregated states
    state_date = Column(DateTime, nullable=False)
    
    # Performance state classification
    performance_state = Column(String(20), nullable=False)  # excellent, good, average, poor, terrible
    state_score = Column(Float, nullable=False)  # Numerical score used for classification
    
    # Metrics used for state calculation
    goals_scored = Column(Float)
    goals_conceded = Column(Float)
    goal_difference = Column(Float)
    win_rate = Column(Float)
    points_per_game = Column(Float)
    form_streak = Column(String(10))  # Recent match results (e.g., "WWDLW")
    
    # Context information
    matches_considered = Column(Integer, default=5)  # Number of matches used for calculation
    home_away_context = Column(String(10))  # 'home', 'away', 'overall'
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_performance_state_team', 'team_id'),
        Index('ix_performance_state_league', 'league_id'),
        Index('ix_performance_state_date', 'state_date'),
        Index('ix_performance_state_fixture', 'fixture_id'),
        Index('ix_performance_state_context', 'home_away_context'),
    )


class MarkovTransitionMatrix(Base):
    """Markov chain transition matrices for team performance states."""
    __tablename__ = 'markov_transition_matrices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    
    # Transition definition
    from_state = Column(String(20), nullable=False)  # Source state
    to_state = Column(String(20), nullable=False)    # Target state
    
    # Transition statistics
    transition_count = Column(Integer, default=0)    # Raw count of transitions
    transition_probability = Column(Float, default=0.0)  # Calculated probability
    smoothed_probability = Column(Float, default=0.0)    # Laplace smoothed probability
    
    # Context and metadata
    home_away_context = Column(String(10), default='overall')  # 'home', 'away', 'overall'
    calculation_date = Column(DateTime, nullable=False)
    data_window_start = Column(DateTime)  # Start of data used for calculation
    data_window_end = Column(DateTime)    # End of data used for calculation
    total_transitions = Column(Integer, default=0)  # Total transitions from source state
    
    # Smoothing parameters
    smoothing_alpha = Column(Float, default=1.0)  # Laplace smoothing parameter
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_transition_team', 'team_id'),
        Index('ix_transition_league', 'league_id'),
        Index('ix_transition_states', 'from_state', 'to_state'),
        Index('ix_transition_context', 'home_away_context'),
        Index('ix_transition_date', 'calculation_date'),
        UniqueConstraint('team_id', 'league_id', 'from_state', 'to_state', 'home_away_context', 
                        name='uq_transition_matrix'),
    )


class MarkovFeatures(Base):
    """Computed Markov chain features for machine learning models."""
    __tablename__ = 'markov_features'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.league_pk'), nullable=False)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=True)  # Null for general features
    feature_date = Column(DateTime, nullable=False)
    
    # Current state information
    current_state = Column(String(20), nullable=False)
    state_duration = Column(Integer, default=1)  # How long in current state
    
    # Momentum and trend features
    momentum_score = Column(Float, default=0.0)  # Weighted momentum calculation
    trend_direction = Column(String(10))  # 'improving', 'declining', 'stable'
    
    # Stability and volatility features
    state_stability = Column(Float, default=0.0)  # How stable the current state is
    transition_entropy = Column(Float, default=0.0)  # Entropy of transition probabilities
    performance_volatility = Column(Float, default=0.0)  # Volatility in performance
    
    # Prediction features
    expected_next_state = Column(String(20))  # Most likely next state
    next_state_probability = Column(Float, default=0.0)  # Probability of next state
    state_confidence = Column(Float, default=0.0)  # Confidence in current state classification
    
    # Advanced features
    mean_return_time = Column(Float)  # Expected time to return to current state
    steady_state_probability = Column(Float)  # Long-term probability of being in current state
    absorption_probability = Column(Float)  # Probability of reaching absorbing states
    
    # Context and metadata
    home_away_context = Column(String(10), default='overall')  # 'home', 'away', 'overall'
    lookback_window = Column(Integer, default=10)  # Number of matches used for calculation
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_markov_features_team', 'team_id'),
        Index('ix_markov_features_league', 'league_id'),
        Index('ix_markov_features_fixture', 'fixture_id'),
        Index('ix_markov_features_date', 'feature_date'),
        Index('ix_markov_features_state', 'current_state'),
        Index('ix_markov_features_context', 'home_away_context'),
    )


class FeatureComputationQueue(Base):
    """Queue for feature computation tasks."""
    __tablename__ = 'feature_computation_queue'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(50), nullable=False)  # 'feature_computation', 'model_training', etc.
    task_priority = Column(String(20), default='medium')  # 'low', 'medium', 'high', 'critical'
    task_status = Column(String(20), default='pending')  # 'pending', 'running', 'completed', 'failed'
    
    # Task parameters (JSON)
    task_parameters = Column(Text)  # JSON string with task-specific parameters
    
    # Dependencies
    depends_on_task_id = Column(Integer, ForeignKey('feature_computation_queue.id'))
    
    # Execution details
    scheduled_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Progress tracking
    progress_percentage = Column(Float, default=0.0)
    progress_message = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dependency = relationship("FeatureComputationQueue", remote_side=[id])
    
    __table_args__ = (
        Index('ix_queue_status', 'task_status'),
        Index('ix_queue_priority', 'task_priority'),
        Index('ix_queue_scheduled', 'scheduled_at'),
        Index('ix_queue_type', 'task_type'),
    )


class HealthChecks(Base):
    """System health check results."""
    __tablename__ = 'health_checks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    check_name = Column(String(100), nullable=False)
    check_type = Column(String(50), nullable=False)  # 'database', 'api', 'disk', 'memory', etc.
    status = Column(String(20), nullable=False)  # 'healthy', 'warning', 'critical', 'unknown'
    
    # Check results
    response_time_ms = Column(Float)
    error_message = Column(Text)
    details = Column(Text)  # JSON string with additional details
    
    # Thresholds
    warning_threshold = Column(Float)
    critical_threshold = Column(Float)
    
    # Timestamps
    check_timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_health_check_name', 'check_name'),
        Index('ix_health_check_type', 'check_type'),
        Index('ix_health_check_status', 'status'),
        Index('ix_health_check_timestamp', 'check_timestamp'),
    )


class Alerts(Base):
    """System alerts and notifications."""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False)  # 'health', 'performance', 'data_quality', etc.
    severity = Column(String(20), nullable=False)  # 'info', 'warning', 'error', 'critical'
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert context
    source_component = Column(String(100))  # Component that generated the alert
    source_details = Column(Text)  # JSON string with source-specific details
    
    # Alert status
    status = Column(String(20), default='active')  # 'active', 'acknowledged', 'resolved'
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    resolved_at = Column(DateTime)
    
    # Notification status
    notification_sent = Column(Boolean, default=False)
    notification_channels = Column(String(255))  # Comma-separated list of channels
    notification_attempts = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_alert_type', 'alert_type'),
        Index('ix_alert_severity', 'severity'),
        Index('ix_alert_status', 'status'),
        Index('ix_alert_created', 'created_at'),
        Index('ix_alert_component', 'source_component'),
    )


class PerformanceMetrics(Base):
    """System performance metrics."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # 'counter', 'gauge', 'histogram', 'timer'
    metric_value = Column(Float, nullable=False)
    
    # Metric context
    component = Column(String(100))  # Component that generated the metric
    tags = Column(Text)  # JSON string with metric tags
    
    # Aggregation data
    aggregation_period = Column(String(20))  # 'minute', 'hour', 'day'
    aggregation_function = Column(String(20))  # 'avg', 'sum', 'min', 'max', 'count'
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_metric_name', 'metric_name'),
        Index('ix_metric_type', 'metric_type'),
        Index('ix_metric_component', 'component'),
        Index('ix_metric_timestamp', 'timestamp'),
        Index('ix_metric_name_timestamp', 'metric_name', 'timestamp'),
    )


class ScheduledJobs(Base):
    """Scheduled job definitions and execution history."""
    __tablename__ = 'scheduled_jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_name = Column(String(100), unique=True, nullable=False)
    job_type = Column(String(50), nullable=False)  # 'feature_computation', 'training', 'monitoring', etc.
    cron_expression = Column(String(100), nullable=False)
    
    # Job configuration
    is_enabled = Column(Boolean, default=True)
    job_parameters = Column(Text)  # JSON string with job-specific parameters
    timeout_seconds = Column(Integer, default=3600)
    max_retries = Column(Integer, default=3)
    
    # Execution tracking
    last_run_at = Column(DateTime)
    last_run_status = Column(String(20))  # 'success', 'failed', 'timeout', 'cancelled'
    last_run_duration_seconds = Column(Float)
    last_run_error = Column(Text)
    next_run_at = Column(DateTime)
    
    # Statistics
    total_runs = Column(Integer, default=0)
    successful_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    average_duration_seconds = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_job_name', 'job_name'),
        Index('ix_job_type', 'job_type'),
        Index('ix_job_enabled', 'is_enabled'),
        Index('ix_job_next_run', 'next_run_at'),
        Index('ix_job_last_run', 'last_run_at'),
    )


class WeatherData(Base):
    """Weather data from Open-Meteo API for match locations."""
    __tablename__ = 'weather_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    
    # Location data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Weather timestamp (match time or forecast time)
    weather_datetime = Column(DateTime, nullable=False)
    data_type = Column(String(20), nullable=False)  # 'historical', 'forecast'
    
    # Temperature data (°C)
    temperature_2m = Column(Float)  # Air temperature at 2m above ground
    apparent_temperature = Column(Float)  # Feels-like temperature
    dew_point_2m = Column(Float)  # Dew point temperature
    
    # Humidity and pressure
    relative_humidity_2m = Column(Float)  # Relative humidity (%)
    pressure_msl = Column(Float)  # Atmospheric pressure (hPa)
    
    # Wind data
    wind_speed_10m = Column(Float)  # Wind speed at 10m (km/h)
    wind_direction_10m = Column(Float)  # Wind direction (degrees)
    wind_gusts_10m = Column(Float)  # Wind gusts (km/h)
    
    # Precipitation data
    precipitation = Column(Float)  # Total precipitation (mm)
    rain = Column(Float)  # Rain (mm)
    snowfall = Column(Float)  # Snowfall (cm)
    precipitation_probability = Column(Float)  # Precipitation probability (%)
    
    # Cloud cover and visibility
    cloud_cover = Column(Float)  # Total cloud cover (%)
    visibility = Column(Float)  # Visibility (meters)
    
    # Weather condition
    weather_code = Column(Integer)  # WMO weather code
    is_day = Column(Boolean)  # 1 if daylight, 0 at night
    
    # Solar radiation (for outdoor conditions)
    shortwave_radiation = Column(Float)  # Solar radiation (W/m²)
    
    # Data quality and metadata
    data_source = Column(String(50), default='open-meteo')  # Data source
    api_response_time_ms = Column(Float)  # API response time
    fetch_timestamp = Column(DateTime, default=datetime.utcnow)  # When data was fetched
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fixture = relationship("Fixture", back_populates="weather_data")
    
    __table_args__ = (
        Index('ix_weather_fixture', 'fixture_id'),
        Index('ix_weather_datetime', 'weather_datetime'),
        Index('ix_weather_location', 'latitude', 'longitude'),
        Index('ix_weather_type', 'data_type'),
        Index('ix_weather_fetch_time', 'fetch_timestamp'),
        UniqueConstraint('fixture_id', 'weather_datetime', 'data_type', 
                        name='uq_weather_fixture_datetime_type'),
    )


class WeatherForecast(Base):
    """Weather forecast data for upcoming matches."""
    __tablename__ = 'weather_forecasts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    
    # Location data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Forecast metadata
    forecast_datetime = Column(DateTime, nullable=False)  # When forecast is for
    forecast_generated_at = Column(DateTime, nullable=False)  # When forecast was generated
    forecast_horizon_hours = Column(Integer)  # Hours ahead of match
    
    # Aggregated weather conditions for match period
    avg_temperature = Column(Float)  # Average temperature during match
    min_temperature = Column(Float)  # Minimum temperature
    max_temperature = Column(Float)  # Maximum temperature
    avg_humidity = Column(Float)  # Average humidity
    avg_wind_speed = Column(Float)  # Average wind speed
    max_wind_speed = Column(Float)  # Maximum wind speed
    total_precipitation = Column(Float)  # Total precipitation during match
    avg_cloud_cover = Column(Float)  # Average cloud cover
    min_visibility = Column(Float)  # Minimum visibility
    
    # Weather condition summary
    dominant_weather_code = Column(Integer)  # Most common weather code
    precipitation_risk = Column(String(20))  # 'low', 'medium', 'high'
    wind_conditions = Column(String(20))  # 'calm', 'moderate', 'strong'
    overall_conditions = Column(String(30))  # 'excellent', 'good', 'fair', 'poor'
    
    # Impact assessment
    playing_conditions_score = Column(Float)  # 0-10 scale for playing conditions
    weather_impact_level = Column(String(20))  # 'minimal', 'moderate', 'significant'
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fixture = relationship("Fixture", back_populates="weather_forecasts")
    
    __table_args__ = (
        Index('ix_forecast_fixture', 'fixture_id'),
        Index('ix_forecast_datetime', 'forecast_datetime'),
        Index('ix_forecast_generated', 'forecast_generated_at'),
        Index('ix_forecast_horizon', 'forecast_horizon_hours'),
        Index('ix_forecast_conditions', 'overall_conditions'),
        UniqueConstraint('fixture_id', 'forecast_generated_at', 
                        name='uq_forecast_fixture_generated'),
    )


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            config = get_config()
            database_url = config.get_database_url()
        
        self.database_url = database_url
        
        # Configure engine based on database type
        if database_url.startswith('sqlite'):
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={'check_same_thread': False},
                echo=False
            )
        else:
            self.engine = create_engine(database_url, echo=False)
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Database manager initialized with URL: {database_url}")    
    def create_materialized_view(self):
        """Create the materialized view for teams."""
        create_view_sql = text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS team_view AS
            SELECT 
                t.id,
                t.name,
                t.short_name,
                t.logo_url,
                l.name AS league_name,
                l.country AS league_country,
                l.season AS league_season,
                COUNT(CASE WHEN f.home_team_id = t.id THEN 1 END) + COUNT(CASE WHEN f.away_team_id = t.id THEN 1 END) AS total_matches,
                COUNT(CASE WHEN f.home_team_id = t.id AND f.home_score > f.away_score THEN 1 
                          WHEN f.away_team_id = t.id AND f.away_score > f.home_score THEN 1 END) AS wins,
                COUNT(CASE WHEN f.home_score = f.away_score THEN 1 END) AS draws,
                COUNT(CASE WHEN f.home_team_id = t.id AND f.home_score < f.away_score THEN 1 
                          WHEN f.away_team_id = t.id AND f.away_score < f.home_score THEN 1 END) AS losses
            FROM teams t
            JOIN leagues l ON t.league_id = l.league_pk
            LEFT JOIN fixtures f ON (t.id = f.home_team_id OR t.id = f.away_team_id) AND f.status = 'finished'
            GROUP BY t.id, l.name, l.country, l.season;
        """)
        with self.engine.connect() as conn:
            conn.execute(create_view_sql)
            conn.commit()
        logger.info("Materialized view 'team_view' created successfully")
    
    def refresh_materialized_view(self):
        """Refresh the materialized view."""
        refresh_sql = text("REFRESH MATERIALIZED VIEW team_view;")
        with self.engine.connect() as conn:
            conn.execute(refresh_sql)
            conn.commit()
        logger.info("Materialized view 'team_view' refreshed successfully")
    
    def create_tables(self):
        """Create all database tables and materialized views."""
        Base.metadata.create_all(bind=self.engine)
        if not self.database_url.startswith('sqlite'):
            self.create_materialized_view()
        logger.info("Database tables and views created successfully")
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped successfully")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close the database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

# Global engine for backward compatibility
engine = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager, engine
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
        engine = _db_manager.engine
    return _db_manager


@contextmanager
def get_db_session():
    """Get a new database session with proper cleanup."""
    session = get_db_manager().get_session()
    try:
        yield session
    finally:
        session.close()


def init_database(database_url: Optional[str] = None, recreate: bool = False):
    """Initialize the database."""
    global _db_manager
    
    if _db_manager is not None:
        _db_manager.close()
    
    _db_manager = DatabaseManager(database_url)
    
    if recreate:
        _db_manager.drop_tables()
    
    _db_manager.create_tables()
    logger.info("Database initialized successfully")


def close_database():
    """Close the database connection."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None