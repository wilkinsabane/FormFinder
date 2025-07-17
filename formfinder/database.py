"""Database models and setup for FormFinder."""

import logging
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, Index, UniqueConstraint
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
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    country = Column(String(100))
    season = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    teams = relationship("Team", back_populates="league")
    fixtures = relationship("Fixture", back_populates="league")
    standings = relationship("Standing", back_populates="league")
    
    __table_args__ = (
        UniqueConstraint('id', 'season', name='uq_league_season'),
    )


class Team(Base):
    """Team information."""
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey('leagues.id'), nullable=False)
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
    
    __table_args__ = (
        Index('ix_team_league', 'league_id'),
    )


class Fixture(Base):
    """Match fixtures and results."""
    __tablename__ = 'fixtures'
    
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey('leagues.id'), nullable=False)
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
    
    # Additional match data
    referee = Column(String(255))
    venue = Column(String(255))
    attendance = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    league = relationship("League", back_populates="fixtures")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_fixtures")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_fixtures")
    predictions = relationship("Prediction", back_populates="fixture")
    
    __table_args__ = (
        Index('ix_fixture_league_date', 'league_id', 'match_date'),
        Index('ix_fixture_teams', 'home_team_id', 'away_team_id'),
        Index('ix_fixture_status', 'status'),
    )


class Standing(Base):
    """League standings/table."""
    __tablename__ = 'standings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    league_id = Column(Integer, ForeignKey('leagues.id'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
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
        UniqueConstraint('league_id', 'team_id', name='uq_standing_league_team'),
        Index('ix_standing_league_position', 'league_id', 'position'),
    )


class Prediction(Base):
    """Match predictions."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey('fixtures.id'), nullable=False)
    
    # Prediction data
    home_win_probability = Column(Float)
    draw_probability = Column(Float)
    away_win_probability = Column(Float)
    predicted_home_score = Column(Float)
    predicted_away_score = Column(Float)
    confidence_score = Column(Float)  # 0-1 scale
    
    # Form analysis
    home_team_form_score = Column(Float)
    away_team_form_score = Column(Float)
    home_team_recent_wins = Column(Integer)
    away_team_recent_wins = Column(Integer)
    
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
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
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


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
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