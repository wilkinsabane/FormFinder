from sqlalchemy import create_engine, Column, Integer, String, DateTime, Date, Boolean, Text, DECIMAL, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    short_code = Column(String(10))
    country = Column(String(100))
    logo_url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sentiment_data = relationship("SentimentData", back_populates="team")
    sentiment_scores = relationship("TeamSentimentScore", back_populates="team")
    form_analysis = relationship("TeamFormAnalysis", back_populates="team")
    combined_analysis = relationship("CombinedAnalysis", back_populates="team")

class League(Base):
    __tablename__ = 'leagues'
    
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    country = Column(String(100))
    season = Column(String(20))
    logo_url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    matches = relationship("Match", back_populates="league")
    form_analysis = relationship("TeamFormAnalysis", back_populates="league")
    combined_analysis = relationship("CombinedAnalysis", back_populates="league")

class Match(Base):
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, unique=True, nullable=False)
    league_id = Column(Integer, ForeignKey('leagues.league_id'))
    home_team_id = Column(Integer, ForeignKey('teams.team_id'))
    away_team_id = Column(Integer, ForeignKey('teams.team_id'))
    match_date = Column(DateTime, nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    status = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

class SentimentSource(Base):
    __tablename__ = 'sentiment_sources'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String(100), unique=True, nullable=False)
    source_type = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sentiment_data = relationship("SentimentData", back_populates="source")

class SentimentData(Base):
    __tablename__ = 'sentiment_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    source_id = Column(Integer, ForeignKey('sentiment_sources.id'))
    content = Column(Text, nullable=False)
    sentiment_score = Column(DECIMAL(3, 2))
    confidence_score = Column(DECIMAL(3, 2))
    sentiment_label = Column(String(20))
    author = Column(String(255))
    post_date = Column(DateTime)
    url = Column(Text)
    raw_data = Column(JSON)
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="sentiment_data")
    source = relationship("SentimentSource", back_populates="sentiment_data")

class TeamSentimentScore(Base):
    __tablename__ = 'team_sentiment_scores'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    date = Column(Date, nullable=False)
    avg_sentiment_score = Column(DECIMAL(3, 2))
    sentiment_volume = Column(Integer, default=0)
    positive_ratio = Column(DECIMAL(3, 2))
    negative_ratio = Column(DECIMAL(3, 2))
    neutral_ratio = Column(DECIMAL(3, 2))
    data_points = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="sentiment_scores")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )

class TeamFormAnalysis(Base):
    __tablename__ = 'team_form_analysis'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    league_id = Column(Integer, ForeignKey('leagues.league_id'))
    analysis_date = Column(Date, nullable=False)
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    win_rate = Column(DECIMAL(3, 2))
    form_score = Column(DECIMAL(3, 2))
    streak = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="form_analysis")
    league = relationship("League", back_populates="form_analysis")

class CombinedAnalysis(Base):
    __tablename__ = 'combined_analysis'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    league_id = Column(Integer, ForeignKey('leagues.league_id'))
    analysis_date = Column(Date, nullable=False)
    sentiment_score = Column(DECIMAL(3, 2))
    form_score = Column(DECIMAL(3, 2))
    combined_score = Column(DECIMAL(3, 2))
    sentiment_weight = Column(DECIMAL(3, 2), default=0.3)
    form_weight = Column(DECIMAL(3, 2), default=0.7)
    prediction_confidence = Column(DECIMAL(3, 2))
    trend_direction = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="combined_analysis")
    league = relationship("League", back_populates="combined_analysis")

class AlertThreshold(Base):
    __tablename__ = 'alert_thresholds'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    threshold_type = Column(String(50), nullable=False)
    min_value = Column(DECIMAL(3, 2))
    max_value = Column(DECIMAL(3, 2))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NotificationLog(Base):
    __tablename__ = 'notification_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    threshold_id = Column(Integer, ForeignKey('alert_thresholds.id'))
    notification_type = Column(String(50))
    message = Column(Text)
    sent_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='sent')

class ProcessingStatus(Base):
    __tablename__ = 'processing_status'
    
    id = Column(Integer, primary_key=True)
    process_name = Column(String(100), unique=True, nullable=False)
    last_run = Column(DateTime)
    status = Column(String(20))
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database connection utilities
class DatabaseManager:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close a database session"""
        session.close()

# Dependency for FastAPI
def get_db():
    """Database dependency for FastAPI"""
    from src.config.database import DatabaseManager
    
    db_manager = DatabaseManager("postgresql://user:password@localhost/formfinder")
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()