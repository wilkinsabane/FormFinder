"""PostgreSQL models for RSS monitoring and caching system.

This module defines SQLAlchemy models for RSS feed monitoring, content caching,
and performance tracking that integrate with the existing PostgreSQL database.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    Index, UniqueConstraint, ForeignKey
)
from sqlalchemy.orm import relationship

from .database import Base

logger = logging.getLogger(__name__)


class RSSFeedHealth(Base):
    """RSS feed health monitoring data."""
    __tablename__ = 'rss_feed_health'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    feed_url = Column(String(500), nullable=False)
    check_time = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    is_successful = Column(Boolean, nullable=False)
    response_time = Column(Float)  # in seconds
    articles_count = Column(Integer)
    error_message = Column(Text)
    http_status_code = Column(Integer)
    
    __table_args__ = (
        Index('ix_rss_feed_health_url_time', 'feed_url', 'check_time'),
        Index('ix_rss_feed_health_check_time', 'check_time'),
    )


class RSSContentQuality(Base):
    """RSS content quality metrics."""
    __tablename__ = 'rss_content_quality'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    feed_url = Column(String(500), nullable=False)
    article_url = Column(String(1000), nullable=False)
    recorded_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    quality_score = Column(Float, nullable=False)
    team_matches = Column(Integer, nullable=False, default=0)
    is_duplicate = Column(Boolean, nullable=False, default=False)
    content_length = Column(Integer)
    has_image = Column(Boolean, default=False)
    source_name = Column(String(200))
    
    __table_args__ = (
        Index('ix_rss_content_quality_feed_time', 'feed_url', 'recorded_at'),
        Index('ix_rss_content_quality_recorded_at', 'recorded_at'),
        UniqueConstraint('article_url', 'recorded_at', name='uq_rss_content_quality_article_time'),
    )


class RSSSentimentAnalysis(Base):
    """RSS sentiment analysis performance tracking."""
    __tablename__ = 'rss_sentiment_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String(200), nullable=False)
    analysis_time = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    source_type = Column(String(50), nullable=False)  # 'rss', 'api', 'hybrid'
    home_team = Column(String(100))
    away_team = Column(String(100))
    home_sentiment = Column(Float)
    away_sentiment = Column(Float)
    home_confidence = Column(Float)
    away_confidence = Column(Float)
    articles_analyzed = Column(Integer, nullable=False, default=0)
    processing_time = Column(Float)  # in seconds
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text)
    
    __table_args__ = (
        Index('ix_rss_sentiment_query_time', 'query', 'analysis_time'),
        Index('ix_rss_sentiment_analysis_time', 'analysis_time'),
        Index('ix_rss_sentiment_source_type', 'source_type'),
    )


class RSSCachedArticle(Base):
    """Cached RSS articles with PostgreSQL storage."""
    __tablename__ = 'rss_cached_articles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(1000), nullable=False, unique=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    content = Column(Text)
    published_date = Column(DateTime)
    source = Column(String(200))
    feed_url = Column(String(500), nullable=False)
    teams = Column(JSON)  # List of team names
    sport = Column(String(50))
    content_hash = Column(String(64), nullable=False, unique=True)
    quality_score = Column(Float, default=0.0)
    cached_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    last_accessed = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    access_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('ix_rss_cached_articles_url', 'url'),
        Index('ix_rss_cached_articles_hash', 'content_hash'),
        Index('ix_rss_cached_articles_feed_url', 'feed_url'),
        Index('ix_rss_cached_articles_cached_at', 'cached_at'),
        Index('ix_rss_cached_articles_teams', 'teams'),
        Index('ix_rss_cached_articles_sport', 'sport'),
    )


# Dataclasses for compatibility with existing code
@dataclass
class FeedStatus:
    """Feed health status data."""
    feed_url: str
    is_successful: bool
    response_time: float
    articles_count: int
    error_message: Optional[str] = None
    check_time: Optional[datetime] = None


@dataclass
class ContentQuality:
    """Content quality metrics data."""
    feed_url: str
    article_url: str
    quality_score: float
    team_matches: int
    is_duplicate: bool
    content_length: Optional[int] = None
    has_image: bool = False
    source_name: Optional[str] = None
    recorded_at: Optional[datetime] = None


@dataclass
class SentimentAccuracy:
    """Sentiment analysis accuracy data."""
    query: str
    source_type: str
    home_team: Optional[str]
    away_team: Optional[str]
    home_sentiment: Optional[float]
    away_sentiment: Optional[float]
    home_confidence: Optional[float]
    away_confidence: Optional[float]
    articles_analyzed: int
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None
    analysis_time: Optional[datetime] = None


@dataclass
class CachedArticle:
    """Cached article data."""
    url: str
    title: str
    description: str
    content: str
    published_date: datetime
    source: str
    feed_url: str
    teams: List[str]
    sport: str
    content_hash: str
    quality_score: float
    cached_at: datetime
    last_accessed: datetime
    access_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility."""
        return {
            'url': self.url,
            'title': self.title,
            'description': self.description,
            'content': self.content,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'source': self.source,
            'feed_url': self.feed_url,
            'teams': self.teams,
            'sport': self.sport,
            'content_hash': self.content_hash,
            'quality_score': self.quality_score,
            'cached_at': self.cached_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CachedArticle':
        """Create from dictionary."""
        return cls(
            url=data['url'],
            title=data['title'],
            description=data['description'],
            content=data['content'],
            published_date=datetime.fromisoformat(data['published_date']) if data['published_date'] else None,
            source=data['source'],
            feed_url=data['feed_url'],
            teams=data['teams'],
            sport=data['sport'],
            content_hash=data['content_hash'],
            quality_score=data['quality_score'],
            cached_at=datetime.fromisoformat(data['cached_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data.get('access_count', 0)
        )