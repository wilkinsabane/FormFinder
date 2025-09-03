"""RSS Monitoring and Metrics System

This module provides comprehensive monitoring for RSS feeds including health checks,
update frequency tracking, content quality metrics, and sentiment analysis accuracy.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class FeedStatus(Enum):
    """RSS feed status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


class ContentQuality(Enum):
    """Content quality levels."""
    EXCELLENT = "excellent"  # 0.8-1.0
    GOOD = "good"           # 0.6-0.8
    FAIR = "fair"           # 0.4-0.6
    POOR = "poor"           # 0.0-0.4


@dataclass
class FeedHealthMetrics:
    """Health metrics for an RSS feed."""
    feed_url: str
    feed_name: str
    sport: str
    status: FeedStatus
    last_check: datetime
    last_successful_fetch: Optional[datetime]
    last_update: Optional[datetime]
    consecutive_failures: int
    total_checks: int
    success_rate: float
    avg_response_time: float
    articles_fetched_24h: int
    articles_fetched_7d: int
    error_message: Optional[str] = None


@dataclass
class ContentQualityMetrics:
    """Content quality metrics."""
    feed_url: str
    period_start: datetime
    period_end: datetime
    total_articles: int
    avg_quality_score: float
    quality_distribution: Dict[ContentQuality, int]
    duplicate_rate: float
    avg_content_length: float
    team_match_rate: float
    sentiment_coverage: float


@dataclass
class SentimentAccuracyMetrics:
    """Sentiment analysis accuracy metrics."""
    period_start: datetime
    period_end: datetime
    total_analyses: int
    avg_confidence: float
    sentiment_distribution: Dict[str, int]
    accuracy_by_source: Dict[str, float]
    processing_time_avg: float
    error_rate: float


class RSSMonitor:
    """Comprehensive RSS monitoring system."""

    def __init__(self, db_path: str = "rss_monitoring.db", 
                 retention_days: int = 90):
        """
        Initialize RSS monitoring system.
        
        Args:
            db_path: Path to monitoring database
            retention_days: Days to retain monitoring data
        """
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.lock = Lock()
        
        # In-memory caches for performance
        self._feed_metrics_cache: Dict[str, FeedHealthMetrics] = {}
        self._last_cleanup = time.time()
        
        self._init_database()
        self._load_feed_metrics()
        
        logger.info(f"RSS monitoring initialized: {self.db_path}")

    def _init_database(self):
        """Initialize monitoring database."""
        with sqlite3.connect(self.db_path) as conn:
            # Feed health metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feed_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_url TEXT NOT NULL,
                    feed_name TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    status TEXT NOT NULL,
                    check_time TEXT NOT NULL,
                    response_time REAL,
                    articles_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    success BOOLEAN NOT NULL
                )
            """)
            
            # Content quality metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_url TEXT NOT NULL,
                    article_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    content_length INTEGER NOT NULL,
                    team_matches TEXT,  -- JSON array
                    is_duplicate BOOLEAN DEFAULT FALSE,
                    sentiment_score REAL,
                    sentiment_confidence REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Sentiment analysis metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    source_type TEXT NOT NULL,  -- 'rss', 'api', etc.
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_sentiment REAL,
                    away_sentiment REAL,
                    home_confidence REAL,
                    away_confidence REAL,
                    articles_analyzed INTEGER NOT NULL,
                    processing_time REAL NOT NULL,
                    error_occurred BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feed_health_url ON feed_health(feed_url)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feed_health_time ON feed_health(check_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_quality_url ON content_quality(feed_url)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_quality_time ON content_quality(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_time ON sentiment_metrics(created_at)")
            
            conn.commit()

    def _load_feed_metrics(self):
        """Load current feed metrics into cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get latest metrics for each feed
                cursor = conn.execute("""
                    SELECT 
                        feed_url,
                        feed_name,
                        sport,
                        status,
                        MAX(check_time) as last_check,
                        COUNT(*) as total_checks,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(response_time) as avg_response_time,
                        SUM(CASE WHEN success AND check_time >= datetime('now', '-1 day') 
                            THEN articles_count ELSE 0 END) as articles_24h,
                        SUM(CASE WHEN success AND check_time >= datetime('now', '-7 days') 
                            THEN articles_count ELSE 0 END) as articles_7d
                    FROM feed_health 
                    WHERE check_time >= datetime('now', '-30 days')
                    GROUP BY feed_url, feed_name, sport
                """)
                
                for row in cursor.fetchall():
                    # Get consecutive failures
                    fail_cursor = conn.execute("""
                        SELECT COUNT(*) FROM feed_health 
                        WHERE feed_url = ? AND success = FALSE 
                        AND id > COALESCE((
                            SELECT MAX(id) FROM feed_health 
                            WHERE feed_url = ? AND success = TRUE
                        ), 0)
                    """, (row['feed_url'], row['feed_url']))
                    
                    consecutive_failures = fail_cursor.fetchone()[0]
                    
                    # Get last successful fetch
                    success_cursor = conn.execute("""
                        SELECT MAX(check_time) FROM feed_health 
                        WHERE feed_url = ? AND success = TRUE
                    """, (row['feed_url'],))
                    
                    last_success = success_cursor.fetchone()[0]
                    
                    metrics = FeedHealthMetrics(
                        feed_url=row['feed_url'],
                        feed_name=row['feed_name'],
                        sport=row['sport'],
                        status=FeedStatus(row['status']),
                        last_check=datetime.fromisoformat(row['last_check']),
                        last_successful_fetch=datetime.fromisoformat(last_success) if last_success else None,
                        last_update=None,  # Will be updated during monitoring
                        consecutive_failures=consecutive_failures,
                        total_checks=row['total_checks'],
                        success_rate=row['success_rate'] or 0.0,
                        avg_response_time=row['avg_response_time'] or 0.0,
                        articles_fetched_24h=row['articles_24h'] or 0,
                        articles_fetched_7d=row['articles_7d'] or 0
                    )
                    
                    self._feed_metrics_cache[row['feed_url']] = metrics
                    
            logger.debug(f"Loaded metrics for {len(self._feed_metrics_cache)} feeds")
            
        except Exception as e:
            logger.error(f"Failed to load feed metrics: {e}")

    def record_feed_check(self, feed_url: str, feed_name: str, sport: str,
                         success: bool, response_time: float, articles_count: int = 0,
                         error_message: Optional[str] = None):
        """Record a feed health check."""
        try:
            now = datetime.now(timezone.utc)
            
            # Determine status
            if success:
                status = FeedStatus.HEALTHY
            else:
                # Check consecutive failures to determine severity
                consecutive_failures = self._get_consecutive_failures(feed_url) + 1
                if consecutive_failures >= 5:
                    status = FeedStatus.ERROR
                else:
                    status = FeedStatus.WARNING
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO feed_health (
                            feed_url, feed_name, sport, status, check_time,
                            response_time, articles_count, error_message, success
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        feed_url, feed_name, sport, status.value, now.isoformat(),
                        response_time, articles_count, error_message, success
                    ))
                    conn.commit()
                
                # Update cache
                if feed_url in self._feed_metrics_cache:
                    metrics = self._feed_metrics_cache[feed_url]
                    metrics.last_check = now
                    if success:
                        metrics.last_successful_fetch = now
                        metrics.consecutive_failures = 0
                    else:
                        metrics.consecutive_failures += 1
                    metrics.status = status
                    metrics.total_checks += 1
                else:
                    # Create new metrics entry
                    metrics = FeedHealthMetrics(
                        feed_url=feed_url,
                        feed_name=feed_name,
                        sport=sport,
                        status=status,
                        last_check=now,
                        last_successful_fetch=now if success else None,
                        last_update=None,
                        consecutive_failures=0 if success else 1,
                        total_checks=1,
                        success_rate=1.0 if success else 0.0,
                        avg_response_time=response_time,
                        articles_fetched_24h=articles_count if success else 0,
                        articles_fetched_7d=articles_count if success else 0
                    )
                    self._feed_metrics_cache[feed_url] = metrics
            
            logger.debug(f"Recorded feed check: {feed_url} - {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to record feed check: {e}")

    def record_content_quality(self, feed_url: str, article_url: str, title: str,
                             quality_score: float, content_length: int,
                             team_matches: List[str], is_duplicate: bool = False,
                             sentiment_score: Optional[float] = None,
                             sentiment_confidence: Optional[float] = None):
        """Record content quality metrics."""
        try:
            now = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO content_quality (
                        feed_url, article_url, title, quality_score, content_length,
                        team_matches, is_duplicate, sentiment_score, sentiment_confidence,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feed_url, article_url, title, quality_score, content_length,
                    json.dumps(team_matches), is_duplicate, sentiment_score,
                    sentiment_confidence, now.isoformat()
                ))
                conn.commit()
            
            logger.debug(f"Recorded content quality: {title[:50]}... - {quality_score:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to record content quality: {e}")

    def record_sentiment_analysis(self, query: str, source_type: str, home_team: str,
                                away_team: str, home_sentiment: Optional[float],
                                away_sentiment: Optional[float], home_confidence: Optional[float],
                                away_confidence: Optional[float], articles_analyzed: int,
                                processing_time: float, error_occurred: bool = False,
                                error_message: Optional[str] = None):
        """Record sentiment analysis metrics."""
        try:
            now = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sentiment_metrics (
                        query, source_type, home_team, away_team, home_sentiment,
                        away_sentiment, home_confidence, away_confidence,
                        articles_analyzed, processing_time, error_occurred,
                        error_message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query, source_type, home_team, away_team, home_sentiment,
                    away_sentiment, home_confidence, away_confidence,
                    articles_analyzed, processing_time, error_occurred,
                    error_message, now.isoformat()
                ))
                conn.commit()
            
            logger.debug(f"Recorded sentiment analysis: {query} - {source_type}")
            
        except Exception as e:
            logger.error(f"Failed to record sentiment analysis: {e}")

    def _get_consecutive_failures(self, feed_url: str) -> int:
        """Get consecutive failures for a feed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM feed_health 
                    WHERE feed_url = ? AND success = FALSE 
                    AND id > COALESCE((
                        SELECT MAX(id) FROM feed_health 
                        WHERE feed_url = ? AND success = TRUE
                    ), 0)
                """, (feed_url, feed_url))
                
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get consecutive failures: {e}")
            return 0

    def get_feed_health_summary(self) -> Dict[str, Any]:
        """Get overall feed health summary."""
        try:
            total_feeds = len(self._feed_metrics_cache)
            if total_feeds == 0:
                return {
                    'total_feeds': 0,
                    'healthy_feeds': 0,
                    'warning_feeds': 0,
                    'error_feeds': 0,
                    'overall_health': 'unknown',
                    'avg_success_rate': 0.0,
                    'total_articles_24h': 0,
                    'feeds_by_sport': {}
                }
            
            healthy = sum(1 for m in self._feed_metrics_cache.values() 
                         if m.status == FeedStatus.HEALTHY)
            warning = sum(1 for m in self._feed_metrics_cache.values() 
                         if m.status == FeedStatus.WARNING)
            error = sum(1 for m in self._feed_metrics_cache.values() 
                       if m.status == FeedStatus.ERROR)
            
            avg_success_rate = sum(m.success_rate for m in self._feed_metrics_cache.values()) / total_feeds
            total_articles_24h = sum(m.articles_fetched_24h for m in self._feed_metrics_cache.values())
            
            # Determine overall health
            if error > total_feeds * 0.3:  # More than 30% in error
                overall_health = 'critical'
            elif warning + error > total_feeds * 0.5:  # More than 50% with issues
                overall_health = 'warning'
            elif healthy > total_feeds * 0.8:  # More than 80% healthy
                overall_health = 'excellent'
            else:
                overall_health = 'good'
            
            # Group by sport
            feeds_by_sport = {}
            for metrics in self._feed_metrics_cache.values():
                sport = metrics.sport
                if sport not in feeds_by_sport:
                    feeds_by_sport[sport] = {'total': 0, 'healthy': 0, 'warning': 0, 'error': 0}
                
                feeds_by_sport[sport]['total'] += 1
                if metrics.status == FeedStatus.HEALTHY:
                    feeds_by_sport[sport]['healthy'] += 1
                elif metrics.status == FeedStatus.WARNING:
                    feeds_by_sport[sport]['warning'] += 1
                elif metrics.status == FeedStatus.ERROR:
                    feeds_by_sport[sport]['error'] += 1
            
            return {
                'total_feeds': total_feeds,
                'healthy_feeds': healthy,
                'warning_feeds': warning,
                'error_feeds': error,
                'overall_health': overall_health,
                'avg_success_rate': avg_success_rate,
                'total_articles_24h': total_articles_24h,
                'feeds_by_sport': feeds_by_sport,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get feed health summary: {e}")
            return {}

    def get_content_quality_report(self, hours_back: int = 24) -> ContentQualityMetrics:
        """Get content quality report for specified time period."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_articles,
                        AVG(quality_score) as avg_quality,
                        AVG(content_length) as avg_length,
                        SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) as duplicates,
                        SUM(CASE WHEN team_matches != '[]' THEN 1 ELSE 0 END) as team_matches,
                        SUM(CASE WHEN sentiment_score IS NOT NULL THEN 1 ELSE 0 END) as sentiment_coverage
                    FROM content_quality 
                    WHERE created_at >= ?
                """, (cutoff_time.isoformat(),))
                
                row = cursor.fetchone()
                
                if not row or row['total_articles'] == 0:
                    return ContentQualityMetrics(
                        feed_url="all",
                        period_start=cutoff_time,
                        period_end=datetime.now(timezone.utc),
                        total_articles=0,
                        avg_quality_score=0.0,
                        quality_distribution={q: 0 for q in ContentQuality},
                        duplicate_rate=0.0,
                        avg_content_length=0.0,
                        team_match_rate=0.0,
                        sentiment_coverage=0.0
                    )
                
                total = row['total_articles']
                
                # Get quality distribution
                quality_cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN quality_score >= 0.8 THEN 'excellent'
                            WHEN quality_score >= 0.6 THEN 'good'
                            WHEN quality_score >= 0.4 THEN 'fair'
                            ELSE 'poor'
                        END as quality_level,
                        COUNT(*) as count
                    FROM content_quality 
                    WHERE created_at >= ?
                    GROUP BY quality_level
                """, (cutoff_time.isoformat(),))
                
                quality_dist = {q: 0 for q in ContentQuality}
                for q_row in quality_cursor.fetchall():
                    quality_dist[ContentQuality(q_row['quality_level'])] = q_row['count']
                
                return ContentQualityMetrics(
                    feed_url="all",
                    period_start=cutoff_time,
                    period_end=datetime.now(timezone.utc),
                    total_articles=total,
                    avg_quality_score=row['avg_quality'] or 0.0,
                    quality_distribution=quality_dist,
                    duplicate_rate=(row['duplicates'] or 0) / total,
                    avg_content_length=row['avg_length'] or 0.0,
                    team_match_rate=(row['team_matches'] or 0) / total,
                    sentiment_coverage=(row['sentiment_coverage'] or 0) / total
                )
                
        except Exception as e:
            logger.error(f"Failed to get content quality report: {e}")
            return ContentQualityMetrics(
                feed_url="all",
                period_start=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc),
                total_articles=0,
                avg_quality_score=0.0,
                quality_distribution={q: 0 for q in ContentQuality},
                duplicate_rate=0.0,
                avg_content_length=0.0,
                team_match_rate=0.0,
                sentiment_coverage=0.0
            )

    def get_sentiment_accuracy_report(self, hours_back: int = 24) -> SentimentAccuracyMetrics:
        """Get sentiment analysis accuracy report."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_analyses,
                        AVG((home_confidence + away_confidence) / 2) as avg_confidence,
                        AVG(processing_time) as avg_processing_time,
                        SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as errors
                    FROM sentiment_metrics 
                    WHERE created_at >= ?
                """, (cutoff_time.isoformat(),))
                
                row = cursor.fetchone()
                
                if not row or row['total_analyses'] == 0:
                    return SentimentAccuracyMetrics(
                        period_start=cutoff_time,
                        period_end=datetime.now(timezone.utc),
                        total_analyses=0,
                        avg_confidence=0.0,
                        sentiment_distribution={},
                        accuracy_by_source={},
                        processing_time_avg=0.0,
                        error_rate=0.0
                    )
                
                total = row['total_analyses']
                
                # Get sentiment distribution
                sentiment_cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN home_sentiment > 0.1 THEN 'positive'
                            WHEN home_sentiment < -0.1 THEN 'negative'
                            ELSE 'neutral'
                        END as sentiment,
                        COUNT(*) as count
                    FROM sentiment_metrics 
                    WHERE created_at >= ? AND home_sentiment IS NOT NULL
                    GROUP BY sentiment
                """, (cutoff_time.isoformat(),))
                
                sentiment_dist = {}
                for s_row in sentiment_cursor.fetchall():
                    sentiment_dist[s_row['sentiment']] = s_row['count']
                
                # Get accuracy by source
                source_cursor = conn.execute("""
                    SELECT 
                        source_type,
                        AVG((home_confidence + away_confidence) / 2) as avg_confidence
                    FROM sentiment_metrics 
                    WHERE created_at >= ? AND home_confidence IS NOT NULL
                    GROUP BY source_type
                """, (cutoff_time.isoformat(),))
                
                accuracy_by_source = {}
                for src_row in source_cursor.fetchall():
                    accuracy_by_source[src_row['source_type']] = src_row['avg_confidence'] or 0.0
                
                return SentimentAccuracyMetrics(
                    period_start=cutoff_time,
                    period_end=datetime.now(timezone.utc),
                    total_analyses=total,
                    avg_confidence=row['avg_confidence'] or 0.0,
                    sentiment_distribution=sentiment_dist,
                    accuracy_by_source=accuracy_by_source,
                    processing_time_avg=row['avg_processing_time'] or 0.0,
                    error_rate=(row['errors'] or 0) / total
                )
                
        except Exception as e:
            logger.error(f"Failed to get sentiment accuracy report: {e}")
            return SentimentAccuracyMetrics(
                period_start=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc),
                total_analyses=0,
                avg_confidence=0.0,
                sentiment_distribution={},
                accuracy_by_source={},
                processing_time_avg=0.0,
                error_rate=0.0
            )

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        try:
            health_summary = self.get_feed_health_summary()
            quality_report = self.get_content_quality_report(hours_back=24)
            sentiment_report = self.get_sentiment_accuracy_report(hours_back=24)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'feed_health': health_summary,
                'content_quality': asdict(quality_report),
                'sentiment_accuracy': asdict(sentiment_report),
                'system_status': {
                    'monitoring_active': True,
                    'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
                    'retention_days': self.retention_days
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring dashboard: {e}")
            return {}

    def cleanup_old_data(self):
        """Clean up old monitoring data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Clean up old feed health records
                    cursor = conn.execute("""
                        DELETE FROM feed_health WHERE check_time < ?
                    """, (cutoff_date.isoformat(),))
                    health_deleted = cursor.rowcount
                    
                    # Clean up old content quality records
                    cursor = conn.execute("""
                        DELETE FROM content_quality WHERE created_at < ?
                    """, (cutoff_date.isoformat(),))
                    quality_deleted = cursor.rowcount
                    
                    # Clean up old sentiment metrics
                    cursor = conn.execute("""
                        DELETE FROM sentiment_metrics WHERE created_at < ?
                    """, (cutoff_date.isoformat(),))
                    sentiment_deleted = cursor.rowcount
                    
                    conn.commit()
            
            logger.info(f"Cleaned up monitoring data: {health_deleted} health, "
                       f"{quality_deleted} quality, {sentiment_deleted} sentiment records")
            
        except Exception as e:
            logger.error(f"Failed to cleanup monitoring data: {e}")


def create_default_rss_monitor() -> RSSMonitor:
    """Create RSS monitor with default settings."""
    return RSSMonitor(
        db_path="rss_monitoring.db",
        retention_days=90
    )