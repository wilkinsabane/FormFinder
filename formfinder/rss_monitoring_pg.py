"""PostgreSQL-based RSS Monitoring System

This module provides comprehensive monitoring and metrics for RSS feeds
using PostgreSQL for data storage, replacing the SQLite implementation.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from sqlalchemy import func, and_, desc
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from .database import get_db_session
from .rss_models import (
    RSSFeedHealth, RSSContentQuality, RSSSentimentAnalysis, RSSCachedArticle,
    FeedStatus, ContentQuality, SentimentAccuracy
)

logger = logging.getLogger(__name__)


class RSSMonitor:
    """PostgreSQL-based RSS monitoring and metrics system."""
    
    def __init__(self):
        """Initialize RSS monitor with PostgreSQL backend."""
        logger.info("Initialized PostgreSQL RSS Monitor")
    
    def record_feed_health(self, status: FeedStatus) -> bool:
        """
        Record RSS feed health check results.
        
        Args:
            status: Feed health status data
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            with get_db_session() as session:
                health_record = RSSFeedHealth(
                    feed_url=status.feed_url,
                    check_time=status.check_time or datetime.now(timezone.utc),
                    is_successful=status.is_successful,
                    response_time=status.response_time,
                    articles_count=status.articles_count,
                    error_message=status.error_message
                )
                
                session.add(health_record)
                session.commit()
                
                logger.debug(f"Recorded feed health for {status.feed_url}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error recording feed health: {e}")
            return False
    
    def record_content_quality(self, quality: ContentQuality) -> bool:
        """
        Record content quality metrics.
        
        Args:
            quality: Content quality data
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            with get_db_session() as session:
                quality_record = RSSContentQuality(
                    feed_url=quality.feed_url,
                    article_url=quality.article_url,
                    recorded_at=quality.recorded_at or datetime.now(timezone.utc),
                    quality_score=quality.quality_score,
                    team_matches=quality.team_matches,
                    is_duplicate=quality.is_duplicate,
                    content_length=quality.content_length,
                    has_image=quality.has_image,
                    source_name=quality.source_name
                )
                
                session.add(quality_record)
                session.commit()
                
                logger.debug(f"Recorded content quality for {quality.article_url}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error recording content quality: {e}")
            return False
    
    def record_sentiment_analysis(self, accuracy: SentimentAccuracy) -> bool:
        """
        Record sentiment analysis performance metrics.
        
        Args:
            accuracy: Sentiment analysis accuracy data
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            with get_db_session() as session:
                sentiment_record = RSSSentimentAnalysis(
                    query=accuracy.query,
                    analysis_time=accuracy.analysis_time or datetime.now(timezone.utc),
                    source_type=accuracy.source_type,
                    home_team=accuracy.home_team,
                    away_team=accuracy.away_team,
                    home_sentiment=accuracy.home_sentiment,
                    away_sentiment=accuracy.away_sentiment,
                    home_confidence=accuracy.home_confidence,
                    away_confidence=accuracy.away_confidence,
                    articles_analyzed=accuracy.articles_analyzed,
                    processing_time=accuracy.processing_time,
                    success=accuracy.success,
                    error_message=accuracy.error_message
                )
                
                session.add(sentiment_record)
                session.commit()
                
                logger.debug(f"Recorded sentiment analysis for query: {accuracy.query}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error recording sentiment analysis: {e}")
            return False
    
    def get_feed_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get feed health summary for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing health summary statistics
        """
        try:
            with get_db_session() as session:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
                
                # Get overall statistics
                total_checks = session.query(RSSFeedHealth).filter(
                    RSSFeedHealth.check_time >= cutoff_time
                ).count()
                
                successful_checks = session.query(RSSFeedHealth).filter(
                    and_(
                        RSSFeedHealth.check_time >= cutoff_time,
                        RSSFeedHealth.is_successful == True
                    )
                ).count()
                
                # Get average response time
                avg_response_time = session.query(
                    func.avg(RSSFeedHealth.response_time)
                ).filter(
                    and_(
                        RSSFeedHealth.check_time >= cutoff_time,
                        RSSFeedHealth.is_successful == True
                    )
                ).scalar() or 0.0
                
                # Get feed-specific statistics
                feed_stats = session.query(
                    RSSFeedHealth.feed_url,
                    func.count(RSSFeedHealth.id).label('total_checks'),
                    func.sum(func.cast(RSSFeedHealth.is_successful, func.Integer)).label('successful_checks'),
                    func.avg(RSSFeedHealth.response_time).label('avg_response_time')
                ).filter(
                    RSSFeedHealth.check_time >= cutoff_time
                ).group_by(RSSFeedHealth.feed_url).all()
                
                feed_health = []
                for stat in feed_stats:
                    success_rate = (stat.successful_checks / stat.total_checks * 100) if stat.total_checks > 0 else 0
                    feed_health.append({
                        'feed_url': stat.feed_url,
                        'total_checks': stat.total_checks,
                        'successful_checks': stat.successful_checks,
                        'success_rate': round(success_rate, 2),
                        'avg_response_time': round(stat.avg_response_time or 0, 3)
                    })
                
                success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
                
                return {
                    'period_hours': hours,
                    'total_checks': total_checks,
                    'successful_checks': successful_checks,
                    'success_rate': round(success_rate, 2),
                    'avg_response_time': round(avg_response_time, 3),
                    'feed_health': feed_health
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting feed health summary: {e}")
            return {
                'period_hours': hours,
                'total_checks': 0,
                'successful_checks': 0,
                'success_rate': 0,
                'avg_response_time': 0,
                'feed_health': []
            }
    
    def get_content_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get content quality report for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing content quality statistics
        """
        try:
            with get_db_session() as session:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
                
                # Get overall statistics
                total_articles = session.query(RSSContentQuality).filter(
                    RSSContentQuality.recorded_at >= cutoff_time
                ).count()
                
                duplicate_articles = session.query(RSSContentQuality).filter(
                    and_(
                        RSSContentQuality.recorded_at >= cutoff_time,
                        RSSContentQuality.is_duplicate == True
                    )
                ).count()
                
                # Get average quality score
                avg_quality = session.query(
                    func.avg(RSSContentQuality.quality_score)
                ).filter(
                    RSSContentQuality.recorded_at >= cutoff_time
                ).scalar() or 0.0
                
                # Get team match statistics
                total_team_matches = session.query(
                    func.sum(RSSContentQuality.team_matches)
                ).filter(
                    RSSContentQuality.recorded_at >= cutoff_time
                ).scalar() or 0
                
                # Get feed-specific statistics
                feed_stats = session.query(
                    RSSContentQuality.feed_url,
                    func.count(RSSContentQuality.id).label('article_count'),
                    func.avg(RSSContentQuality.quality_score).label('avg_quality'),
                    func.sum(RSSContentQuality.team_matches).label('team_matches'),
                    func.sum(func.cast(RSSContentQuality.is_duplicate, func.Integer)).label('duplicates')
                ).filter(
                    RSSContentQuality.recorded_at >= cutoff_time
                ).group_by(RSSContentQuality.feed_url).all()
                
                feed_quality = []
                for stat in feed_stats:
                    duplicate_rate = (stat.duplicates / stat.article_count * 100) if stat.article_count > 0 else 0
                    feed_quality.append({
                        'feed_url': stat.feed_url,
                        'article_count': stat.article_count,
                        'avg_quality': round(stat.avg_quality or 0, 3),
                        'team_matches': stat.team_matches or 0,
                        'duplicates': stat.duplicates or 0,
                        'duplicate_rate': round(duplicate_rate, 2)
                    })
                
                duplicate_rate = (duplicate_articles / total_articles * 100) if total_articles > 0 else 0
                
                return {
                    'period_hours': hours,
                    'total_articles': total_articles,
                    'duplicate_articles': duplicate_articles,
                    'duplicate_rate': round(duplicate_rate, 2),
                    'avg_quality_score': round(avg_quality, 3),
                    'total_team_matches': total_team_matches,
                    'feed_quality': feed_quality
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting content quality report: {e}")
            return {
                'period_hours': hours,
                'total_articles': 0,
                'duplicate_articles': 0,
                'duplicate_rate': 0,
                'avg_quality_score': 0,
                'total_team_matches': 0,
                'feed_quality': []
            }
    
    def get_sentiment_accuracy_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get sentiment analysis accuracy report.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing sentiment analysis statistics
        """
        try:
            with get_db_session() as session:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
                
                # Get overall statistics
                total_analyses = session.query(RSSSentimentAnalysis).filter(
                    RSSSentimentAnalysis.analysis_time >= cutoff_time
                ).count()
                
                successful_analyses = session.query(RSSSentimentAnalysis).filter(
                    and_(
                        RSSSentimentAnalysis.analysis_time >= cutoff_time,
                        RSSSentimentAnalysis.success == True
                    )
                ).count()
                
                # Get average processing time
                avg_processing_time = session.query(
                    func.avg(RSSSentimentAnalysis.processing_time)
                ).filter(
                    and_(
                        RSSSentimentAnalysis.analysis_time >= cutoff_time,
                        RSSSentimentAnalysis.success == True
                    )
                ).scalar() or 0.0
                
                # Get total articles analyzed
                total_articles_analyzed = session.query(
                    func.sum(RSSSentimentAnalysis.articles_analyzed)
                ).filter(
                    RSSSentimentAnalysis.analysis_time >= cutoff_time
                ).scalar() or 0
                
                # Get source type statistics
                source_stats = session.query(
                    RSSSentimentAnalysis.source_type,
                    func.count(RSSSentimentAnalysis.id).label('analysis_count'),
                    func.avg(RSSSentimentAnalysis.processing_time).label('avg_processing_time'),
                    func.sum(RSSSentimentAnalysis.articles_analyzed).label('articles_analyzed')
                ).filter(
                    RSSSentimentAnalysis.analysis_time >= cutoff_time
                ).group_by(RSSSentimentAnalysis.source_type).all()
                
                source_performance = []
                for stat in source_stats:
                    source_performance.append({
                        'source_type': stat.source_type,
                        'analysis_count': stat.analysis_count,
                        'avg_processing_time': round(stat.avg_processing_time or 0, 3),
                        'articles_analyzed': stat.articles_analyzed or 0
                    })
                
                success_rate = (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
                
                return {
                    'period_hours': hours,
                    'total_analyses': total_analyses,
                    'successful_analyses': successful_analyses,
                    'success_rate': round(success_rate, 2),
                    'avg_processing_time': round(avg_processing_time, 3),
                    'total_articles_analyzed': total_articles_analyzed,
                    'source_performance': source_performance
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting sentiment accuracy report: {e}")
            return {
                'period_hours': hours,
                'total_analyses': 0,
                'successful_analyses': 0,
                'success_rate': 0,
                'avg_processing_time': 0,
                'total_articles_analyzed': 0,
                'source_performance': []
            }
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring dashboard data.
        
        Returns:
            Dictionary containing all monitoring metrics
        """
        return {
            'feed_health_24h': self.get_feed_health_summary(24),
            'feed_health_7d': self.get_feed_health_summary(168),  # 7 days
            'content_quality_24h': self.get_content_quality_report(24),
            'content_quality_7d': self.get_content_quality_report(168),
            'sentiment_accuracy_24h': self.get_sentiment_accuracy_report(24),
            'sentiment_accuracy_7d': self.get_sentiment_accuracy_report(168),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def cleanup_old_data(self, retention_days: int = 30) -> Dict[str, int]:
        """
        Clean up old monitoring data.
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            with get_db_session() as session:
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)
                
                # Delete old feed health records
                health_deleted = session.query(RSSFeedHealth).filter(
                    RSSFeedHealth.check_time < cutoff_time
                ).delete()
                
                # Delete old content quality records
                quality_deleted = session.query(RSSContentQuality).filter(
                    RSSContentQuality.recorded_at < cutoff_time
                ).delete()
                
                # Delete old sentiment analysis records
                sentiment_deleted = session.query(RSSSentimentAnalysis).filter(
                    RSSSentimentAnalysis.analysis_time < cutoff_time
                ).delete()
                
                session.commit()
                
                logger.info(f"Cleaned up monitoring data: {health_deleted} health, "
                           f"{quality_deleted} quality, {sentiment_deleted} sentiment records")
                
                return {
                    'health_records_deleted': health_deleted,
                    'quality_records_deleted': quality_deleted,
                    'sentiment_records_deleted': sentiment_deleted,
                    'total_deleted': health_deleted + quality_deleted + sentiment_deleted
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error cleaning up monitoring data: {e}")
            return {
                'health_records_deleted': 0,
                'quality_records_deleted': 0,
                'sentiment_records_deleted': 0,
                'total_deleted': 0
            }


def create_default_rss_monitor() -> RSSMonitor:
    """Create a default RSS monitor instance."""
    return RSSMonitor()