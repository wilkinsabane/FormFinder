"""PostgreSQL-based RSS Content Caching System

This module provides efficient caching for RSS content using PostgreSQL
instead of SQLite, with improved performance and scalability.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .database import get_db_session
from .rss_models import RSSCachedArticle, CachedArticle
from .rss_content_parser import ParsedArticle

logger = logging.getLogger(__name__)


class RSSContentCache:
    """PostgreSQL-based RSS content cache with deduplication and expiration."""
    
    def __init__(self, default_ttl_hours: int = 24):
        """
        Initialize RSS content cache.
        
        Args:
            default_ttl_hours: Default time-to-live for cached articles in hours
        """
        self.default_ttl_hours = default_ttl_hours
        logger.info(f"Initialized PostgreSQL RSS cache with {default_ttl_hours}h TTL")
    
    def get_cached_articles(self, source_url: str, max_age_hours: int = None) -> List[CachedArticle]:
        """
        Retrieve cached articles from a specific source.
        
        Args:
            source_url: RSS feed URL to get articles from
            max_age_hours: Maximum age of articles to return (None for default TTL)
            
        Returns:
            List of cached articles
        """
        try:
            with get_db_session() as session:
                query = session.query(RSSCachedArticle).filter(
                    RSSCachedArticle.source_url == source_url
                )
                
                # Apply age filter
                if max_age_hours is not None:
                    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
                    query = query.filter(RSSCachedArticle.cached_at >= cutoff_time)
                else:
                    # Use expiration time
                    query = query.filter(
                        or_(
                            RSSCachedArticle.expires_at.is_(None),
                            RSSCachedArticle.expires_at > datetime.now(timezone.utc)
                        )
                    )
                
                # Order by most recent first
                query = query.order_by(desc(RSSCachedArticle.cached_at))
                
                cached_records = query.all()
                
                # Convert to CachedArticle dataclass
                articles = []
                for record in cached_records:
                    article = CachedArticle(
                        url=record.url,
                        title=record.title,
                        content=record.content,
                        published_date=record.published_date,
                        source_url=record.source_url,
                        source_name=record.source_name,
                        tags=record.tags or [],
                        image_url=record.image_url,
                        cached_at=record.cached_at
                    )
                    articles.append(article)
                
                logger.debug(f"Retrieved {len(articles)} cached articles from {source_url}")
                return articles
                
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving cached articles: {e}")
            return []
    
    def cache_articles(self, articles: List[ParsedArticle], source_url: str, 
                      source_name: str = None, ttl_hours: int = None) -> int:
        """
        Cache a list of articles with deduplication.
        
        Args:
            articles: List of parsed articles to cache
            source_url: RSS feed URL
            source_name: Human-readable source name
            ttl_hours: Time-to-live in hours (None for default)
            
        Returns:
            Number of articles successfully cached
        """
        if not articles:
            return 0
        
        ttl = ttl_hours or self.default_ttl_hours
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl)
        cached_count = 0
        
        try:
            with get_db_session() as session:
                for article in articles:
                    try:
                        # Check if article already exists
                        existing = session.query(RSSCachedArticle).filter(
                            RSSCachedArticle.url == article.url
                        ).first()
                        
                        if existing:
                            # Update existing article
                            existing.title = article.title
                            existing.content = article.content
                            existing.published_date = article.published_date
                            existing.source_url = source_url
                            existing.source_name = source_name or existing.source_name
                            existing.tags = article.tags
                            existing.image_url = getattr(article, 'image_url', None)
                            existing.cached_at = datetime.now(timezone.utc)
                            existing.expires_at = expires_at
                            
                            logger.debug(f"Updated cached article: {article.url}")
                        else:
                            # Create new cached article
                            cached_article = RSSCachedArticle(
                                url=article.url,
                                title=article.title,
                                content=article.content,
                                published_date=article.published_date,
                                source_url=source_url,
                                source_name=source_name,
                                tags=article.tags,
                                image_url=getattr(article, 'image_url', None),
                                cached_at=datetime.now(timezone.utc),
                                expires_at=expires_at
                            )
                            
                            session.add(cached_article)
                            logger.debug(f"Cached new article: {article.url}")
                        
                        cached_count += 1
                        
                    except IntegrityError as e:
                        # Handle duplicate URL constraint
                        logger.debug(f"Duplicate article URL ignored: {article.url}")
                        session.rollback()
                        continue
                    except Exception as e:
                        logger.warning(f"Error caching article {article.url}: {e}")
                        continue
                
                session.commit()
                logger.info(f"Successfully cached {cached_count}/{len(articles)} articles from {source_url}")
                return cached_count
                
        except SQLAlchemyError as e:
            logger.error(f"Error caching articles: {e}")
            return 0
    
    def is_duplicate(self, article_url: str) -> bool:
        """
        Check if an article URL already exists in cache.
        
        Args:
            article_url: URL to check
            
        Returns:
            True if article exists in cache, False otherwise
        """
        try:
            with get_db_session() as session:
                exists = session.query(RSSCachedArticle).filter(
                    RSSCachedArticle.url == article_url
                ).first() is not None
                
                return exists
                
        except SQLAlchemyError as e:
            logger.error(f"Error checking for duplicate: {e}")
            return False
    
    def search_articles(self, query: str, max_results: int = 50, 
                       max_age_hours: int = 168) -> List[CachedArticle]:
        """
        Search cached articles by content.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            max_age_hours: Maximum age of articles to search (default: 7 days)
            
        Returns:
            List of matching cached articles
        """
        try:
            with get_db_session() as session:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
                
                # Use PostgreSQL full-text search or simple ILIKE
                search_query = session.query(RSSCachedArticle).filter(
                    and_(
                        RSSCachedArticle.cached_at >= cutoff_time,
                        or_(
                            RSSCachedArticle.title.ilike(f'%{query}%'),
                            RSSCachedArticle.content.ilike(f'%{query}%')
                        )
                    )
                ).order_by(desc(RSSCachedArticle.cached_at)).limit(max_results)
                
                cached_records = search_query.all()
                
                # Convert to CachedArticle dataclass
                articles = []
                for record in cached_records:
                    article = CachedArticle(
                        url=record.url,
                        title=record.title,
                        content=record.content,
                        published_date=record.published_date,
                        source_url=record.source_url,
                        source_name=record.source_name,
                        tags=record.tags or [],
                        image_url=record.image_url,
                        cached_at=record.cached_at
                    )
                    articles.append(article)
                
                logger.debug(f"Found {len(articles)} articles matching '{query}'")
                return articles
                
        except SQLAlchemyError as e:
            logger.error(f"Error searching articles: {e}")
            return []
    
    def cleanup_expired(self) -> int:
        """
        Remove expired articles from cache.
        
        Returns:
            Number of articles removed
        """
        try:
            with get_db_session() as session:
                current_time = datetime.now(timezone.utc)
                
                # Delete expired articles
                deleted_count = session.query(RSSCachedArticle).filter(
                    and_(
                        RSSCachedArticle.expires_at.isnot(None),
                        RSSCachedArticle.expires_at <= current_time
                    )
                ).delete()
                
                session.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired articles")
                
                return deleted_count
                
        except SQLAlchemyError as e:
            logger.error(f"Error cleaning up expired articles: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            with get_db_session() as session:
                current_time = datetime.now(timezone.utc)
                
                # Total articles
                total_articles = session.query(RSSCachedArticle).count()
                
                # Active (non-expired) articles
                active_articles = session.query(RSSCachedArticle).filter(
                    or_(
                        RSSCachedArticle.expires_at.is_(None),
                        RSSCachedArticle.expires_at > current_time
                    )
                ).count()
                
                # Expired articles
                expired_articles = session.query(RSSCachedArticle).filter(
                    and_(
                        RSSCachedArticle.expires_at.isnot(None),
                        RSSCachedArticle.expires_at <= current_time
                    )
                ).count()
                
                # Articles by source
                source_stats = session.query(
                    RSSCachedArticle.source_url,
                    RSSCachedArticle.source_name,
                    func.count(RSSCachedArticle.id).label('article_count')
                ).group_by(
                    RSSCachedArticle.source_url,
                    RSSCachedArticle.source_name
                ).all()
                
                sources = []
                for stat in source_stats:
                    sources.append({
                        'source_url': stat.source_url,
                        'source_name': stat.source_name,
                        'article_count': stat.article_count
                    })
                
                # Recent activity (last 24 hours)
                recent_cutoff = current_time - timedelta(hours=24)
                recent_articles = session.query(RSSCachedArticle).filter(
                    RSSCachedArticle.cached_at >= recent_cutoff
                ).count()
                
                return {
                    'total_articles': total_articles,
                    'active_articles': active_articles,
                    'expired_articles': expired_articles,
                    'recent_articles_24h': recent_articles,
                    'sources': sources,
                    'cache_efficiency': round((active_articles / total_articles * 100) if total_articles > 0 else 0, 2)
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'total_articles': 0,
                'active_articles': 0,
                'expired_articles': 0,
                'recent_articles_24h': 0,
                'sources': [],
                'cache_efficiency': 0
            }
    
    def clear_cache(self, source_url: str = None) -> int:
        """
        Clear cache entries.
        
        Args:
            source_url: If provided, only clear articles from this source
            
        Returns:
            Number of articles removed
        """
        try:
            with get_db_session() as session:
                query = session.query(RSSCachedArticle)
                
                if source_url:
                    query = query.filter(RSSCachedArticle.source_url == source_url)
                    logger.info(f"Clearing cache for source: {source_url}")
                else:
                    logger.info("Clearing entire cache")
                
                deleted_count = query.delete()
                session.commit()
                
                logger.info(f"Cleared {deleted_count} articles from cache")
                return deleted_count
                
        except SQLAlchemyError as e:
            logger.error(f"Error clearing cache: {e}")
            return 0


def create_default_rss_cache() -> RSSContentCache:
    """Create a default RSS content cache instance."""
    return RSSContentCache()