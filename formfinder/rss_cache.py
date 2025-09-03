"""RSS Content Caching and Storage System

This module provides efficient caching and storage for RSS content with
duplicate detection, data retention policies, and performance optimization.
"""

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class CachedArticle:
    """Represents a cached RSS article."""
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
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['published_date'] = self.published_date.isoformat()
        data['cached_at'] = self.cached_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'CachedArticle':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        data['published_date'] = datetime.fromisoformat(data['published_date'])
        data['cached_at'] = datetime.fromisoformat(data['cached_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class RSSContentCache:
    """Efficient caching system for RSS content."""

    def __init__(self, cache_dir: str = "rss_cache", max_cache_size_mb: int = 100,
                 retention_days: int = 30, cleanup_interval_hours: int = 24):
        """
        Initialize RSS content cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_cache_size_mb: Maximum cache size in MB
            retention_days: Days to retain cached content
            cleanup_interval_hours: Hours between cleanup operations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.retention_days = retention_days
        self.cleanup_interval = cleanup_interval_hours * 3600  # Convert to seconds
        
        self.db_path = self.cache_dir / "rss_cache.db"
        self.lock = Lock()
        
        # In-memory caches for performance
        self._url_hash_cache: Dict[str, str] = {}
        self._duplicate_hashes: Set[str] = set()
        
        self.last_cleanup = time.time()
        
        self._init_database()
        self._load_duplicate_hashes()
        
        logger.info(f"RSS cache initialized: {self.cache_dir}")
        logger.info(f"Max size: {max_cache_size_mb}MB, Retention: {retention_days} days")

    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_articles (
                    url TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    content TEXT,
                    published_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    feed_url TEXT NOT NULL,
                    teams TEXT NOT NULL,  -- JSON array
                    sport TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    cached_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    file_size INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash 
                ON cached_articles(content_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_published_date 
                ON cached_articles(published_date)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_teams 
                ON cached_articles(teams)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sport 
                ON cached_articles(sport)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cached_articles(last_accessed)
            """)
            
            conn.commit()

    def _load_duplicate_hashes(self):
        """Load content hashes for duplicate detection."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT content_hash FROM cached_articles")
                self._duplicate_hashes = {row[0] for row in cursor.fetchall()}
            
            logger.debug(f"Loaded {len(self._duplicate_hashes)} content hashes")
        except Exception as e:
            logger.error(f"Failed to load duplicate hashes: {e}")
            self._duplicate_hashes = set()

    def _generate_content_hash(self, title: str, content: str, published_date: datetime) -> str:
        """Generate hash for duplicate detection."""
        # Normalize content for hashing
        normalized_title = title.lower().strip()
        normalized_content = content.lower().strip()[:1000]  # First 1000 chars
        date_str = published_date.strftime('%Y-%m-%d')
        
        hash_input = f"{normalized_title}|{normalized_content}|{date_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def is_duplicate(self, title: str, content: str, published_date: datetime) -> bool:
        """Check if content is a duplicate."""
        content_hash = self._generate_content_hash(title, content, published_date)
        return content_hash in self._duplicate_hashes

    def cache_article(self, article_data: Dict) -> bool:
        """Cache an article with duplicate detection."""
        try:
            # Generate content hash
            published_date = article_data.get('published_date')
            if isinstance(published_date, str):
                published_date = datetime.fromisoformat(published_date)
            
            content_hash = self._generate_content_hash(
                article_data['title'],
                article_data.get('content', ''),
                published_date
            )
            
            # Check for duplicates
            if content_hash in self._duplicate_hashes:
                logger.debug(f"Duplicate article detected: {article_data['title'][:50]}...")
                return False
            
            # Create cached article
            now = datetime.now(timezone.utc)
            cached_article = CachedArticle(
                url=article_data['url'],
                title=article_data['title'],
                description=article_data.get('description', ''),
                content=article_data.get('content', ''),
                published_date=published_date,
                source=article_data['source'],
                feed_url=article_data['feed_url'],
                teams=article_data.get('teams', []),
                sport=article_data.get('sport', 'football'),
                content_hash=content_hash,
                quality_score=article_data.get('quality_score', 0.5),
                cached_at=now,
                last_accessed=now
            )
            
            # Store in database
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cached_articles (
                            url, title, description, content, published_date,
                            source, feed_url, teams, sport, content_hash,
                            quality_score, cached_at, last_accessed, access_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cached_article.url,
                        cached_article.title,
                        cached_article.description,
                        cached_article.content,
                        cached_article.published_date.isoformat(),
                        cached_article.source,
                        cached_article.feed_url,
                        json.dumps(cached_article.teams),
                        cached_article.sport,
                        cached_article.content_hash,
                        cached_article.quality_score,
                        cached_article.cached_at.isoformat(),
                        cached_article.last_accessed.isoformat(),
                        cached_article.access_count
                    ))
                    conn.commit()
                
                # Update in-memory cache
                self._duplicate_hashes.add(content_hash)
                self._url_hash_cache[cached_article.url] = content_hash
            
            logger.debug(f"Cached article: {cached_article.title[:50]}...")
            
            # Trigger cleanup if needed
            self._maybe_cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache article: {e}")
            return False

    def get_cached_articles(self, teams: List[str] = None, sport: str = None,
                          hours_back: int = 24, max_results: int = 100) -> List[Dict]:
        """Retrieve cached articles with filtering."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            query = """
                SELECT * FROM cached_articles 
                WHERE published_date >= ?
            """
            params = [cutoff_time.isoformat()]
            
            # Add team filtering
            if teams:
                team_conditions = []
                for team in teams:
                    team_conditions.append("teams LIKE ?")
                    params.append(f"%{team}%")
                query += f" AND ({' OR '.join(team_conditions)})"
            
            # Add sport filtering
            if sport:
                query += " AND sport = ?"
                params.append(sport)
            
            query += " ORDER BY quality_score DESC, published_date DESC LIMIT ?"
            params.append(max_results)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article_dict = dict(row)
                # Parse JSON teams
                article_dict['teams'] = json.loads(article_dict['teams'])
                # Convert ISO strings to datetime
                article_dict['published_date'] = datetime.fromisoformat(article_dict['published_date'])
                articles.append(article_dict)
                
                # Update access statistics
                self._update_access_stats(article_dict['url'])
            
            logger.debug(f"Retrieved {len(articles)} cached articles")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached articles: {e}")
            return []

    def _update_access_stats(self, url: str):
        """Update access statistics for an article."""
        try:
            now = datetime.now(timezone.utc)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE cached_articles 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE url = ?
                """, (now.isoformat(), url))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update access stats: {e}")

    def _maybe_cleanup(self):
        """Trigger cleanup if interval has passed."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_cache()
            self.last_cleanup = current_time

    def cleanup_cache(self):
        """Clean up old and low-quality cached content."""
        try:
            logger.info("Starting cache cleanup...")
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Remove old articles
                    cursor = conn.execute("""
                        DELETE FROM cached_articles 
                        WHERE cached_at < ?
                    """, (cutoff_date.isoformat(),))
                    
                    old_count = cursor.rowcount
                    
                    # Check cache size and remove low-quality articles if needed
                    cursor = conn.execute("""
                        SELECT COUNT(*), SUM(LENGTH(content)) as total_size
                        FROM cached_articles
                    """)
                    
                    count, total_size = cursor.fetchone()
                    total_size = total_size or 0
                    
                    if total_size > self.max_cache_size:
                        # Remove articles with lowest quality score and access count
                        excess_size = total_size - self.max_cache_size
                        
                        cursor = conn.execute("""
                            DELETE FROM cached_articles 
                            WHERE url IN (
                                SELECT url FROM cached_articles 
                                ORDER BY quality_score ASC, access_count ASC, last_accessed ASC
                                LIMIT (SELECT COUNT(*) * 0.1 FROM cached_articles)
                            )
                        """)
                        
                        size_count = cursor.rowcount
                        logger.info(f"Removed {size_count} articles due to size limit")
                    
                    conn.commit()
            
            # Reload duplicate hashes
            self._load_duplicate_hashes()
            
            logger.info(f"Cache cleanup complete: removed {old_count} old articles")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_articles,
                        SUM(LENGTH(content)) as total_size,
                        AVG(quality_score) as avg_quality,
                        MIN(cached_at) as oldest_cached,
                        MAX(cached_at) as newest_cached,
                        SUM(access_count) as total_accesses
                    FROM cached_articles
                """)
                
                row = cursor.fetchone()
                
                return {
                    'total_articles': row[0] or 0,
                    'total_size_mb': (row[1] or 0) / (1024 * 1024),
                    'avg_quality_score': row[2] or 0.0,
                    'oldest_cached': row[3],
                    'newest_cached': row[4],
                    'total_accesses': row[5] or 0,
                    'cache_dir': str(self.cache_dir),
                    'max_size_mb': self.max_cache_size / (1024 * 1024),
                    'retention_days': self.retention_days
                }
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def clear_cache(self):
        """Clear all cached content."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cached_articles")
                    conn.commit()
                
                self._duplicate_hashes.clear()
                self._url_hash_cache.clear()
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


def create_default_rss_cache() -> RSSContentCache:
    """Create RSS cache with default settings."""
    return RSSContentCache(
        cache_dir="rss_cache",
        max_cache_size_mb=100,
        retention_days=30,
        cleanup_interval_hours=24
    )