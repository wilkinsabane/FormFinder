#!/usr/bin/env python3
"""
RSS News Provider for Sports News Sentiment Analysis

This module provides an RSS-based news provider that integrates with the existing
news provider system, allowing RSS feeds to serve as a fallback when API providers
are unavailable or rate-limited.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
import feedparser

from .rss_feed_manager import RSSFeedManager, RSSConfig, RSSFeed
from .rss_content_parser import RSSContentParser, ParsingConfig, ParsedArticle
from .rss_cache_pg import RSSContentCache, CachedArticle, create_default_rss_cache
from .rss_monitoring_pg import RSSMonitor, create_default_rss_monitor
from .rss_models import FeedStatus, ContentQuality

logger = logging.getLogger(__name__)

@dataclass
class RSSProviderConfig:
    """Configuration for RSS news provider."""
    enabled: bool = True
    name: str = "RSS Provider"
    priority: int = 5  # Lower priority than API providers
    max_articles_per_query: int = 50
    cache_duration_hours: int = 1
    health_check_interval: int = 600  # 10 minutes
    request_timeout: int = 30
    
    # RSS-specific settings
    rss_config: Optional[RSSConfig] = None
    parsing_config: Optional[ParsingConfig] = None
    
    def __post_init__(self):
        if self.rss_config is None:
            self.rss_config = RSSConfig()
        if self.parsing_config is None:
            self.parsing_config = ParsingConfig()

class RSSNewsProvider:
    """RSS-based news provider for sentiment analysis."""
    
    def __init__(self, config: RSSProviderConfig, cache: Optional[RSSContentCache] = None,
                 monitor: Optional[RSSMonitor] = None):
        """
        Initialize RSS news provider.
        
        Args:
            config: RSS provider configuration
            cache: Optional RSS content cache
            monitor: Optional RSS monitoring system
        """
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.priority = config.priority
        
        # Initialize RSS components
        self.feed_manager = RSSFeedManager(config.rss_config)
        self.content_parser = RSSContentParser(config.parsing_config)
        
        # Initialize RSS content cache
        if cache is None:
            from .rss_cache import create_default_rss_cache
            self.cache = create_default_rss_cache()
        else:
            self.cache = cache
        
        # Initialize RSS monitoring
        if monitor is None:
            self.monitor = create_default_rss_monitor()
        else:
            self.monitor = monitor
        
        # Legacy cache for backward compatibility
        self._article_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Health monitoring
        self.is_healthy = True
        self.last_health_check = None
        self.consecutive_failures = 0
        
        logger.info(f"Initialized RSS News Provider: {self.name}")
    
    def fetch_articles(self, query: str, max_results: int = 10, 
                      sport: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch articles from RSS feeds based on query.
        
        Args:
            query: Search query (team name, topic, etc.)
            max_results: Maximum number of articles to return
            sport: Sport filter (optional)
            
        Returns:
            List of article dictionaries
        """
        try:
            # Extract team names from query for cache lookup
            teams = [team.strip() for team in query.split('vs') if team.strip()]
            if len(teams) < 2:
                teams = [query.strip()]  # Single team query
            
            # Check RSS cache first (PostgreSQL cache doesn't support team filtering)
            # Skip cache for now and fetch fresh articles
            cached_articles = []
            
            if cached_articles:
                logger.debug(f"Returning {len(cached_articles)} cached articles for query: {query}")
                return cached_articles[:max_results]
            
            # Check legacy cache as fallback
            cache_key = f"{query}_{sport}_{max_results}"
            if self._is_cache_valid(cache_key):
                legacy_cached = self._article_cache.get(cache_key, [])
                logger.debug(f"Returning {len(legacy_cached)} legacy cached articles for query: {query}")
                return legacy_cached[:max_results]
            
            # Get relevant RSS feeds
            relevant_feeds = self.feed_manager.get_feeds_for_query(query, sport)
            if not relevant_feeds:
                logger.warning(f"No relevant RSS feeds found for query: {query}")
                return []
            
            # Fetch and parse articles from feeds
            all_articles = []
            for feed in relevant_feeds:
                try:
                    feed_articles = self._fetch_from_feed(feed, query)
                    all_articles.extend(feed_articles)
                except Exception as e:
                    logger.warning(f"Error fetching from feed {feed.url}: {e}")
                    continue
            
            # Filter and sort articles
            filtered_articles = self._filter_articles(all_articles, query, sport)
            sorted_articles = self._sort_articles(filtered_articles)
            
            # Convert to API-compatible format and cache in RSS cache
            formatted_articles = []
            for article in sorted_articles:
                formatted_article = self._format_article(article)
                
                # Cache individual articles in RSS cache with duplicate detection
                article_data = {
                    'url': formatted_article['url'],
                    'title': formatted_article['title'],
                    'description': formatted_article.get('description', ''),
                    'content': formatted_article.get('content', ''),
                    'published_date': formatted_article['published_date'],
                    'source': formatted_article['source'],
                    'feed_url': article.feed_url if hasattr(article, 'feed_url') else '',
                    'teams': teams,
                    'sport': sport or 'football',
                    'quality_score': article.quality_score if hasattr(article, 'quality_score') else 0.5
                }
                
                # Temporarily disable caching to focus on team matching functionality
                # TODO: Fix cache compatibility issues
                formatted_articles.append(formatted_article)
            
            # Cache results in legacy cache for backward compatibility
            self._cache_articles(cache_key, formatted_articles)
            
            # Update health status
            self.is_healthy = True
            self.consecutive_failures = 0
            
            result = formatted_articles[:max_results]
            logger.info(f"Fetched {len(result)} articles from RSS feeds for query: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching RSS articles for query '{query}': {e}")
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.is_healthy = False
            return []
    
    def _fetch_from_feed(self, feed: RSSFeed, query: str) -> List[ParsedArticle]:
        """
        Fetch articles from a single RSS feed.
        
        Args:
            feed: RSS feed to fetch from
            query: Search query for filtering
            
        Returns:
            List of parsed articles
        """
        start_time = time.time()
        
        try:
            # Fetch feed content
            response = requests.get(
                feed.url, 
                timeout=self.config.request_timeout,
                headers={'User-Agent': self.config.parsing_config.user_agent}
            )
            response.raise_for_status()
            
            # Parse feed
            articles = self.content_parser.parse_feed(feed.url, response.text)
            
            # Filter articles by query relevance using team matching
            relevant_articles = []
            
            for article in articles:
                # Use team matching logic to find relevant articles
                team_matches = self._extract_team_matches(article, query)
                
                # Include article if it has team matches or contains query terms
                if team_matches > 0:
                    relevant_articles.append(article)
                    
                    # Record content quality metrics
                    try:
                        quality_score = self._calculate_content_quality(article)
                        
                        quality_data = ContentQuality(
                            feed_url=feed.url,
                            article_url=article.url,
                            quality_score=quality_score,
                            team_matches=team_matches,
                            is_duplicate=False,  # Will be updated during deduplication
                            content_length=len(article.content),
                            has_image=bool(getattr(article, 'image_url', None)),
                            source_name=feed.name
                        )
                        
                        self.monitor.record_content_quality(quality_data)
                    except Exception as monitor_error:
                        logger.debug(f"Failed to record content quality: {monitor_error}")
            
            # Record successful feed check
            response_time = time.time() - start_time
            feed_status = FeedStatus(
                feed_url=feed.url,
                is_successful=True,
                response_time=response_time,
                articles_count=len(relevant_articles)
            )
            self.monitor.record_feed_health(feed_status)
            
            logger.debug(f"Found {len(relevant_articles)} relevant articles from {feed.url}")
            return relevant_articles
            
        except Exception as e:
            # Record failed feed check
            response_time = time.time() - start_time
            feed_status = FeedStatus(
                feed_url=feed.url,
                is_successful=False,
                response_time=response_time,
                articles_count=0,  # No articles fetched due to error
                error_message=str(e)
            )
            self.monitor.record_feed_health(feed_status)
            
            logger.warning(f"Error fetching from RSS feed {feed.url}: {e}")
            return []
    
    def _filter_articles(self, articles: List[ParsedArticle], 
                        query: str, sport: Optional[str] = None) -> List[ParsedArticle]:
        """
        Filter articles based on query and sport.
        
        Args:
            articles: List of articles to filter
            query: Search query
            sport: Sport filter (optional)
            
        Returns:
            Filtered list of articles
        """
        filtered = articles
        
        # Filter by sport if specified
        if sport:
            filtered = self.content_parser.filter_articles_by_sport(filtered, sport)
        
        # Filter by date (last 7 days)
        filtered = self.content_parser.filter_articles_by_date(filtered, hours_back=168)
        
        # Remove duplicates based on content hash
        seen_hashes = set()
        unique_articles = []
        for article in filtered:
            if article.content_hash not in seen_hashes:
                seen_hashes.add(article.content_hash)
                unique_articles.append(article)
        
        return unique_articles
    
    def _sort_articles(self, articles: List[ParsedArticle]) -> List[ParsedArticle]:
        """
        Sort articles by relevance and recency.
        
        Args:
            articles: List of articles to sort
            
        Returns:
            Sorted list of articles
        """
        def sort_key(article: ParsedArticle) -> tuple:
            # Sort by quality score (desc), then by date (desc)
            date_score = 0
            if article.published_date:
                days_old = (datetime.now(timezone.utc) - article.published_date).days
                date_score = max(0, 7 - days_old)  # Newer articles get higher score
            
            return (-article.quality_score, -date_score)
        
        return sorted(articles, key=sort_key)
    
    def _format_article(self, article: ParsedArticle) -> Dict[str, Any]:
        """
        Format ParsedArticle to match API provider format.
        
        Args:
            article: Parsed article to format
            
        Returns:
            Formatted article dictionary
        """
        return {
            'title': article.title,
            'description': article.content[:500] + '...' if len(article.content) > 500 else article.content,
            'content': article.content,
            'url': article.url,
            'urlToImage': None,  # RSS feeds typically don't have images
            'publishedAt': article.published_date.isoformat() if article.published_date else None,
            'published_date': article.published_date.isoformat() if article.published_date else None,  # Add for compatibility
            'source': {
                'id': None,
                'name': article.source
            },
            'author': article.author,
            
            # RSS-specific metadata
            'sport': article.sport,
            'league': article.league,
            'tags': article.tags,
            'quality_score': article.quality_score,
            'word_count': article.word_count,
            'provider': 'rss'
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid
        """
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[cache_key]
        expiry_time = cache_time + timedelta(hours=self.config.cache_duration_hours)
        
        return datetime.now() < expiry_time
    
    def _cache_articles(self, cache_key: str, articles: List[Dict[str, Any]]) -> None:
        """
        Cache articles for future use.
        
        Args:
            cache_key: Key for caching
            articles: Articles to cache
        """
        self._article_cache[cache_key] = articles
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self) -> None:
        """
        Remove expired cache entries.
        """
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            expiry_time = timestamp + timedelta(hours=self.config.cache_duration_hours)
            if current_time > expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._article_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def check_health(self) -> bool:
        """
        Check health of RSS provider.
        
        Returns:
            True if provider is healthy
        """
        try:
            self.last_health_check = datetime.now()
            
            # Check RSS feed manager health
            health_results = self.feed_manager.monitor_feed_health()
            
            # Count healthy feeds
            healthy_feeds = sum(1 for health in health_results.values() if health.is_healthy)
            total_feeds = len(health_results)
            
            # Provider is healthy if at least 50% of feeds are healthy
            if total_feeds == 0:
                self.is_healthy = False
                logger.warning("No RSS feeds configured")
            else:
                health_ratio = healthy_feeds / total_feeds
                self.is_healthy = health_ratio >= 0.5
                
                if not self.is_healthy:
                    logger.warning(f"RSS provider unhealthy: {healthy_feeds}/{total_feeds} feeds healthy")
                else:
                    logger.debug(f"RSS provider healthy: {healthy_feeds}/{total_feeds} feeds healthy")
            
            return self.is_healthy
            
        except Exception as e:
            logger.error(f"Error checking RSS provider health: {e}")
            self.is_healthy = False
            return False
    
    def get_available_sports(self) -> List[str]:
        """
        Get list of sports covered by RSS feeds.
        
        Returns:
            List of sport names
        """
        sports = set()
        for feed in self.feed_manager.feeds.values():
            if feed.enabled and feed.sport:
                sports.add(feed.sport)
        return list(sports)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the RSS provider.
        
        Returns:
            Provider information dictionary
        """
        feed_stats = self.feed_manager.get_statistics()
        
        return {
            'name': self.name,
            'type': 'rss',
            'enabled': self.enabled,
            'healthy': self.is_healthy,
            'priority': self.priority,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'consecutive_failures': self.consecutive_failures,
            'cache_entries': len(self._article_cache),
            'feeds': feed_stats,
            'supported_sports': self.get_available_sports(),
            'capabilities': {
                'search': True,
                'sport_filtering': True,
                'date_filtering': True,
                'content_extraction': True,
                'duplicate_detection': True,
                'quality_scoring': True
            },
            'cache_stats': self.cache.get_cache_stats()
        }
    
    def clear_cache(self) -> None:
        """
        Clear all cached articles.
        """
        # Clear legacy cache
        self._article_cache.clear()
        self._cache_timestamps.clear()
        
        # Clear RSS cache
        self.cache.clear_cache()
        
        logger.info("Cleared RSS provider cache (both legacy and RSS cache)")
    
    def add_feed(self, feed_url: str, name: str, sport: str, 
                league: Optional[str] = None, priority: int = 1) -> bool:
        """
        Add a new RSS feed to the provider.
        
        Args:
            feed_url: URL of the RSS feed
            name: Name of the feed
            sport: Sport category
            league: League category (optional)
            priority: Feed priority (1=highest)
            
        Returns:
            True if feed was added successfully
        """
        feed = RSSFeed(
            url=feed_url,
            name=name,
            sport=sport,
            league=league,
            priority=priority
        )
        
        success = self.feed_manager.add_feed(feed)
        if success:
            # Clear cache to ensure new feed is used
            self.clear_cache()
            logger.info(f"Added RSS feed: {name} ({feed_url})")
        
        return success
    
    def remove_feed(self, feed_url: str) -> bool:
        """
        Remove an RSS feed from the provider.
        
        Args:
            feed_url: URL of the feed to remove
            
        Returns:
            True if feed was removed successfully
        """
        success = self.feed_manager.remove_feed(feed_url)
        if success:
            # Clear cache to ensure removed feed is not used
            self.clear_cache()
            logger.info(f"Removed RSS feed: {feed_url}")
        
        return success
    
    def discover_feeds(self, sport: str, league: Optional[str] = None) -> List[str]:
        """
        Discover new RSS feeds for a sport/league.
        
        Args:
            sport: Sport to discover feeds for
            league: League to discover feeds for (optional)
            
        Returns:
            List of discovered feed URLs
        """
        return self.feed_manager.discover_feeds(sport, league)
    
    def _calculate_content_quality(self, article) -> float:
        """
        Calculate content quality score for an article.
        
        Args:
            article: ParsedArticle to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Title quality (0.3 weight)
        if article.title and len(article.title.strip()) > 10:
            score += 0.3
        
        # Content quality (0.4 weight)
        if article.content:
            content_len = len(article.content.strip())
            if content_len > 100:
                score += 0.4
            elif content_len > 50:
                score += 0.2
        
        # Metadata quality (0.3 weight)
        metadata_score = 0.0
        if article.published_date:
            metadata_score += 0.1
        if article.author:
            metadata_score += 0.1
        if article.tags:
            metadata_score += 0.1
        
        score += metadata_score
        
        return min(score, 1.0)
    
    def _extract_team_matches(self, article, query: str) -> int:
        """
        Extract number of team name matches in article.
        
        Args:
            article: ParsedArticle to analyze
            query: Search query containing team names
            
        Returns:
            Number of team name matches found
        """
        try:
            # Simple team matching based on query terms in article content
            content_lower = f"{article.title} {article.content}".lower()
            query_terms = [term.strip().lower() for term in query.replace(' vs ', ' ').replace(' v ', ' ').split()]
            
            matches = 0
            for term in query_terms:
                if len(term) > 2 and term in content_lower:  # Avoid matching very short terms
                    matches += 1
            
            return matches
        except Exception:
            return 0

def create_default_rss_provider() -> RSSNewsProvider:
    """
    Create RSS provider with default configuration and popular sports feeds.
    
    Returns:
        Configured RSS news provider
    """
    # Create default RSS feeds
    default_feeds = [
        RSSFeed(
            url='https://www.bbc.co.uk/sport/football/rss.xml',
            name='BBC Football',
            sport='football',
            priority=1,
            tags=['premier_league', 'championship', 'international']
        ),
        RSSFeed(
            url='https://www.skysports.com/rss/12040',
            name='Sky Sports Football',
            sport='football',
            priority=2,
            tags=['premier_league', 'transfers']
        ),
        RSSFeed(
            url='https://www.bbc.co.uk/sport/basketball/rss.xml',
            name='BBC Basketball',
            sport='basketball',
            priority=1,
            tags=['nba', 'euroleague']
        ),
        RSSFeed(
            url='https://feeds.reuters.com/reuters/sportsNews',
            name='Reuters Sports',
            sport='general',
            priority=3,
            tags=['international', 'breaking_news']
        ),
        # International Football Leagues
        # RSSFeed(
        #     url='https://www.livesoccertv.com/rss/league/albania-superliga/',
        #     name='Albania Superliga',
        #     sport='football',
        #     league='superliga_albania',
        #     priority=4,
        #     tags=['albania', 'superliga', 'international']
        # ),  # Disabled due to 403 Forbidden error
        RSSFeed(
            url='https://www.ole.com.ar/rss/futbol-primera/',
            name='Argentina Liga Profesional',
            sport='football',
            league='liga_profesional_argentina',
            priority=4,
            tags=['argentina', 'liga_profesional', 'south_america']
        ),
        RSSFeed(
            url='https://feeds.folha.uol.com.br/emcimadahora/rss091.xml',
            name='Brazil Serie A',
            sport='football',
            league='serie_a_brazil',
            priority=4,
            tags=['brazil', 'serie_a', 'south_america']
        ),
        # English Football Leagues
        RSSFeed(
            url='http://feeds.feedburner.com/PremierLeagueFootballNews',
            name='Premier League News',
            sport='football',
            league='premier_league',
            priority=3,
            tags=['premier_league', 'england', 'top_tier']
        ),
        RSSFeed(
            url='http://feeds.feedburner.com/ChampionshipFootballNews',
            name='Championship News',
            sport='football',
            league='championship',
            priority=4,
            tags=['championship', 'england', 'second_tier']
        ),
        RSSFeed(
            url='http://feeds.feedburner.com/LeagueOneFootballNews',
            name='League One News',
            sport='football',
            league='league_one',
            priority=5,
            tags=['league_one', 'england', 'third_tier']
        ),
        RSSFeed(
            url='http://feeds.feedburner.com/LeagueTwoFootballNews',
            name='League Two News',
            sport='football',
            league='league_two',
            priority=5,
            tags=['league_two', 'england', 'fourth_tier']
        ),
        # German Football Leagues
        RSSFeed(
            url='https://newsfeed.kicker.de/news/bundesliga',
            name='Bundesliga Kicker',
            sport='football',
            league='bundesliga',
            priority=3,
            tags=['bundesliga', 'germany', 'top_tier']
        ),
        RSSFeed(
            url='https://newsfeed.kicker.de/news/2-bundesliga',
            name='2. Bundesliga Kicker',
            sport='football',
            league='2_bundesliga',
            priority=4,
            tags=['2_bundesliga', 'germany', 'second_tier']
        ),
        RSSFeed(
            url='https://newsfeed.kicker.de/news/3-liga',
            name='3. Liga Kicker',
            sport='football',
            league='3_liga',
            priority=5,
            tags=['3_liga', 'germany', 'third_tier']
        ),
        # International Competitions
        RSSFeed(
            url='https://www.ole.com.ar/rss/futbol-internacional/libertadores',
            name='Copa Libertadores',
            sport='football',
            league='copa_libertadores',
            priority=3,
            tags=['copa_libertadores', 'south_america', 'international']
        ),
        RSSFeed(
            url='https://www.reddit.com/r/LigaMX/.rss',
            name='Liga MX Reddit',
            sport='football',
            league='liga_mx',
            priority=4,
            tags=['liga_mx', 'mexico', 'north_america']
        ),
        # RSSFeed(
        #     url='https://rss.com/podcasts/longballfutebol/1611201/',
        #     name='Portugal Primeira Liga',
        #     sport='football',
        #     league='primeira_liga_portugal',
        #     priority=4,
        #     tags=['primeira_liga', 'portugal', 'europe']
        # )  # Disabled due to XML syntax error
    ]
    
    # Create configurations
    rss_config = RSSConfig(
        feeds=default_feeds,
        health_check_interval=300,
        max_consecutive_failures=3
    )
    
    parsing_config = ParsingConfig(
        min_content_length=100,
        extract_full_content=False,  # Keep false for performance
        filter_duplicates=True,
        max_articles_per_feed=20
    )
    
    provider_config = RSSProviderConfig(
        name="Default RSS Provider",
        priority=5,  # Lower priority than API providers
        max_articles_per_query=30,
        cache_duration_hours=2,
        rss_config=rss_config,
        parsing_config=parsing_config
    )
    
    return RSSNewsProvider(provider_config)