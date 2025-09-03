#!/usr/bin/env python3
"""
RSS Feed Manager for Sports News Sentiment Analysis

This module provides RSS feed management capabilities including:
- Feed discovery and validation
- Health monitoring and failover
- Content categorization by sport/league
- Update frequency optimization
"""

import logging
import time
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FeedHealth:
    """Health status of an RSS feed."""
    url: str
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    average_response_time: float = 0.0
    total_articles: int = 0
    quality_score: float = 0.0
    error_message: Optional[str] = None

@dataclass
class RSSFeed:
    """RSS feed configuration and metadata."""
    url: str
    name: str
    sport: str
    league: Optional[str] = None
    update_interval: int = 300  # seconds
    priority: int = 1  # 1=highest, 5=lowest
    enabled: bool = True
    quality_score: float = 0.8
    tags: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    health: Optional[FeedHealth] = None
    
    def __post_init__(self):
        if self.health is None:
            self.health = FeedHealth(url=self.url)

@dataclass
class RSSConfig:
    """Configuration for RSS feed management."""
    feeds: List[RSSFeed] = field(default_factory=list)
    cache_dir: str = "data/rss_cache"
    max_cache_age_hours: int = 48
    health_check_interval: int = 300  # seconds
    max_consecutive_failures: int = 5
    request_timeout: int = 30
    user_agent: str = "FormFinder RSS Reader 1.0"
    enable_health_monitoring: bool = True

class RSSFeedManager:
    """Manages RSS feeds for sports news sentiment analysis."""
    
    def __init__(self, config: RSSConfig):
        """
        Initialize RSS feed manager.
        
        Args:
            config: RSS configuration settings
        """
        self.config = config
        self.feeds = {feed.url: feed for feed in config.feeds}
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Accept': 'application/rss+xml, application/xml, text/xml'
        })
        
        logger.info(f"Initialized RSS Feed Manager with {len(self.feeds)} feeds")
    
    def discover_feeds(self, sport: str, league: Optional[str] = None) -> List[str]:
        """
        Discover RSS feeds for a specific sport/league.
        
        Args:
            sport: Sport name (e.g., 'football', 'basketball')
            league: League name (e.g., 'premier_league', 'nba')
            
        Returns:
            List of discovered feed URLs
        """
        discovered_feeds = []
        
        # Predefined feed sources for major sports
        feed_sources = self._get_predefined_feeds(sport, league)
        
        for feed_url in feed_sources:
            if self.validate_feed(feed_url):
                discovered_feeds.append(feed_url)
                logger.info(f"Discovered valid RSS feed: {feed_url}")
            else:
                logger.warning(f"Invalid RSS feed discovered: {feed_url}")
        
        return discovered_feeds
    
    def _get_predefined_feeds(self, sport: str, league: Optional[str] = None) -> List[str]:
        """
        Get predefined RSS feeds for major sports news sources.
        
        Args:
            sport: Sport name
            league: League name (optional)
            
        Returns:
            List of predefined feed URLs
        """
        feeds = {
            'football': {
                'general': [
                    'https://www.bbc.co.uk/sport/football/rss.xml',
                    'https://www.skysports.com/rss/12040',
                    'https://feeds.reuters.com/reuters/UKSportsNews',
                    'https://www.espn.com/espn/rss/soccer/news'
                ],
                'premier_league': [
                    'https://www.bbc.co.uk/sport/football/premier-league/rss.xml',
                    'https://www.skysports.com/rss/12040',
                    'http://feeds.feedburner.com/PremierLeagueFootballNews'
                ],
                'championship': [
                    'https://www.bbc.co.uk/sport/football/championship/rss.xml',
                    'http://feeds.feedburner.com/ChampionshipFootballNews'
                ],
                'league_one': [
                    'http://feeds.feedburner.com/LeagueOneFootballNews'
                ],
                'league_two': [
                    'http://feeds.feedburner.com/LeagueTwoFootballNews'
                ],
                'bundesliga': [
                    'https://newsfeed.kicker.de/news/bundesliga'
                ],
                '2_bundesliga': [
                    'https://newsfeed.kicker.de/news/2-bundesliga'
                ],
                '3_liga': [
                    'https://newsfeed.kicker.de/news/3-liga'
                ],
                # 'superliga_albania': [
                #     'https://www.livesoccertv.com/rss/league/albania-superliga/'
                # ],  # Disabled due to 403 Forbidden error
                'liga_profesional_argentina': [
                    'https://www.ole.com.ar/rss/futbol-primera/'
                ],
                'serie_a_brazil': [
                    'https://feeds.folha.uol.com.br/emcimadahora/rss091.xml'
                ],
                'copa_libertadores': [
                    'https://www.ole.com.ar/rss/futbol-internacional/libertadores'
                ],
                'liga_mx': [
                    'https://www.reddit.com/r/LigaMX/.rss'
                ],
                # 'primeira_liga_portugal': [
                #     'https://rss.com/podcasts/longballfutebol/1611201/'
                # ]  # Disabled due to XML syntax error
            },
            'basketball': {
                'general': [
                    'https://www.espn.com/espn/rss/nba/news',
                    'https://www.bbc.co.uk/sport/basketball/rss.xml'
                ],
                'nba': [
                    'https://www.espn.com/espn/rss/nba/news'
                ]
            },
            'general_sports': [
                'https://feeds.reuters.com/reuters/sportsNews',
                'https://www.bbc.co.uk/sport/rss.xml',
                'https://www.skysports.com/rss/sports'
            ]
        }
        
        result = []
        
        if sport in feeds:
            sport_feeds = feeds[sport]
            if isinstance(sport_feeds, dict):
                # Add general feeds for the sport
                if 'general' in sport_feeds:
                    result.extend(sport_feeds['general'])
                
                # Add league-specific feeds if specified
                if league and league in sport_feeds:
                    result.extend(sport_feeds[league])
            else:
                result.extend(sport_feeds)
        
        # Always include general sports feeds
        result.extend(feeds.get('general_sports', []))
        
        return list(set(result))  # Remove duplicates
    
    def validate_feed(self, feed_url: str, timeout: Optional[int] = None) -> bool:
        """
        Validate if an RSS feed is accessible and contains valid content.
        
        Args:
            feed_url: URL of the RSS feed to validate
            timeout: Request timeout in seconds
            
        Returns:
            True if feed is valid, False otherwise
        """
        try:
            timeout = timeout or self.config.request_timeout
            
            # Check if URL is properly formatted
            parsed_url = urlparse(feed_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL format: {feed_url}")
                return False
            
            # Fetch the feed
            response = self.session.get(feed_url, timeout=timeout)
            response.raise_for_status()
            
            # Parse the feed
            feed = feedparser.parse(response.content)
            
            # Check if feed is valid
            if feed.bozo:
                logger.warning(f"Feed parsing error for {feed_url}: {feed.bozo_exception}")
                return False
            
            # Check if feed has entries
            if not feed.entries:
                logger.warning(f"Feed has no entries: {feed_url}")
                return False
            
            # Check if feed has required fields
            if not hasattr(feed.feed, 'title'):
                logger.warning(f"Feed missing title: {feed_url}")
                return False
            
            logger.debug(f"Feed validation successful: {feed_url} ({len(feed.entries)} entries)")
            return True
            
        except requests.RequestException as e:
            logger.warning(f"Network error validating feed {feed_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating feed {feed_url}: {e}")
            return False
    
    def get_active_feeds(self, sport: Optional[str] = None, league: Optional[str] = None) -> List[RSSFeed]:
        """
        Get list of active (healthy and enabled) RSS feeds.
        
        Args:
            sport: Filter by sport (optional)
            league: Filter by league (optional)
            
        Returns:
            List of active RSS feeds
        """
        active_feeds = []
        
        for feed in self.feeds.values():
            # Check if feed is enabled
            if not feed.enabled:
                continue
            
            # Check if feed is healthy
            if feed.health and not feed.health.is_healthy:
                continue
            
            # Apply sport filter
            if sport and feed.sport.lower() != sport.lower():
                continue
            
            # Apply league filter
            if league and feed.league and feed.league.lower() != league.lower():
                continue
            
            active_feeds.append(feed)
        
        # Sort by priority (lower number = higher priority)
        active_feeds.sort(key=lambda f: f.priority)
        
        return active_feeds
    
    def monitor_feed_health(self) -> Dict[str, FeedHealth]:
        """
        Monitor health of all configured RSS feeds.
        
        Returns:
            Dictionary mapping feed URLs to their health status
        """
        health_results = {}
        
        for feed_url, feed in self.feeds.items():
            if not feed.enabled:
                continue
            
            health = self._check_feed_health(feed)
            health_results[feed_url] = health
            
            # Update feed health
            feed.health = health
        
        logger.info(f"Health check completed for {len(health_results)} feeds")
        return health_results
    
    def _check_feed_health(self, feed: RSSFeed) -> FeedHealth:
        """
        Check health of a single RSS feed.
        
        Args:
            feed: RSS feed to check
            
        Returns:
            Updated health status
        """
        health = feed.health or FeedHealth(url=feed.url)
        health.last_check = datetime.now()
        
        try:
            start_time = time.time()
            
            # Fetch and parse feed
            response = self.session.get(feed.url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            parsed_feed = feedparser.parse(response.content)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update health metrics
            if health.average_response_time == 0:
                health.average_response_time = response_time
            else:
                health.average_response_time = (health.average_response_time + response_time) / 2
            
            # Check feed quality
            if parsed_feed.bozo or not parsed_feed.entries:
                raise ValueError("Feed parsing failed or no entries found")
            
            # Update success metrics
            health.is_healthy = True
            health.last_success = datetime.now()
            health.consecutive_failures = 0
            health.total_articles = len(parsed_feed.entries)
            health.quality_score = self._calculate_quality_score(parsed_feed)
            health.error_message = None
            
            logger.debug(f"Health check passed for {feed.url}: {health.total_articles} articles")
            
        except Exception as e:
            # Update failure metrics
            health.is_healthy = False
            health.consecutive_failures += 1
            health.error_message = str(e)
            
            logger.warning(f"Health check failed for {feed.url}: {e}")
            
            # Disable feed if too many consecutive failures
            if health.consecutive_failures >= self.config.max_consecutive_failures:
                feed.enabled = False
                logger.error(f"Disabled feed {feed.url} after {health.consecutive_failures} failures")
        
        return health
    
    def _calculate_quality_score(self, parsed_feed) -> float:
        """
        Calculate quality score for an RSS feed.
        
        Args:
            parsed_feed: Parsed feedparser object
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check if feed has title
        if hasattr(parsed_feed.feed, 'title') and parsed_feed.feed.title:
            score += 0.2
        
        # Check if feed has description
        if hasattr(parsed_feed.feed, 'description') and parsed_feed.feed.description:
            score += 0.1
        
        # Check entry quality
        if parsed_feed.entries:
            entry_scores = []
            for entry in parsed_feed.entries[:10]:  # Check first 10 entries
                entry_score = 0.0
                
                # Check if entry has title
                if hasattr(entry, 'title') and entry.title:
                    entry_score += 0.3
                
                # Check if entry has description
                if hasattr(entry, 'description') and entry.description:
                    entry_score += 0.2
                
                # Check if entry has link
                if hasattr(entry, 'link') and entry.link:
                    entry_score += 0.2
                
                # Check if entry has publication date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    entry_score += 0.1
                
                entry_scores.append(entry_score)
            
            if entry_scores:
                score += 0.7 * (sum(entry_scores) / len(entry_scores))
        
        return min(score, 1.0)
    
    def rotate_feeds(self, sport: Optional[str] = None) -> None:
        """
        Rotate feed priorities to distribute load.
        
        Args:
            sport: Rotate feeds for specific sport only (optional)
        """
        feeds_to_rotate = []
        
        for feed in self.feeds.values():
            if sport and feed.sport.lower() != sport.lower():
                continue
            if feed.enabled and feed.health and feed.health.is_healthy:
                feeds_to_rotate.append(feed)
        
        if len(feeds_to_rotate) > 1:
            # Rotate priorities
            priorities = [feed.priority for feed in feeds_to_rotate]
            priorities = priorities[1:] + [priorities[0]]
            
            for i, feed in enumerate(feeds_to_rotate):
                feed.priority = priorities[i]
            
            logger.info(f"Rotated priorities for {len(feeds_to_rotate)} feeds")
    
    def get_feeds_for_query(self, query: str, sport: Optional[str] = None) -> List[RSSFeed]:
        """
        Get relevant RSS feeds for a search query.
        
        Args:
            query: Search query (team name, etc.)
            sport: Sport filter (optional)
            
        Returns:
            List of relevant RSS feeds
        """
        relevant_feeds = []
        query_lower = query.lower()
        
        for feed in self.get_active_feeds(sport=sport):
            # Check if query matches feed tags
            if any(tag.lower() in query_lower for tag in feed.tags):
                relevant_feeds.append(feed)
                continue
            
            # Check if query matches sport or league
            if feed.sport.lower() in query_lower:
                relevant_feeds.append(feed)
                continue
            
            if feed.league and feed.league.lower() in query_lower:
                relevant_feeds.append(feed)
                continue
        
        # If no specific matches, return all active feeds for the sport
        if not relevant_feeds:
            relevant_feeds = self.get_active_feeds(sport=sport)
        
        return relevant_feeds
    
    def add_feed(self, feed: RSSFeed) -> bool:
        """
        Add a new RSS feed to the manager.
        
        Args:
            feed: RSS feed to add
            
        Returns:
            True if feed was added successfully
        """
        if feed.url in self.feeds:
            logger.warning(f"Feed already exists: {feed.url}")
            return False
        
        if not self.validate_feed(feed.url):
            logger.error(f"Cannot add invalid feed: {feed.url}")
            return False
        
        self.feeds[feed.url] = feed
        logger.info(f"Added new RSS feed: {feed.url}")
        return True
    
    def remove_feed(self, feed_url: str) -> bool:
        """
        Remove an RSS feed from the manager.
        
        Args:
            feed_url: URL of feed to remove
            
        Returns:
            True if feed was removed successfully
        """
        if feed_url not in self.feeds:
            logger.warning(f"Feed not found: {feed_url}")
            return False
        
        del self.feeds[feed_url]
        logger.info(f"Removed RSS feed: {feed_url}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get RSS feed manager statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        total_feeds = len(self.feeds)
        enabled_feeds = sum(1 for feed in self.feeds.values() if feed.enabled)
        healthy_feeds = sum(1 for feed in self.feeds.values() 
                          if feed.health and feed.health.is_healthy)
        
        sports = set(feed.sport for feed in self.feeds.values())
        leagues = set(feed.league for feed in self.feeds.values() if feed.league)
        
        avg_quality = 0.0
        if self.feeds:
            quality_scores = [feed.health.quality_score for feed in self.feeds.values() 
                            if feed.health and feed.health.quality_score > 0]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
        
        return {
            'total_feeds': total_feeds,
            'enabled_feeds': enabled_feeds,
            'healthy_feeds': healthy_feeds,
            'disabled_feeds': total_feeds - enabled_feeds,
            'unhealthy_feeds': enabled_feeds - healthy_feeds,
            'sports_covered': list(sports),
            'leagues_covered': list(leagues),
            'average_quality_score': round(avg_quality, 3),
            'health_check_enabled': self.config.enable_health_monitoring
        }
    
    def export_config(self, filepath: str) -> bool:
        """
        Export current feed configuration to file.
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            True if export was successful
        """
        try:
            config_data = {
                'feeds': [
                    {
                        'url': feed.url,
                        'name': feed.name,
                        'sport': feed.sport,
                        'league': feed.league,
                        'update_interval': feed.update_interval,
                        'priority': feed.priority,
                        'enabled': feed.enabled,
                        'quality_score': feed.quality_score,
                        'tags': feed.tags
                    }
                    for feed in self.feeds.values()
                ],
                'config': {
                    'cache_dir': self.config.cache_dir,
                    'max_cache_age_hours': self.config.max_cache_age_hours,
                    'health_check_interval': self.config.health_check_interval,
                    'max_consecutive_failures': self.config.max_consecutive_failures,
                    'request_timeout': self.config.request_timeout,
                    'enable_health_monitoring': self.config.enable_health_monitoring
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Exported RSS configuration to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False