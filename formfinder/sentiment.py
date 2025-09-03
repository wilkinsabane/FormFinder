"""
Sentiment Analysis Module for FormFinder

This module provides the SentimentAnalyzer class for fetching and analyzing
news sentiment related to football teams and matches.
"""

import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass

from .news_manager import NewsProviderManager, ProviderConfig
from .news_monitoring import NewsProviderMonitor
from .config import get_config
from .rss_news_provider import RSSNewsProvider, create_default_rss_provider
from .rss_models import SentimentAccuracy


class _SingletonNewsProviderMonitor:
    """
    Singleton wrapper for NewsProviderMonitor to ensure shared state across instances.
    This prevents the issue where each SentimentAnalyzer creates its own monitor,
    leading to unshared rate limiting states.
    """
    _instance = None
    _monitor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._monitor = NewsProviderMonitor(log_dir="data/logs/news_monitoring")
        return cls._instance
    
    def get_monitor(self):
        """Get the singleton monitor instance."""
        return self._monitor

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result container for sentiment analysis."""
    home_sentiment: float
    away_sentiment: float
    home_article_count: int
    away_article_count: int
    home_articles: List[Dict]
    away_articles: List[Dict]


class SentimentAnalyzer:
    """
    Encapsulates sentiment analysis functionality for football teams.
    
    This class handles fetching news articles about teams and analyzing
    their sentiment to provide insights for match predictions.
    Uses a multi-provider failover system for robust news fetching.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_hours: int = 24):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            api_key: Legacy NewsAPI key (for backward compatibility)
            cache_hours: How long to cache sentiment results (default: 24 hours)
        """
        self.cache_hours = cache_hours
        self.cache = {}
        
        # Initialize the news provider manager
        try:
            config = get_config()
            sentiment_config = config.sentiment_analysis
            
            # Initialize monitoring using singleton pattern
            singleton_monitor = _SingletonNewsProviderMonitor()
            self.monitor = singleton_monitor.get_monitor()
            
            # Handle legacy API key
            if api_key and sentiment_config.providers.get('newsapi'):
                sentiment_config.providers['newsapi'].api_key = api_key
            elif sentiment_config.news_api_key and sentiment_config.providers.get('newsapi'):
                sentiment_config.providers['newsapi'].api_key = sentiment_config.news_api_key
            
            # Convert sentiment config to provider configs
            provider_configs = self._convert_sentiment_config_to_provider_configs(sentiment_config)
            
            # Check if any providers are enabled - if not, raise exception to trigger fallback
            if not provider_configs:
                raise ValueError("No news providers are enabled in configuration")
            
            self.news_manager = NewsProviderManager(provider_configs, monitor=self.monitor)
            self.use_manager = True
            self.use_rss = False
            self.rss_provider = None
            
            # Initialize legacy compatibility attributes
            self.base_url = "https://newsapi.org/v2/everything"
            self.last_request_time = 0
            self.min_request_interval = 1.0
            self.monitor_enabled = True
            
            available_providers = self.news_manager._get_available_providers()
            logger.info(f"Initialized SentimentAnalyzer with {len(available_providers)} providers and monitoring")
            
        except Exception as e:
            logger.error(f"Failed to initialize NewsProviderManager: {e}")
            # Try RSS fallback before legacy mode
            try:
                logger.info("Attempting RSS fallback provider initialization")
                self.rss_provider = create_default_rss_provider()
                self.news_manager = None
                self.monitor = None
                self.use_manager = False
                self.use_rss = True
                self.base_url = "https://newsapi.org/v2/everything"
                self.last_request_time = 0
                self.min_request_interval = 1.0
                self.monitor_enabled = False
                logger.info("Successfully initialized RSS fallback provider")
            except Exception as rss_error:
                logger.error(f"RSS fallback also failed: {rss_error}")
                # Final fallback to legacy mode
                self.rss_provider = None
                self.news_manager = None
                self.monitor = None
                self.use_manager = False
                self.use_rss = False
                self.base_url = "https://newsapi.org/v2/everything"
                self.last_request_time = 0
                self.min_request_interval = 1.0
                self.monitor_enabled = False
                logger.warning("Falling back to legacy NewsAPI mode")
        
        # Always set api_key for legacy fallback compatibility
        if api_key:
            self.api_key = api_key
        elif hasattr(self, 'news_manager') and self.news_manager:
            # Try to get API key from the first available NewsAPI provider
            try:
                config = get_config()
                newsapi_config = config.sentiment_analysis.providers.get('newsapi')
                if newsapi_config and newsapi_config.api_key:
                    self.api_key = newsapi_config.api_key
                else:
                    self.api_key = ""
            except:
                self.api_key = ""
        else:
            self.api_key = ""
    
    def _convert_sentiment_config_to_provider_configs(self, sentiment_config) -> List[ProviderConfig]:
        """
        Convert SentimentAnalysisConfig to List[ProviderConfig] for NewsProviderManager.
        
        Args:
            sentiment_config: SentimentAnalysisConfig instance
            
        Returns:
            List of ProviderConfig objects
        """
        provider_configs = []
        for name, news_provider_config in sentiment_config.providers.items():
            if news_provider_config.enabled:
                provider_config = ProviderConfig(
                    name=name,
                    api_key=news_provider_config.api_key,
                    priority=news_provider_config.priority,
                    enabled=news_provider_config.enabled,
                    max_retries=news_provider_config.max_retries
                )
                provider_configs.append(provider_config)
        
        return provider_configs
        
    def _fetch_team_articles(
        self, 
        team_name: str, 
        start_date: str, 
        end_date: str,
        max_articles: int = 10
    ) -> List[Dict]:
        """
        Fetch news articles for a specific team within a date range.
        
        Args:
            team_name: Name of the football team
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_articles: Maximum number of articles to return
            
        Returns:
            List of article dictionaries with title, description, and content
        """
        # Use multi-provider system if available
        if self.news_manager:
            try:
                query = f'"{team_name}" football soccer'
                response = self.news_manager.fetch_articles(
                    query=query,
                    from_date=start_date,
                    to_date=end_date,
                    language='en',
                    page_size=max_articles
                )
                
                if response.success:
                    articles = [
                        {
                            'title': article.title,
                            'description': article.description,
                            'content': article.content or article.description,
                            'url': article.url,
                            'publishedAt': article.published_at,
                            'source': article.source
                        }
                        for article in response.articles
                    ]
                    logger.debug(f"Fetched {len(articles)} articles for {team_name} using multi-provider system")
                    return articles
                else:
                    logger.warning(f"Multi-provider system failed for {team_name}: {response.error_message}")
                    # Fall through to RSS or legacy implementation
            except Exception as e:
                logger.error(f"Multi-provider system failed for {team_name}: {e}")
                # Fall through to RSS or legacy implementation
        
        # Try RSS provider if available
        if hasattr(self, 'use_rss') and self.use_rss and self.rss_provider:
            try:
                logger.info(f"Using RSS fallback provider for {team_name}")
                articles = self.rss_provider.fetch_articles(
                    query=f'{team_name} football soccer',
                    max_results=max_articles,
                    sport='football'
                )
                
                if articles:
                    # RSS provider returns a list of dictionaries directly
                    logger.info(f"RSS provider fetched {len(articles)} articles for {team_name}")
                    return articles
                else:
                    logger.warning(f"RSS provider failed for {team_name}: No articles found")
            except Exception as e:
                logger.error(f"RSS provider failed for {team_name}: {e}")
        
        # Legacy implementation fallback
        return self._fetch_team_articles_legacy(team_name, start_date, end_date, max_articles)
    
    def _fetch_team_articles_legacy(
        self, 
        team_name: str, 
        start_date: str, 
        end_date: str,
        max_articles: int = 10
    ) -> List[Dict]:
        """
        Legacy implementation for fetching articles using direct NewsAPI calls.
        
        Args:
            team_name: Name of the football team
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_articles: Maximum number of articles to return
            
        Returns:
            List of article dictionaries with title, description, and content
        """
        if not hasattr(self, 'api_key') or not self.api_key or self.api_key == "YOUR_SECRET_API_KEY_HERE":
            logger.warning("No valid NewsAPI key provided, skipping sentiment analysis")
            return []
            
        params = {
            'q': f'"{team_name}" football soccer',
            'from': start_date,
            'to': end_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': max_articles,
            'apiKey': self.api_key
        }
        
        # Implement rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                self.last_request_time = time.time()
                response = requests.get(self.base_url, params=params, timeout=10)
                
                # Handle rate limiting specifically
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                    logger.warning(f"Rate limit hit for {team_name}. Waiting {retry_after} seconds before retry {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        time.sleep(retry_after)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded for {team_name} after {max_retries} attempts")
                        return []
                
                response.raise_for_status()
                data = response.json()
                
                articles = []
                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', 'Unknown')
                    })
                
                logger.debug(f"Fetched {len(articles)} articles for {team_name} using legacy NewsAPI")
                return articles
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Request failed for {team_name} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay} seconds")
                    time.sleep(delay)
                else:
                    logger.error(f"Error fetching articles for {team_name} after {max_retries} attempts: {e}")
                    return []
            except Exception as e:
                logger.error(f"Unexpected error fetching articles for {team_name}: {e}")
                return []
    
    def _analyze_article_sentiment(self, articles: List[Dict]) -> float:
        """
        Analyze sentiment of a list of articles.
        
        This is a simplified sentiment analysis that looks for positive/negative keywords.
        In a production environment, you might use more sophisticated NLP libraries.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Average sentiment score (-1.0 to 1.0)
        """
        if not articles:
            return 0.0
            
        positive_keywords = [
            'win', 'wins', 'winning', 'victory', 'victories', 'triumph', 'success',
            'successful', 'excellent', 'great', 'amazing', 'outstanding', 'brilliant',
            'dominant', 'strong', 'impressive', 'fantastic', 'superb', 'remarkable',
            'championship', 'trophy', 'title', 'promotion', 'qualify', 'qualified'
        ]
        
        negative_keywords = [
            'loss', 'losses', 'losing', 'defeat', 'defeated', 'failure', 'failed',
            'poor', 'bad', 'terrible', 'awful', 'disappointing', 'weak', 'struggle',
            'struggling', 'crisis', 'relegation', 'eliminated', 'injury', 'injuries',
            'suspended', 'banned', 'controversy', 'problems', 'issues'
        ]
        
        sentiments = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            positive_score = sum(1 for keyword in positive_keywords if keyword in text)
            negative_score = sum(1 for keyword in negative_keywords if keyword in text)
            
            if positive_score + negative_score > 0:
                sentiment = (positive_score - negative_score) / (positive_score + negative_score)
                sentiments.append(sentiment)
        
        if not sentiments:
            return 0.0
            
        avg_sentiment = sum(sentiments) / len(sentiments)
        logger.debug(f"Average sentiment from {len(sentiments)} articles: {avg_sentiment:.2f}")
        return avg_sentiment
    
    def get_sentiment_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None,
        days_back: int = 7
    ) -> SentimentResult:
        """
        Get sentiment analysis for a specific match.
        
        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            match_date: Date of the match (defaults to today)
            days_back: Number of days to look back for articles
            
        Returns:
            SentimentResult containing sentiment scores and article details
        """
        start_time = time.time()
        query = f"{home_team} vs {away_team}"
        
        if match_date is None:
            match_date = datetime.now()
        
        end_date = match_date
        start_date = match_date - timedelta(days=days_back)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Analyzing sentiment for {home_team} vs {away_team}")
        logger.debug(f"Date range: {start_str} to {end_str}")
        
        try:
            # Fetch articles for both teams
            home_articles = self._fetch_team_articles(home_team, start_str, end_str)
            away_articles = self._fetch_team_articles(away_team, start_str, end_str)
            
            # Analyze sentiment
            home_sentiment = self._analyze_article_sentiment(home_articles)
            away_sentiment = self._analyze_article_sentiment(away_articles)
            
            # Calculate confidence scores (based on article count and sentiment strength)
            home_confidence = min(len(home_articles) / 10.0, 1.0) * (1.0 - abs(home_sentiment) * 0.1)
            away_confidence = min(len(away_articles) / 10.0, 1.0) * (1.0 - abs(away_sentiment) * 0.1)
            
            result = SentimentResult(
                home_sentiment=home_sentiment,
                away_sentiment=away_sentiment,
                home_article_count=len(home_articles),
                away_article_count=len(away_articles),
                home_articles=home_articles,
                away_articles=away_articles
            )
            
            # Record sentiment analysis metrics if RSS monitoring is available
            processing_time = time.time() - start_time
            total_articles = len(home_articles) + len(away_articles)
            
            if hasattr(self, 'use_rss') and self.use_rss and self.rss_provider and hasattr(self.rss_provider, 'monitor'):
                try:
                    sentiment_data = SentimentAccuracy(
                        query=query,
                        source_type="rss",
                        home_team=home_team,
                        away_team=away_team,
                        home_sentiment=home_sentiment,
                        away_sentiment=away_sentiment,
                        home_confidence=home_confidence,
                        away_confidence=away_confidence,
                        articles_analyzed=total_articles,
                        processing_time=processing_time,
                        success=True
                    )
                    self.rss_provider.monitor.record_sentiment_analysis(sentiment_data)
                except Exception as monitor_error:
                    logger.debug(f"Failed to record sentiment analysis metrics: {monitor_error}")
            
            logger.info(f"Sentiment analysis complete:")
            logger.info(f"  {home_team}: {home_sentiment:.2f} ({len(home_articles)} articles)")
            logger.info(f"  {away_team}: {away_sentiment:.2f} ({len(away_articles)} articles)")
            
            return result
            
        except Exception as e:
            # Record failed sentiment analysis
            processing_time = time.time() - start_time
            
            if hasattr(self, 'use_rss') and self.use_rss and self.rss_provider and hasattr(self.rss_provider, 'monitor'):
                try:
                    sentiment_data = SentimentAccuracy(
                        query=query,
                        source_type="rss",
                        home_team=home_team,
                        away_team=away_team,
                        home_sentiment=None,
                        away_sentiment=None,
                        home_confidence=None,
                        away_confidence=None,
                        articles_analyzed=0,
                        processing_time=processing_time,
                        success=False,
                        error_message=str(e)
                    )
                    self.rss_provider.monitor.record_sentiment_analysis(sentiment_data)
                except Exception as monitor_error:
                    logger.debug(f"Failed to record sentiment analysis error: {monitor_error}")
            
            logger.error(f"Sentiment analysis failed for {home_team} vs {away_team}: {e}")
            raise
    
    def get_team_sentiment(self, team_name: str, days_back: int = 7) -> float:
        """
        Get overall sentiment for a specific team.
        
        Args:
            team_name: Name of the team
            days_back: Number of days to look back
            
        Returns:
            Overall sentiment score for the team
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        articles = self._fetch_team_articles(
            team_name,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        return self._analyze_article_sentiment(articles)
    
    def get_provider_stats(self) -> Dict:
        """
        Get statistics and health information for all news providers.
        
        Returns:
            Dictionary containing provider statistics and health status
        """
        if not self.news_manager:
            return {"status": "legacy_mode", "providers": {}}
        
        return {
            "status": "multi_provider",
            "available_providers": self.news_manager._get_available_providers(),
            "provider_stats": self.news_manager.get_provider_status(),
            "current_strategy": self.news_manager.load_balancing_strategy
        }
    
    def switch_provider_strategy(self, strategy: str) -> bool:
        """
        Switch the load balancing strategy for news providers.
        
        Args:
            strategy: New strategy ('priority', 'round_robin', 'random', 'least_used')
            
        Returns:
            True if strategy was changed successfully, False otherwise
        """
        if not self.news_manager:
            logger.warning("Cannot switch strategy: multi-provider system not available")
            return False
        
        try:
            self.news_manager.load_balancing_strategy = strategy
            logger.info(f"Switched news provider strategy to: {strategy}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch provider strategy: {e}")
            return False
    
    def get_system_monitoring_stats(self) -> Optional[Dict]:
        """
        Get overall system monitoring statistics.
        
        Returns:
            Dictionary containing system monitoring data or None if not available
        """
        if not self.news_manager:
            return None
        
        return self.news_manager.get_system_stats()
    
    def get_provider_health(self, provider_name: Optional[str] = None) -> Optional[Dict]:
        """
        Get health information for a specific provider or all providers.
        
        Args:
            provider_name: Name of specific provider, or None for all providers
            
        Returns:
            Health information dictionary or None if not available
        """
        if not (self.news_manager and self.monitor):
            return None
        
        if provider_name:
            return self.monitor.get_provider_health(provider_name)
        else:
            # Return health for all providers
            stats = self.news_manager.get_provider_status()
            return {name: {'health_status': data.get('health_status', 'unknown'),
                          'is_in_cooldown': data.get('is_in_cooldown', False),
                          'success_rate': data.get('success_rate', 0),
                          'average_response_time': data.get('average_response_time', 0)}
                   for name, data in stats.items()}
    
    def export_monitoring_data(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Export monitoring data to a file.
        
        Args:
            filename: Optional filename for export
            
        Returns:
            Filepath of exported data or None if failed
        """
        if not (self.news_manager and self.monitor):
            return None
        
        try:
            filepath = self.monitor.export_metrics(filename)
            logger.info(f"Monitoring data exported to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to export monitoring data: {e}")
            return None
    
    def reset_monitoring_data(self) -> bool:
        """
        Reset all monitoring data.
        
        Returns:
            True if reset was successful, False otherwise
        """
        if not (self.news_manager and self.monitor):
            return False
        
        try:
            self.monitor.reset_metrics()
            logger.info("Monitoring data has been reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset monitoring data: {e}")
            return False