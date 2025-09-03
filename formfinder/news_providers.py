"""News API provider abstraction layer with failover support.

This module provides a unified interface for multiple news API services
with automatic failover capabilities to handle rate limiting and service
availability issues.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests


class ProviderStatus(Enum):
    """Status of a news provider."""
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class NewsArticle:
    """Standardized news article structure."""
    title: str
    description: str
    url: str
    published_at: str
    source: str
    content: Optional[str] = None


@dataclass
class ProviderResponse:
    """Response from a news provider."""
    success: bool
    articles: List[NewsArticle]
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    rate_limited: bool = False


class NewsProvider(ABC):
    """Abstract base class for news API providers."""
    
    def __init__(self, api_key: str, name: str):
        self.api_key = api_key
        self.name = name
        self.status = ProviderStatus.ACTIVE
        self.last_request_time = 0
        self.rate_limit_reset_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        self.logger = logging.getLogger(f"news_provider.{name}")
    
    @abstractmethod
    def fetch_articles(self, query: str, **kwargs) -> ProviderResponse:
        """Fetch articles from the provider.
        
        Args:
            query: Search query for articles
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ProviderResponse with articles or error information
        """
        pass
    
    @abstractmethod
    def _normalize_response(self, response_data: Dict[str, Any]) -> List[NewsArticle]:
        """Normalize provider-specific response to standard format.
        
        Args:
            response_data: Raw response from the provider API
            
        Returns:
            List of normalized NewsArticle objects
        """
        pass
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        # Check if we're still rate limited
        if current_time < self.rate_limit_reset_time:
            wait_time = self.rate_limit_reset_time - current_time
            self.logger.info(f"Waiting {wait_time:.2f}s for rate limit reset")
            time.sleep(wait_time)
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _handle_rate_limit(self, response: requests.Response):
        """Handle rate limit response and update status.
        
        Args:
            response: HTTP response object
        """
        if response.status_code == 429:
            self.status = ProviderStatus.RATE_LIMITED
            
            # Try to get reset time from headers
            reset_header = response.headers.get('X-RateLimit-Reset') or response.headers.get('Retry-After')
            if reset_header:
                try:
                    reset_time = int(reset_header)
                    # If it's a timestamp, use it directly; if it's seconds, add to current time
                    if reset_time > time.time():
                        self.rate_limit_reset_time = reset_time
                    else:
                        self.rate_limit_reset_time = time.time() + reset_time
                except ValueError:
                    # Default to 15 minutes if we can't parse the header
                    self.rate_limit_reset_time = time.time() + 900
            else:
                # Default to 15 minutes if no reset header
                self.rate_limit_reset_time = time.time() + 900
            
            self.logger.warning(f"Rate limited until {self.rate_limit_reset_time}")
    
    def is_available(self) -> bool:
        """Check if the provider is currently available.
        
        Returns:
            True if provider can be used, False otherwise
        """
        if self.status == ProviderStatus.DISABLED:
            return False
        
        if self.status == ProviderStatus.RATE_LIMITED:
            if time.time() >= self.rate_limit_reset_time:
                self.status = ProviderStatus.ACTIVE
                return True
            return False
        
        return self.status == ProviderStatus.ACTIVE


class NewsAPIProvider(NewsProvider):
    """NewsAPI.org provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "newsapi")
        self.base_url = "https://newsapi.org/v2"
        self.min_request_interval = 1.0  # NewsAPI allows 1000 requests per day on free plan
    
    def fetch_articles(self, query: str, **kwargs) -> ProviderResponse:
        """Fetch articles from NewsAPI.org."""
        self._wait_for_rate_limit()
        
        params = {
            'q': query,
            'apiKey': self.api_key,
            'sortBy': kwargs.get('sort_by', 'relevancy'),
            'pageSize': kwargs.get('page_size', 20),
            'language': kwargs.get('language', 'en')
        }
        
        try:
            response = requests.get(f"{self.base_url}/everything", params=params, timeout=30)
            
            if response.status_code == 429:
                self._handle_rate_limit(response)
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message="Rate limit exceeded",
                    status_code=429,
                    rate_limited=True
                )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = self._normalize_response(data)
                return ProviderResponse(success=True, articles=articles)
            else:
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message=data.get('message', 'Unknown error'),
                    status_code=response.status_code
                )
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            self.status = ProviderStatus.ERROR
            return ProviderResponse(
                success=False,
                articles=[],
                error_message=str(e)
            )
    
    def _normalize_response(self, response_data: Dict[str, Any]) -> List[NewsArticle]:
        """Normalize NewsAPI response to standard format."""
        articles = []
        for article_data in response_data.get('articles', []):
            article = NewsArticle(
                title=article_data.get('title', ''),
                description=article_data.get('description', ''),
                url=article_data.get('url', ''),
                published_at=article_data.get('publishedAt', ''),
                source=article_data.get('source', {}).get('name', ''),
                content=article_data.get('content')
            )
            articles.append(article)
        return articles


class NewsDataIOProvider(NewsProvider):
    """NewsData.io provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "newsdata_io")
        self.base_url = "https://newsdata.io/api/1"
        self.min_request_interval = 0.5  # NewsData.io has different rate limits
    
    def fetch_articles(self, query: str, **kwargs) -> ProviderResponse:
        """Fetch articles from NewsData.io."""
        self._wait_for_rate_limit()
        
        params = {
            'q': query,
            'apikey': self.api_key,
            'language': kwargs.get('language', 'en'),
            'size': kwargs.get('page_size', 10)
        }
        
        try:
            response = requests.get(f"{self.base_url}/news", params=params, timeout=30)
            
            if response.status_code == 429:
                self._handle_rate_limit(response)
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message="Rate limit exceeded",
                    status_code=429,
                    rate_limited=True
                )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success':
                articles = self._normalize_response(data)
                return ProviderResponse(success=True, articles=articles)
            else:
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message=data.get('message', 'Unknown error'),
                    status_code=response.status_code
                )
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            self.status = ProviderStatus.ERROR
            return ProviderResponse(
                success=False,
                articles=[],
                error_message=str(e)
            )
    
    def _normalize_response(self, response_data: Dict[str, Any]) -> List[NewsArticle]:
        """Normalize NewsData.io response to standard format."""
        articles = []
        for article_data in response_data.get('results', []):
            article = NewsArticle(
                title=article_data.get('title', ''),
                description=article_data.get('description', ''),
                url=article_data.get('link', ''),
                published_at=article_data.get('pubDate', ''),
                source=article_data.get('source_id', ''),
                content=article_data.get('content')
            )
            articles.append(article)
        return articles


class TheNewsAPIProvider(NewsProvider):
    """TheNewsAPI.com provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "thenewsapi")
        self.base_url = "https://api.thenewsapi.com/v1"
        self.min_request_interval = 1.0  # Conservative rate limiting
    
    def fetch_articles(self, query: str, **kwargs) -> ProviderResponse:
        """Fetch articles from TheNewsAPI.com."""
        self._wait_for_rate_limit()
        
        params = {
            'search': query,
            'api_token': self.api_key,
            'language': kwargs.get('language', 'en'),
            'limit': kwargs.get('page_size', 10)
        }
        
        try:
            response = requests.get(f"{self.base_url}/news/all", params=params, timeout=30)
            
            if response.status_code == 429:
                self._handle_rate_limit(response)
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message="Rate limit exceeded",
                    status_code=429,
                    rate_limited=True
                )
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                articles = self._normalize_response(data)
                return ProviderResponse(success=True, articles=articles)
            else:
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message=data.get('message', 'Unknown error'),
                    status_code=response.status_code
                )
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            self.status = ProviderStatus.ERROR
            return ProviderResponse(
                success=False,
                articles=[],
                error_message=str(e)
            )
    
    def _normalize_response(self, response_data: Dict[str, Any]) -> List[NewsArticle]:
        """Normalize TheNewsAPI response to standard format."""
        articles = []
        for article_data in response_data.get('data', []):
            article = NewsArticle(
                title=article_data.get('title', ''),
                description=article_data.get('description', ''),
                url=article_data.get('url', ''),
                published_at=article_data.get('published_at', ''),
                source=article_data.get('source', ''),
                content=article_data.get('snippet')
            )
            articles.append(article)
        return articles