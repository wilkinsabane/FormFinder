"""Tests for the news provider failover system.

This module contains comprehensive tests for the multi-provider news API
failover mechanism, including provider switching, load balancing, and monitoring.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from formfinder.news_providers import NewsProvider, NewsAPIProvider, NewsDataIOProvider, TheNewsAPIProvider, ProviderResponse, NewsArticle
from formfinder.news_manager import NewsProviderManager, ProviderConfig, LoadBalancingStrategy
from formfinder.news_monitoring import NewsProviderMonitor, ProviderMetrics, FailoverEvent
from formfinder.sentiment import SentimentAnalyzer
from formfinder.config import SentimentAnalysisConfig, NewsProviderConfig


class MockNewsProvider(NewsProvider):
    """Mock news provider for testing."""
    
    def __init__(self, name: str, should_fail: bool = False, rate_limited: bool = False):
        super().__init__(api_key="test_key", name=name)
        self.should_fail = should_fail
        self.rate_limited = rate_limited
        self.request_count = 0
    
    def can_make_request(self) -> bool:
        """Check if provider can make requests."""
        return not self.rate_limited
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return not self.should_fail
        
    def fetch_articles(self, query: str, **kwargs) -> ProviderResponse:
        """Mock article fetching."""
        self.request_count += 1
        
        if self.should_fail:
            if self.rate_limited:
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message="429 Too Many Requests",
                    status_code=429,
                    rate_limited=True
                )
            else:
                return ProviderResponse(
                    success=False,
                    articles=[],
                    error_message="Connection error",
                    status_code=500
                )
        
        articles = [
            NewsArticle(
                title=f'Test Article {i} from {self.name}',
                description=f'Test description {i}',
                url=f'https://example.com/article{i}',
                published_at='2024-01-01T12:00:00Z',
                source=f'Test Source {self.name}',
                content=f'Test content {i}'
            )
            for i in range(min(kwargs.get('max_articles', 10), 5))
        ]
        
        return ProviderResponse(
            success=True,
            articles=articles
        )
    
    def _normalize_response(self, response_data: Dict[str, Any]) -> List[NewsArticle]:
        """Mock response normalization."""
        return []  # Not used in mock


class TestNewsProviderMonitor:
    """Test the news provider monitoring system."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = NewsProviderMonitor(log_dir="test_logs", max_events=500)
        
        assert monitor.max_events == 500
        assert len(monitor.provider_metrics) == 0
        assert len(monitor.failover_events) == 0
        assert monitor.system_metrics.total_requests == 0
    
    def test_record_successful_request(self):
        """Test recording a successful request."""
        monitor = NewsProviderMonitor()
        
        monitor.record_request("TestProvider", "test query", True, 1.5)
        
        assert "TestProvider" in monitor.provider_metrics
        metrics = monitor.provider_metrics["TestProvider"]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 100.0
        assert metrics.average_response_time == 1.5
        assert metrics.consecutive_failures == 0
        assert metrics.is_healthy is True
    
    def test_record_failed_request(self):
        """Test recording a failed request."""
        monitor = NewsProviderMonitor()
        
        monitor.record_request("TestProvider", "test query", False, 0.5, "connection")
        
        metrics = monitor.provider_metrics["TestProvider"]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 0.0
        assert metrics.consecutive_failures == 1
    
    def test_record_rate_limited_request(self):
        """Test recording a rate-limited request."""
        monitor = NewsProviderMonitor()
        
        monitor.record_request("TestProvider", "test query", False, 0.1, "rate_limit")
        
        metrics = monitor.provider_metrics["TestProvider"]
        assert metrics.rate_limited_requests == 1
        assert metrics.failed_requests == 1
    
    def test_provider_health_update(self):
        """Test provider health status updates."""
        monitor = NewsProviderMonitor()
        monitor.max_consecutive_failures = 3
        
        # Provider should be healthy initially
        for i in range(2):
            monitor.record_request("TestProvider", "test", False, 1.0)
        
        metrics = monitor.provider_metrics["TestProvider"]
        assert metrics.is_healthy is True  # Still under threshold
        
        # Should become unhealthy after exceeding threshold
        monitor.record_request("TestProvider", "test", False, 1.0)
        assert metrics.is_healthy is False
        
        # Should recover after successful request
        monitor.record_request("TestProvider", "test", True, 1.0)
        assert metrics.is_healthy is True
    
    def test_record_failover_event(self):
        """Test recording failover events."""
        monitor = NewsProviderMonitor()
        
        monitor.record_failover("Provider1", "Provider2", "rate_limit", "test query", True)
        
        assert len(monitor.failover_events) == 1
        event = monitor.failover_events[0]
        assert event.from_provider == "Provider1"
        assert event.to_provider == "Provider2"
        assert event.reason == "rate_limit"
        assert event.success is True
        
        assert monitor.system_metrics.total_failovers == 1
        assert monitor.system_metrics.successful_failovers == 1
    
    def test_provider_cooldown(self):
        """Test provider cooldown functionality."""
        monitor = NewsProviderMonitor()
        
        monitor.set_provider_cooldown("TestProvider", 5)  # 5 minutes
        
        metrics = monitor.provider_metrics["TestProvider"]
        assert metrics.is_in_cooldown is True
        assert metrics.is_healthy is False
        
        health_info = monitor.get_provider_health("TestProvider")
        assert health_info['status'] == 'unhealthy'
        assert health_info['is_in_cooldown'] is True
    
    def test_system_stats(self):
        """Test system statistics calculation."""
        monitor = NewsProviderMonitor()
        
        # Record some requests
        monitor.record_request("Provider1", "test", True, 1.0)
        monitor.record_request("Provider2", "test", False, 2.0)
        monitor.record_failover("Provider2", "Provider1", "error", "test", True)
        
        stats = monitor.get_system_stats()
        assert stats['total_requests'] == 2
        assert stats['success_rate'] == 50.0
        assert stats['total_failovers'] == 1
        assert stats['failover_success_rate'] == 100.0


class TestNewsProviderManager:
    """Test the news provider manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return [
            ProviderConfig(
                name='newsapi',
                api_key='test_key_1',
                priority=1,
                enabled=True
            ),
            ProviderConfig(
                name='newsdata_io',
                api_key='test_key_2',
                priority=2,
                enabled=True
            ),
            ProviderConfig(
                name='thenewsapi',
                api_key='test_key_3',
                priority=3,
                enabled=True
            )
        ]
    
    def test_manager_initialization(self, mock_config):
        """Test manager initialization with monitoring."""
        monitor = NewsProviderMonitor()
        manager = NewsProviderManager(
            provider_configs=mock_config,
            monitor=monitor
        )
        
        assert len(manager.providers) == 3
        assert manager.monitor is monitor
        assert 'newsapi' in manager.providers
        assert 'newsdata_io' in manager.providers
        assert 'thenewsapi' in manager.providers
    
    def test_successful_fetch(self, mock_config):
        """Test successful article fetching."""
        monitor = NewsProviderMonitor()
        
        with patch.object(NewsProviderManager, '_create_provider') as mock_create:
            # Create mock providers
            mock_providers = {
                'newsapi': MockNewsProvider('newsapi'),
                'newsdata_io': MockNewsProvider('newsdata_io'),
                'thenewsapi': MockNewsProvider('thenewsapi')
            }
            
            def create_provider_side_effect(config):
                return mock_providers.get(config.name)
            
            mock_create.side_effect = create_provider_side_effect
            
            manager = NewsProviderManager(
                provider_configs=mock_config,
                monitor=monitor
            )
            
            # Initialize provider health in monitor
            for provider_name in mock_providers.keys():
                monitor.record_request(provider_name, "test", True, 1.0)

            response = manager.fetch_articles("test query")

            assert response.success is True
            assert len(response.articles) > 0
            assert 'Test Article' in response.articles[0].title
            
            # Check monitoring recorded the request
            stats = monitor.get_system_stats()
            assert stats['total_requests'] >= 1
            assert stats['success_rate'] > 0.0
    
    def test_failover_on_rate_limit(self, mock_config):
        """Test failover when primary provider is rate limited."""
        monitor = NewsProviderMonitor()
        
        with patch.object(NewsProviderManager, '_create_provider') as mock_create:
            # Create mock providers with different behaviors
            mock_newsapi = MockNewsProvider('newsapi')
            mock_newsdata = MockNewsProvider('newsdata_io')
            mock_thenews = MockNewsProvider('thenewsapi')
            
            # Make newsapi fail with rate limit
            mock_newsapi.fetch_articles = Mock(return_value=ProviderResponse(
                success=False,
                articles=[],
                error_message="Rate limit exceeded",
                status_code=429,
                rate_limited=True
            ))
            
            mock_providers = {
                'newsapi': mock_newsapi,
                'newsdata_io': mock_newsdata,
                'thenewsapi': mock_thenews
            }
            
            def create_provider_side_effect(config):
                return mock_providers.get(config.name)
            
            mock_create.side_effect = create_provider_side_effect
            
            manager = NewsProviderManager(
                provider_configs=mock_config,
                monitor=monitor
            )
            
            # Initialize provider health in monitor
            for provider_name in mock_providers.keys():
                monitor.record_request(provider_name, "test", True, 1.0)

            response = manager.fetch_articles("test query")

            assert response.success is True
            assert len(response.articles) > 0
            assert 'Test Article' in response.articles[0].title
            
            # Check that failover was recorded
            assert len(monitor.failover_events) > 0
            
            # Check provider health
            newsapi_health = monitor.get_provider_health('newsapi')
            assert newsapi_health['status'] == 'unhealthy'
    
    def test_load_balancing_strategies(self, mock_config):
        """Test different load balancing strategies."""
        monitor = NewsProviderMonitor()
        manager = NewsProviderManager(
            provider_configs=mock_config,
            monitor=monitor
        )
        
        # Test strategy switching by directly setting the attribute
        manager.load_balancing_strategy = LoadBalancingStrategy.RANDOM
        assert manager.load_balancing_strategy == LoadBalancingStrategy.RANDOM
        
        manager.load_balancing_strategy = LoadBalancingStrategy.LEAST_USED
        assert manager.load_balancing_strategy == LoadBalancingStrategy.LEAST_USED
    
    def test_provider_exclusion(self, mock_config):
        """Test provider exclusion during selection."""
        monitor = NewsProviderMonitor()
        
        with patch.object(NewsProviderManager, '_create_provider') as mock_create:
            # Create mock providers
            mock_providers = {
                'newsapi': MockNewsProvider('newsapi'),
                'newsdata_io': MockNewsProvider('newsdata_io'),
                'thenewsapi': MockNewsProvider('thenewsapi')
            }
            
            def create_provider_side_effect(config):
                return mock_providers.get(config.name)
            
            mock_create.side_effect = create_provider_side_effect
            
            manager = NewsProviderManager(
                provider_configs=mock_config,
                monitor=monitor
            )
            
            # Initialize provider health in monitor
            for provider_name in mock_providers.keys():
                monitor.record_request(provider_name, "test", True, 1.0)
            
            # Test that we can get available providers
            all_available = manager._get_available_providers()
            assert len(all_available) == 3
            
            # Get available providers excluding one
            available = manager._get_available_providers(exclude=['newsapi'])
            
            assert 'newsapi' not in available
            assert len(available) == 2


class TestSentimentAnalyzerIntegration:
    """Test sentiment analyzer integration with failover system."""
    
    @pytest.fixture
    def mock_sentiment_config(self):
        """Create mock sentiment analysis configuration."""
        config = Mock(spec=SentimentAnalysisConfig)
        config.enabled = True
        config.cache_hours = 24
        config.enable_failover = True
        config.load_balancing_strategy = "priority"
        config.provider_health_check_interval = 300
        config.provider_cooldown_minutes = 15
        
        # Mock provider configs
        newsapi_config = Mock(spec=NewsProviderConfig)
        newsapi_config.api_key = "test_key_1"
        newsapi_config.enabled = True
        newsapi_config.priority = 1
        newsapi_config.rate_limit = 1000
        
        newsdata_config = Mock(spec=NewsProviderConfig)
        newsdata_config.api_key = "test_key_2"
        newsdata_config.enabled = True
        newsdata_config.priority = 2
        newsdata_config.rate_limit = 200
        
        config.news_providers = {
            'newsapi': newsapi_config,
            'newsdata': newsdata_config
        }
        
        config.get_provider_config.return_value = Mock(spec=NewsProviderConfig)
        config.get_provider_config.return_value.api_key = "test_key"
        config.get_provider_config.return_value.enabled = True
        config.get_provider_config.return_value.priority = 1
        config.get_provider_config.return_value.rate_limit = 1000
        
        return config
    
    @patch('formfinder.config.get_config')
    def test_sentiment_analyzer_with_monitoring(self, mock_get_config, mock_sentiment_config):
        """Test sentiment analyzer initialization with monitoring."""
        mock_config = Mock()
        mock_config.sentiment_analysis = mock_sentiment_config
        mock_get_config.return_value = mock_config
        
        analyzer = SentimentAnalyzer()
        
        assert analyzer.use_manager is True
        assert analyzer.news_manager is not None
        assert analyzer.monitor is not None
    
    @patch('formfinder.config.get_config')
    @patch('formfinder.news_providers.NewsAPIProvider.fetch_articles')
    def test_monitoring_methods(self, mock_fetch, mock_get_config, mock_sentiment_config):
        """Test monitoring-related methods in sentiment analyzer."""
        mock_fetch.return_value = [{'title': 'Test'}]
        
        mock_config = Mock()
        mock_config.sentiment_analysis = mock_sentiment_config
        mock_get_config.return_value = mock_config
        
        analyzer = SentimentAnalyzer()
        
        # Test provider stats
        stats = analyzer.get_provider_stats()
        assert stats is not None
        
        # Test system monitoring stats
        system_stats = analyzer.get_system_monitoring_stats()
        assert system_stats is not None
        
        # Test provider health
        health = analyzer.get_provider_health()
        assert health is not None
        
        # Test strategy switching
        result = analyzer.switch_provider_strategy(LoadBalancingStrategy.ROUND_ROBIN)
        assert result is True
        
        # Test monitoring data export
        filepath = analyzer.export_monitoring_data()
        assert filepath is not None
        
        # Test monitoring data reset
        result = analyzer.reset_monitoring_data()
        assert result is True


class TestFailoverScenarios:
    """Test various failover scenarios."""
    
    def test_all_providers_fail(self):
        """Test behavior when all providers fail."""
        monitor = NewsProviderMonitor()
        
        # Create providers that all fail
        providers = {
            'provider1': MockNewsProvider('Provider1', should_fail=True),
            'provider2': MockNewsProvider('Provider2', should_fail=True),
            'provider3': MockNewsProvider('Provider3', should_fail=True)
        }
        
        config = [
            ProviderConfig(name='provider1', api_key='key1', priority=1, enabled=True),
            ProviderConfig(name='provider2', api_key='key2', priority=2, enabled=True),
            ProviderConfig(name='provider3', api_key='key3', priority=3, enabled=True)
        ]
        
        with patch.object(NewsProviderManager, '_create_provider', return_value=None):
            manager = NewsProviderManager(provider_configs=config, monitor=monitor)
            manager.providers = providers
            
            response = manager.fetch_articles("test query")
            assert response.success is False
            assert response.error_message is not None
    
    def test_rate_limit_recovery(self):
        """Test provider recovery after rate limiting."""
        monitor = NewsProviderMonitor()
        
        # Simulate rate limiting and recovery
        provider = MockNewsProvider('TestProvider', rate_limited=True)
        
        # Initially rate limited
        assert provider.can_make_request() is False
        
        # Simulate recovery
        provider.rate_limited = False
        assert provider.can_make_request() is True
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        monitor = NewsProviderMonitor()
        
        # Simulate multiple concurrent requests
        for i in range(10):
            monitor.record_request(f"Provider{i % 3}", f"query{i}", True, 1.0)
        
        stats = monitor.get_system_stats()
        assert stats['total_requests'] == 10
        assert stats['success_rate'] == 100.0
    
    def test_monitoring_data_persistence(self):
        """Test monitoring data export and import."""
        monitor = NewsProviderMonitor(log_dir="test_logs")
        
        # Generate some test data
        monitor.record_request("Provider1", "test", True, 1.5)
        monitor.record_request("Provider2", "test", False, 0.5, "timeout")
        monitor.record_failover("Provider2", "Provider1", "timeout", "test", True)
        
        # Export data
        filepath = monitor.export_metrics("test_export.json")
        assert filepath is not None
        
        # Verify file exists and contains data
        import json
        from pathlib import Path
        
        export_file = Path(filepath)
        assert export_file.exists()
        
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        assert 'system_metrics' in data
        assert 'provider_metrics' in data
        assert 'recent_failovers' in data
        assert data['system_metrics']['total_requests'] == 2
        assert data['system_metrics']['total_failovers'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])