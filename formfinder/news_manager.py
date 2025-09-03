"""News provider manager with automatic failover and load balancing.

This module manages multiple news API providers and handles automatic
failover when providers encounter rate limits or errors.
"""

import time
import logging
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .news_providers import (
    NewsProvider, ProviderResponse, NewsArticle, ProviderStatus,
    NewsAPIProvider, NewsDataIOProvider, TheNewsAPIProvider
)
from .news_monitoring import NewsProviderMonitor


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for provider selection."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    PRIORITY = "priority"
    LEAST_USED = "least_used"


@dataclass
class ProviderConfig:
    """Configuration for a news provider."""
    name: str
    api_key: str
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class ProviderStats:
    """Statistics for a news provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    last_used: float = 0
    average_response_time: float = 0


class NewsProviderManager:
    """Manages multiple news providers with failover and load balancing."""
    
    def __init__(self, 
                 provider_configs: List[ProviderConfig],
                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.PRIORITY,
                 max_concurrent_failures: int = 2,
                 monitor: Optional[NewsProviderMonitor] = None):
        """
        Initialize the news provider manager.
        
        Args:
            provider_configs: List of provider configurations
            load_balancing_strategy: Strategy for selecting providers
            max_concurrent_failures: Max failures before switching strategies
            monitor: Optional monitoring instance
        """
        self.logger = logging.getLogger("news_provider_manager")
        self.load_balancing_strategy = load_balancing_strategy
        self.max_concurrent_failures = max_concurrent_failures
        
        # Initialize monitoring
        self.monitor = monitor or NewsProviderMonitor()
        
        # Initialize providers
        self.providers: Dict[str, NewsProvider] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}
        self.provider_stats: Dict[str, ProviderStats] = {}
        
        for config in provider_configs:
            if config.enabled:
                provider = self._create_provider(config)
                if provider:
                    self.providers[config.name] = provider
                    self.provider_configs[config.name] = config
                    self.provider_stats[config.name] = ProviderStats()
        
        # Load balancing state
        self.current_provider_index = 0
        self.consecutive_failures = 0
        
        self.logger.info(f"Initialized with {len(self.providers)} providers: {list(self.providers.keys())}")
    
    def _create_provider(self, config: ProviderConfig) -> Optional[NewsProvider]:
        """Create a provider instance based on configuration.
        
        Args:
            config: Provider configuration
            
        Returns:
            NewsProvider instance or None if creation failed
        """
        try:
            if config.name == "newsapi":
                return NewsAPIProvider(config.api_key)
            elif config.name == "newsdata_io":
                return NewsDataIOProvider(config.api_key)
            elif config.name == "thenewsapi":
                return TheNewsAPIProvider(config.api_key)
            else:
                self.logger.error(f"Unknown provider type: {config.name}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create provider {config.name}: {e}")
            return None
    
    def _get_available_providers(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Get list of currently available provider names.
        
        Args:
            exclude: List of provider names to exclude
            
        Returns:
            List of available provider names
        """
        exclude = exclude or []
        available = []
        
        # Check if we have any providers that could potentially be recovered
        all_providers_unhealthy = True
        for name, provider in self.providers.items():
            if name not in exclude and provider.is_available():
                health_info = self.monitor.get_provider_health(name)
                if health_info['status'] in ['healthy', 'unknown'] and not health_info['is_in_cooldown']:
                    all_providers_unhealthy = False
                    break
        
        for name, provider in self.providers.items():
            if name not in exclude and provider.is_available():
                # Check monitoring health status
                health_info = self.monitor.get_provider_health(name)
                
                # Auto-recovery logic: More aggressive recovery when all providers are unhealthy
                should_attempt_recovery = False
                
                if health_info['status'] == 'unhealthy' and not health_info['is_in_cooldown']:
                    # Standard recovery: Provider available but marked unhealthy
                    should_attempt_recovery = True
                elif all_providers_unhealthy and health_info['status'] == 'unhealthy':
                    # Emergency recovery: All providers unhealthy, try to recover even if in cooldown
                    # This handles cases where all providers fail due to auth issues
                    should_attempt_recovery = True
                    self.logger.warning(f"Emergency recovery attempt for {name} - all providers unhealthy")
                
                if should_attempt_recovery:
                    self.logger.info(f"Attempting to recover health status for provider {name}")
                    
                    # Reset consecutive failures and health status
                    if name in self.monitor.provider_metrics:
                        metrics = self.monitor.provider_metrics[name]
                        # Reduce consecutive failures instead of resetting to 0 for more gradual recovery
                        metrics.consecutive_failures = max(0, metrics.consecutive_failures - 3)
                        
                        # If failures are now low enough, mark as healthy
                        if metrics.consecutive_failures < self.monitor.max_consecutive_failures:
                            metrics.is_healthy = True
                            self.logger.info(f"Reset health status for recovered provider {name}")
                        
                        # Clear cooldown if in emergency recovery mode
                        if all_providers_unhealthy and metrics.cooldown_until:
                            metrics.cooldown_until = None
                            self.logger.info(f"Cleared cooldown for emergency recovery of {name}")
                    
                    # Re-check health after recovery
                    health_info = self.monitor.get_provider_health(name)
                
                # Consider providers with 'healthy' or 'unknown' status as available
                # 'unknown' means no requests have been made yet, so provider should be available
                if health_info['status'] in ['healthy', 'unknown'] and not health_info['is_in_cooldown']:
                    available.append(name)
        
        # If still no providers available, log detailed status for debugging
        if not available:
            self.logger.error("No providers available after recovery attempts")
            for name, provider in self.providers.items():
                if name not in exclude:
                    health_info = self.monitor.get_provider_health(name)
                    self.logger.error(f"  {name}: available={provider.is_available()}, "
                                    f"status={health_info['status']}, "
                                    f"cooldown={health_info['is_in_cooldown']}, "
                                    f"failures={health_info['consecutive_failures']}")
        
        return available
    
    def _select_provider(self, available_providers: List[str], exclude: Optional[List[str]] = None) -> Optional[str]:
        """Select a provider based on the load balancing strategy.
        
        Args:
            available_providers: List of available provider names
            exclude: List of provider names to exclude
            
        Returns:
            Selected provider name or None if no providers available
        """
        exclude = exclude or []
        filtered_providers = [p for p in available_providers if p not in exclude]
        
        if not filtered_providers:
            return None
        
        if self.load_balancing_strategy == LoadBalancingStrategy.PRIORITY:
            # Sort by priority (lower number = higher priority)
            sorted_providers = sorted(
                filtered_providers,
                key=lambda name: self.provider_configs[name].priority
            )
            return sorted_providers[0]
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round robin through available providers
            if self.current_provider_index >= len(filtered_providers):
                self.current_provider_index = 0
            selected = filtered_providers[self.current_provider_index]
            self.current_provider_index += 1
            return selected
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(filtered_providers)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_USED:
            # Select provider with least recent usage
            least_used = min(
                filtered_providers,
                key=lambda name: self.provider_stats[name].last_used
            )
            return least_used
        
        return filtered_providers[0]
    
    def _update_provider_stats(self, provider_name: str, response: ProviderResponse, response_time: float):
        """Update statistics for a provider.
        
        Args:
            provider_name: Name of the provider
            response: Provider response
            response_time: Time taken for the request
        """
        stats = self.provider_stats[provider_name]
        stats.total_requests += 1
        stats.last_used = time.time()
        
        # Determine error type for monitoring
        error_type = None
        if not response.success:
            if response.rate_limited:
                error_type = "rate_limit"
            else:
                error_type = "other"
        
        # Record in monitoring system
        self.monitor.record_request(provider_name, "", response.success, response_time, error_type)
        
        if response.success:
            stats.successful_requests += 1
            # Update average response time
            if stats.average_response_time == 0:
                stats.average_response_time = response_time
            else:
                stats.average_response_time = (
                    stats.average_response_time * 0.8 + response_time * 0.2
                )
        else:
            stats.failed_requests += 1
            if response.rate_limited:
                stats.rate_limited_requests += 1
                # Set cooldown in monitoring system
                self.monitor.set_provider_cooldown(provider_name, 15)
    
    def fetch_articles(self, query: str, **kwargs) -> ProviderResponse:
        """Fetch articles with automatic failover.
        
        Args:
            query: Search query for articles
            **kwargs: Additional parameters passed to providers
            
        Returns:
            ProviderResponse with articles or error information
        """
        available_providers = self._get_available_providers()
        
        if not available_providers:
            self.logger.error("No providers available")
            return ProviderResponse(
                success=False,
                articles=[],
                error_message="No news providers available"
            )
        
        # Try providers until one succeeds or all fail
        attempted_providers = set()
        
        while len(attempted_providers) < len(available_providers):
            provider_name = self._select_provider(
                [p for p in available_providers if p not in attempted_providers]
            )
            
            if not provider_name:
                break
            
            attempted_providers.add(provider_name)
            provider = self.providers[provider_name]
            config = self.provider_configs[provider_name]
            
            self.logger.info(f"Attempting to fetch articles from {provider_name}")
            
            # Try the provider with retries
            for attempt in range(config.max_retries):
                start_time = time.time()
                response = provider.fetch_articles(query, **kwargs)
                response_time = time.time() - start_time
                
                self._update_provider_stats(provider_name, response, response_time)
                
                if response.success:
                    self.consecutive_failures = 0
                    self.logger.info(
                        f"Successfully fetched {len(response.articles)} articles from {provider_name}"
                    )
                    return response
                
                elif response.rate_limited:
                    self.logger.warning(f"{provider_name} rate limited, trying next provider")
                    
                    # Record failover for rate limit
                    next_available = [p for p in available_providers if p not in attempted_providers]
                    if next_available:
                        next_provider = self._select_provider(next_available)
                        if next_provider:
                            self.monitor.record_failover(provider_name, next_provider, "rate_limit", query, False)
                    
                    break  # Don't retry rate limited requests, try next provider
                
                else:
                    self.logger.warning(
                        f"{provider_name} failed (attempt {attempt + 1}/{config.max_retries}): {response.error_message}"
                    )
                    
                    # Record failover if we're going to try another provider
                    if attempt == config.max_retries - 1 and len(attempted_providers) < len(available_providers):
                        next_available = [p for p in available_providers if p not in attempted_providers]
                        if next_available:
                            next_provider = self._select_provider(next_available)
                            if next_provider:
                                error_type = "rate_limit" if response.rate_limited else "other"
                                self.monitor.record_failover(provider_name, next_provider, error_type, query, False)
                    
                    if attempt < config.max_retries - 1:
                        time.sleep(config.retry_delay * (2 ** attempt))  # Exponential backoff
            
            # Update available providers list in case provider status changed
            available_providers = self._get_available_providers()
        
        # All providers failed
        self.consecutive_failures += 1
        self.logger.error(f"All providers failed. Consecutive failures: {self.consecutive_failures}")
        
        return ProviderResponse(
            success=False,
            articles=[],
            error_message="All news providers failed or are rate limited"
        )
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status and statistics for all providers.
        
        Returns:
            Dictionary with provider status and statistics
        """
        status = {}
        for name, provider in self.providers.items():
            stats = self.provider_stats[name]
            health_info = self.monitor.get_provider_health(name)
            monitoring_stats = self.monitor.get_all_provider_stats().get(name, {})
            
            status[name] = {
                'status': provider.status.value,
                'available': provider.is_available(),
                'total_requests': stats.total_requests,
                'successful_requests': stats.successful_requests,
                'failed_requests': stats.failed_requests,
                'rate_limited_requests': stats.rate_limited_requests,
                'success_rate': (
                    stats.successful_requests / stats.total_requests 
                    if stats.total_requests > 0 else 0
                ),
                'average_response_time': stats.average_response_time,
                'last_used': stats.last_used,
                'rate_limit_reset_time': provider.rate_limit_reset_time,
                'health_status': health_info['status'],
                'is_in_cooldown': health_info['is_in_cooldown'],
                **monitoring_stats
            }
        return status
    
    def reset_provider_status(self, provider_name: str):
        """Reset a provider's status to active.
        
        Args:
            provider_name: Name of the provider to reset
        """
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.status = ProviderStatus.ACTIVE
            provider.rate_limit_reset_time = 0
            self.logger.info(f"Reset status for provider {provider_name}")
        else:
            self.logger.error(f"Provider {provider_name} not found")
    
    def disable_provider(self, provider_name: str):
        """Disable a provider.
        
        Args:
            provider_name: Name of the provider to disable
        """
        if provider_name in self.providers:
            self.providers[provider_name].status = ProviderStatus.DISABLED
            self.logger.info(f"Disabled provider {provider_name}")
        else:
            self.logger.error(f"Provider {provider_name} not found")
    
    def enable_provider(self, provider_name: str):
        """Enable a provider.
        
        Args:
            provider_name: Name of the provider to enable
        """
        if provider_name in self.providers:
            self.providers[provider_name].status = ProviderStatus.ACTIVE
            self.logger.info(f"Enabled provider {provider_name}")
        else:
            self.logger.error(f"Provider {provider_name} not found")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return self.monitor.get_system_stats()
    
    def export_monitoring_data(self, filename: Optional[str] = None) -> str:
        """Export monitoring data to file."""
        return self.monitor.export_metrics(filename)
    
    def reset_monitoring_data(self) -> None:
        """Reset all monitoring data."""
        self.monitor.reset_metrics()
        self.logger.info("Monitoring data has been reset")