"""News Provider Monitoring and Logging Module

This module provides comprehensive monitoring and logging capabilities
for the multi-provider news API system, tracking performance, health,
and failover events.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Metrics for a single news provider."""
    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    is_healthy: bool = True
    cooldown_until: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in seconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests
    
    @property
    def is_in_cooldown(self) -> bool:
        """Check if provider is currently in cooldown period."""
        if not self.cooldown_until:
            return False
        return datetime.now() < self.cooldown_until


@dataclass
class FailoverEvent:
    """Record of a failover event."""
    timestamp: datetime
    from_provider: str
    to_provider: str
    reason: str
    query: str
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'from_provider': self.from_provider,
            'to_provider': self.to_provider,
            'reason': self.reason,
            'query': self.query,
            'success': self.success
        }


@dataclass
class SystemMetrics:
    """Overall system metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_failovers: int = 0
    successful_failovers: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def uptime_hours(self) -> float:
        """Calculate system uptime in hours."""
        return (datetime.now() - self.start_time).total_seconds() / 3600
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def failover_success_rate(self) -> float:
        """Calculate failover success rate."""
        if self.total_failovers == 0:
            return 0.0
        return (self.successful_failovers / self.total_failovers) * 100


class NewsProviderMonitor:
    """Monitor and log news provider performance and health."""
    
    def __init__(self, log_dir: str = "data/logs", max_events: int = 1000):
        """
        Initialize the news provider monitor.
        
        Args:
            log_dir: Directory to store monitoring logs
            max_events: Maximum number of events to keep in memory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        
        # Metrics storage
        self.provider_metrics: Dict[str, ProviderMetrics] = {}
        self.system_metrics = SystemMetrics()
        self.failover_events: deque = deque(maxlen=max_events)
        self.recent_requests: deque = deque(maxlen=max_events)
        
        # Health check thresholds
        self.max_consecutive_failures = 5
        self.min_success_rate = 70.0  # Percentage
        self.max_response_time = 30.0  # Seconds
        
        logger.info(f"Initialized NewsProviderMonitor with log directory: {log_dir}")
    
    def record_request(self, provider_name: str, query: str, 
                      success: bool, response_time: float, 
                      error_type: Optional[str] = None) -> None:
        """Record a request made to a news provider."""
        if provider_name not in self.provider_metrics:
            self.provider_metrics[provider_name] = ProviderMetrics(provider_name)
        
        metrics = self.provider_metrics[provider_name]
        now = datetime.now()
        
        # Update provider metrics
        metrics.total_requests += 1
        metrics.last_request_time = now
        
        if success:
            metrics.successful_requests += 1
            metrics.total_response_time += response_time
            metrics.last_success_time = now
            metrics.consecutive_failures = 0
        else:
            metrics.failed_requests += 1
            metrics.last_failure_time = now
            metrics.consecutive_failures += 1
            
            if error_type == "rate_limit":
                metrics.rate_limited_requests += 1
        
        # Update system metrics
        self.system_metrics.total_requests += 1
        if success:
            self.system_metrics.successful_requests += 1
        else:
            self.system_metrics.failed_requests += 1
        
        # Store recent request info
        self.recent_requests.append({
            'timestamp': now,
            'provider': provider_name,
            'query': query,
            'success': success,
            'response_time': response_time,
            'error_type': error_type
        })
        
        # Update provider health status
        self._update_provider_health(provider_name)
        
        logger.debug(f"Recorded request for {provider_name}: success={success}, time={response_time:.2f}s")
    
    def record_failover(self, from_provider: str, to_provider: str, 
                       reason: str, query: str, success: bool) -> None:
        """Record a failover event."""
        event = FailoverEvent(
            timestamp=datetime.now(),
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
            query=query,
            success=success
        )
        
        self.failover_events.append(event)
        
        # Update system metrics
        self.system_metrics.total_failovers += 1
        if success:
            self.system_metrics.successful_failovers += 1
        
        logger.info(f"Failover recorded: {from_provider} -> {to_provider} ({reason}), success={success}")
    
    def set_provider_cooldown(self, provider_name: str, minutes: int) -> None:
        """Set a cooldown period for a provider."""
        if provider_name not in self.provider_metrics:
            self.provider_metrics[provider_name] = ProviderMetrics(provider_name)
        
        cooldown_until = datetime.now() + timedelta(minutes=minutes)
        self.provider_metrics[provider_name].cooldown_until = cooldown_until
        self.provider_metrics[provider_name].is_healthy = False
        
        logger.warning(f"Provider {provider_name} set to cooldown until {cooldown_until}")
    
    def _update_provider_health(self, provider_name: str) -> None:
        """Update the health status of a provider based on metrics."""
        metrics = self.provider_metrics[provider_name]
        
        # Check if in cooldown
        if metrics.is_in_cooldown:
            return
        
        # Health criteria
        is_healthy = True
        reasons = []
        
        # Check consecutive failures
        if metrics.consecutive_failures >= self.max_consecutive_failures:
            is_healthy = False
            reasons.append(f"consecutive failures: {metrics.consecutive_failures}")
        
        # Check success rate (only if we have enough requests)
        if metrics.total_requests >= 10 and metrics.success_rate < self.min_success_rate:
            is_healthy = False
            reasons.append(f"low success rate: {metrics.success_rate:.1f}%")
        
        # Check average response time
        if metrics.successful_requests > 0 and metrics.average_response_time > self.max_response_time:
            is_healthy = False
            reasons.append(f"high response time: {metrics.average_response_time:.1f}s")
        
        # Update health status
        was_healthy = metrics.is_healthy
        metrics.is_healthy = is_healthy
        
        # Log health changes
        if was_healthy and not is_healthy:
            logger.warning(f"Provider {provider_name} marked as unhealthy: {', '.join(reasons)}")
        elif not was_healthy and is_healthy:
            logger.info(f"Provider {provider_name} recovered and marked as healthy")
    
    def get_provider_health(self, provider_name: str) -> Dict[str, Any]:
        """Get health information for a specific provider."""
        if provider_name not in self.provider_metrics:
            return {
                'status': 'unknown', 
                'is_in_cooldown': False,
                'success_rate': 0.0,
                'average_response_time': 0.0,
                'consecutive_failures': 0,
                'total_requests': 0,
                'last_success': None,
                'last_failure': None,
                'reason': 'no metrics available'
            }
        
        metrics = self.provider_metrics[provider_name]
        
        health_info = {
            'status': 'healthy' if metrics.is_healthy else 'unhealthy',
            'is_in_cooldown': metrics.is_in_cooldown,
            'success_rate': metrics.success_rate,
            'average_response_time': metrics.average_response_time,
            'consecutive_failures': metrics.consecutive_failures,
            'total_requests': metrics.total_requests,
            'last_success': metrics.last_success_time.isoformat() if metrics.last_success_time else None,
            'last_failure': metrics.last_failure_time.isoformat() if metrics.last_failure_time else None
        }
        
        if metrics.cooldown_until:
            health_info['cooldown_until'] = metrics.cooldown_until.isoformat()
        
        return health_info
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return {
            'uptime_hours': self.system_metrics.uptime_hours,
            'total_requests': self.system_metrics.total_requests,
            'success_rate': self.system_metrics.success_rate,
            'total_failovers': self.system_metrics.total_failovers,
            'failover_success_rate': self.system_metrics.failover_success_rate,
            'active_providers': len([p for p in self.provider_metrics.values() if p.is_healthy]),
            'total_providers': len(self.provider_metrics),
            'recent_requests_per_minute': self._calculate_recent_request_rate()
        }
    
    def get_all_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        return {
            name: {
                'total_requests': metrics.total_requests,
                'success_rate': metrics.success_rate,
                'average_response_time': metrics.average_response_time,
                'is_healthy': metrics.is_healthy,
                'is_in_cooldown': metrics.is_in_cooldown,
                'consecutive_failures': metrics.consecutive_failures,
                'rate_limited_requests': metrics.rate_limited_requests
            }
            for name, metrics in self.provider_metrics.items()
        }
    
    def _calculate_recent_request_rate(self) -> float:
        """Calculate requests per minute for the last 5 minutes."""
        cutoff = datetime.now() - timedelta(minutes=5)
        recent_count = sum(1 for req in self.recent_requests 
                          if req['timestamp'] > cutoff)
        return recent_count / 5.0  # Per minute
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Export all metrics to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"news_provider_metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'uptime_hours': self.system_metrics.uptime_hours,
                'total_requests': self.system_metrics.total_requests,
                'success_rate': self.system_metrics.success_rate,
                'total_failovers': self.system_metrics.total_failovers,
                'failover_success_rate': self.system_metrics.failover_success_rate
            },
            'provider_metrics': self.get_all_provider_stats(),
            'recent_failovers': [event.to_dict() for event in list(self.failover_events)[-50:]]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
        return str(filepath)
    
    def reset_metrics(self) -> None:
        """Reset all metrics and start fresh."""
        self.provider_metrics.clear()
        self.system_metrics = SystemMetrics()
        self.failover_events.clear()
        self.recent_requests.clear()
        
        logger.info("All metrics have been reset")