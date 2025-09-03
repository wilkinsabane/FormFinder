#!/usr/bin/env python3
"""
H2H Monitoring and Alerting System

Provides comprehensive monitoring, metrics collection, and alerting for the H2H data collection system.
Tracks API usage, cache performance, collection service health, and data quality metrics.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from sqlalchemy.orm import Session
from sqlalchemy import text


@dataclass
class H2HMetrics:
    """Container for H2H system metrics."""
    timestamp: datetime
    api_requests_total: int
    api_requests_successful: int
    api_requests_failed: int
    api_rate_limit_hits: int
    cache_hits: int
    cache_misses: int
    cache_size: int
    collection_queue_size: int
    collection_processed: int
    collection_failed: int
    avg_response_time: float
    data_quality_score: float


class H2HMonitor:
    """Monitors H2H system performance and health."""
    
    def __init__(self, db_session: Session, alert_thresholds: Dict[str, Any] = None):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'api_failure_rate': 0.1,  # 10% failure rate
            'cache_hit_rate': 0.7,    # 70% cache hit rate
            'avg_response_time': 5.0,  # 5 seconds
            'queue_size': 1000,       # 1000 pending requests
            'data_quality_score': 0.8  # 80% data quality
        }
        
        # Metrics storage (in-memory for recent data)
        self.metrics_history = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.api_response_times = deque(maxlen=100)  # Last 100 API calls
        self.alerts_sent = defaultdict(lambda: datetime.min)  # Alert cooldown tracking
        
        # Performance counters
        self.counters = {
            'api_requests_total': 0,
            'api_requests_successful': 0,
            'api_requests_failed': 0,
            'api_rate_limit_hits': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'collection_processed': 0,
            'collection_failed': 0
        }
    
    def record_api_request(self, success: bool, response_time: float, rate_limited: bool = False):
        """Record API request metrics."""
        self.counters['api_requests_total'] += 1
        
        if success:
            self.counters['api_requests_successful'] += 1
        else:
            self.counters['api_requests_failed'] += 1
        
        if rate_limited:
            self.counters['api_rate_limit_hits'] += 1
        
        self.api_response_times.append(response_time)
        
        self.logger.debug(f"API request recorded: success={success}, time={response_time:.2f}s")
    
    def record_cache_access(self, hit: bool):
        """Record cache access metrics."""
        if hit:
            self.counters['cache_hits'] += 1
        else:
            self.counters['cache_misses'] += 1
    
    def record_collection_result(self, success: bool):
        """Record collection service results."""
        if success:
            self.counters['collection_processed'] += 1
        else:
            self.counters['collection_failed'] += 1
    
    def get_current_metrics(self, collection_service=None) -> H2HMetrics:
        """Get current system metrics."""
        try:
            # Calculate derived metrics
            total_requests = self.counters['api_requests_total']
            api_failure_rate = (
                self.counters['api_requests_failed'] / max(total_requests, 1)
            )
            
            total_cache_access = self.counters['cache_hits'] + self.counters['cache_misses']
            cache_hit_rate = (
                self.counters['cache_hits'] / max(total_cache_access, 1)
            )
            
            avg_response_time = (
                sum(self.api_response_times) / max(len(self.api_response_times), 1)
            )
            
            # Get cache size from database
            cache_size = self._get_cache_size()
            
            # Get collection queue size
            queue_size = 0
            if collection_service:
                queue_size = collection_service.get_queue_size()
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score()
            
            return H2HMetrics(
                timestamp=datetime.now(),
                api_requests_total=self.counters['api_requests_total'],
                api_requests_successful=self.counters['api_requests_successful'],
                api_requests_failed=self.counters['api_requests_failed'],
                api_rate_limit_hits=self.counters['api_rate_limit_hits'],
                cache_hits=self.counters['cache_hits'],
                cache_misses=self.counters['cache_misses'],
                cache_size=cache_size,
                collection_queue_size=queue_size,
                collection_processed=self.counters['collection_processed'],
                collection_failed=self.counters['collection_failed'],
                avg_response_time=avg_response_time,
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return self._get_default_metrics()
    
    def check_alerts(self, metrics: H2HMetrics) -> List[Dict[str, Any]]:
        """Check for alert conditions and return alerts."""
        alerts = []
        now = datetime.now()
        cooldown_period = timedelta(minutes=15)  # 15-minute cooldown
        
        # API failure rate alert
        if metrics.api_requests_total > 0:
            failure_rate = metrics.api_requests_failed / metrics.api_requests_total
            if (failure_rate > self.alert_thresholds['api_failure_rate'] and 
                now - self.alerts_sent['api_failure_rate'] > cooldown_period):
                alerts.append({
                    'type': 'api_failure_rate',
                    'severity': 'high',
                    'message': f"API failure rate is {failure_rate:.1%} (threshold: {self.alert_thresholds['api_failure_rate']:.1%})",
                    'value': failure_rate,
                    'threshold': self.alert_thresholds['api_failure_rate']
                })
                self.alerts_sent['api_failure_rate'] = now
        
        # Cache hit rate alert
        total_cache_access = metrics.cache_hits + metrics.cache_misses
        if total_cache_access > 0:
            cache_hit_rate = metrics.cache_hits / total_cache_access
            if (cache_hit_rate < self.alert_thresholds['cache_hit_rate'] and 
                now - self.alerts_sent['cache_hit_rate'] > cooldown_period):
                alerts.append({
                    'type': 'cache_hit_rate',
                    'severity': 'medium',
                    'message': f"Cache hit rate is {cache_hit_rate:.1%} (threshold: {self.alert_thresholds['cache_hit_rate']:.1%})",
                    'value': cache_hit_rate,
                    'threshold': self.alert_thresholds['cache_hit_rate']
                })
                self.alerts_sent['cache_hit_rate'] = now
        
        # Response time alert
        if (metrics.avg_response_time > self.alert_thresholds['avg_response_time'] and 
            now - self.alerts_sent['avg_response_time'] > cooldown_period):
            alerts.append({
                'type': 'avg_response_time',
                'severity': 'medium',
                'message': f"Average response time is {metrics.avg_response_time:.2f}s (threshold: {self.alert_thresholds['avg_response_time']:.2f}s)",
                'value': metrics.avg_response_time,
                'threshold': self.alert_thresholds['avg_response_time']
            })
            self.alerts_sent['avg_response_time'] = now
        
        # Queue size alert
        if (metrics.collection_queue_size > self.alert_thresholds['queue_size'] and 
            now - self.alerts_sent['queue_size'] > cooldown_period):
            alerts.append({
                'type': 'queue_size',
                'severity': 'high',
                'message': f"Collection queue size is {metrics.collection_queue_size} (threshold: {self.alert_thresholds['queue_size']})",
                'value': metrics.collection_queue_size,
                'threshold': self.alert_thresholds['queue_size']
            })
            self.alerts_sent['queue_size'] = now
        
        # Data quality alert
        if (metrics.data_quality_score < self.alert_thresholds['data_quality_score'] and 
            now - self.alerts_sent['data_quality_score'] > cooldown_period):
            alerts.append({
                'type': 'data_quality_score',
                'severity': 'high',
                'message': f"Data quality score is {metrics.data_quality_score:.1%} (threshold: {self.alert_thresholds['data_quality_score']:.1%})",
                'value': metrics.data_quality_score,
                'threshold': self.alert_thresholds['data_quality_score']
            })
            self.alerts_sent['data_quality_score'] = now
        
        return alerts
    
    def store_metrics(self, metrics: H2HMetrics):
        """Store metrics for historical analysis."""
        self.metrics_history.append(metrics)
        
        # Optionally store in database for long-term analysis
        try:
            query = text("""
                INSERT INTO h2h_metrics 
                (timestamp, api_requests_total, api_requests_successful, api_requests_failed,
                 api_rate_limit_hits, cache_hits, cache_misses, cache_size,
                 collection_queue_size, collection_processed, collection_failed,
                 avg_response_time, data_quality_score)
                VALUES 
                (:timestamp, :api_requests_total, :api_requests_successful, :api_requests_failed,
                 :api_rate_limit_hits, :cache_hits, :cache_misses, :cache_size,
                 :collection_queue_size, :collection_processed, :collection_failed,
                 :avg_response_time, :data_quality_score)
            """)
            
            self.db_session.execute(query, asdict(metrics))
            self.db_session.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
            self.db_session.rollback()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            metrics = self.get_current_metrics()
            alerts = self.check_alerts(metrics)
            
            # Determine overall health
            if any(alert['severity'] == 'high' for alert in alerts):
                health = 'unhealthy'
            elif any(alert['severity'] == 'medium' for alert in alerts):
                health = 'degraded'
            else:
                health = 'healthy'
            
            return {
                'status': health,
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(metrics),
                'alerts': alerts,
                'uptime_hours': self._get_uptime_hours()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return {
                'status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def reset_counters(self):
        """Reset performance counters (typically called daily)."""
        self.counters = {key: 0 for key in self.counters}
        self.api_response_times.clear()
        self.logger.info("Performance counters reset")
    
    def _get_cache_size(self) -> int:
        """Get current cache size from database."""
        try:
            query = text("SELECT COUNT(*) as cache_size FROM h2h_cache")
            result = self.db_session.execute(query).fetchone()
            return result.cache_size if result else 0
        except Exception as e:
            self.logger.error(f"Error getting cache size: {e}")
            return 0
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate data quality score based on cache completeness and freshness."""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN overall_games_played > 0 THEN 1 END) as valid_entries,
                    COUNT(CASE WHEN last_fetched_at > :recent_threshold THEN 1 END) as fresh_entries
                FROM h2h_cache
            """)
            
            recent_threshold = datetime.now() - timedelta(hours=24)
            result = self.db_session.execute(query, {
                'recent_threshold': recent_threshold
            }).fetchone()
            
            if result and result.total_entries > 0:
                completeness_score = result.valid_entries / result.total_entries
                freshness_score = result.fresh_entries / result.total_entries
                return (completeness_score + freshness_score) / 2
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality score: {e}")
            return 0.0
    
    def _get_uptime_hours(self) -> float:
        """Get system uptime in hours (simplified implementation)."""
        # This would typically track actual service start time
        # For now, return hours since first metric
        if self.metrics_history:
            first_metric = self.metrics_history[0]
            uptime = datetime.now() - first_metric.timestamp
            return uptime.total_seconds() / 3600
        return 0.0
    
    def _get_default_metrics(self) -> H2HMetrics:
        """Get default metrics when calculation fails."""
        return H2HMetrics(
            timestamp=datetime.now(),
            api_requests_total=0,
            api_requests_successful=0,
            api_requests_failed=0,
            api_rate_limit_hits=0,
            cache_hits=0,
            cache_misses=0,
            cache_size=0,
            collection_queue_size=0,
            collection_processed=0,
            collection_failed=0,
            avg_response_time=0.0,
            data_quality_score=0.0
        )


class H2HAlertManager:
    """Manages alert notifications for the H2H system."""
    
    def __init__(self, notification_config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.notification_config = notification_config or {}
        
    def send_alert(self, alert: Dict[str, Any]):
        """Send alert notification."""
        try:
            # Log the alert
            self.logger.warning(
                f"H2H Alert [{alert['type']}]: {alert['message']} "
                f"(severity: {alert['severity']})"
            )
            
            # Here you would integrate with your notification system
            # Examples: email, Slack, PagerDuty, etc.
            if self.notification_config.get('email_enabled'):
                self._send_email_alert(alert)
            
            if self.notification_config.get('slack_enabled'):
                self._send_slack_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert (placeholder implementation)."""
        # Implement email notification logic
        self.logger.info(f"Email alert sent for {alert['type']}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert (placeholder implementation)."""
        # Implement Slack notification logic
        self.logger.info(f"Slack alert sent for {alert['type']}")