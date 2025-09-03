"""Monitoring Module for FormFinder2

This module provides comprehensive monitoring capabilities for the FormFinder2 system,
including health checks, performance tracking, and alerting.

Author: FormFinder2 Team
Created: 2025-01-01
Purpose: System monitoring and alerting
"""

import logging
import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psycopg2
from psycopg2.extras import RealDictCursor

from .config import load_config
from .exceptions import (
    HealthCheckError, AlertError, MetricsError, 
    DatabaseError, NotificationError
)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Represents a health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = None
    response_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    component: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the system monitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.db_connection = None
        
        # Monitoring state
        self.health_checks: Dict[str, HealthCheck] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.performance_metrics: List[PerformanceMetric] = []
        self.is_monitoring = False
        
        # Health check functions
        self.health_check_functions: Dict[str, Callable] = {
            'database': self._check_database_health,
            'api_quota': self._check_api_quota_health,
            'feature_computation': self._check_feature_computation_health,
            'disk_space': self._check_disk_space_health,
            'memory': self._check_memory_health,
            'cpu': self._check_cpu_health,
            'data_quality': self._check_data_quality_health,
            'training_performance': self._check_training_performance_health
        }
        
        # Alert thresholds from config
        self.monitoring_config = self.config.monitoring
    
    def connect_to_database(self) -> None:
        """Establish database connection."""
        try:
            db_config = self.config.database.postgresql
            self.db_connection = psycopg2.connect(
                host=db_config.host,
                port=db_config.port,
                database=db_config.database,
                user=db_config.username,
                password=db_config.password
            )
            self.logger.info("Database connection established for monitoring")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to database: {str(e)}")
    
    def close_database_connection(self) -> None:
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.logger.info("Database connection closed")
    
    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks.
        
        Returns:
            Dictionary of health check results
        """
        if not self.db_connection:
            self.connect_to_database()
        
        health_results = {}
        
        for component, check_function in self.health_check_functions.items():
            try:
                start_time = time.time()
                health_check = await check_function()
                end_time = time.time()
                
                health_check.response_time_ms = (end_time - start_time) * 1000
                health_results[component] = health_check
                
                # Store in instance state
                self.health_checks[component] = health_check
                
                # Log health check result
                self.logger.info(
                    f"Health check {component}: {health_check.status.value} "
                    f"({health_check.response_time_ms:.2f}ms)"
                )
                
            except Exception as e:
                error_check = HealthCheck(
                    component=component,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now()
                )
                health_results[component] = error_check
                self.health_checks[component] = error_check
                
                self.logger.error(f"Health check {component} failed: {str(e)}")
        
        # Persist health check results
        await self._persist_health_checks(health_results)
        
        # Check for alerts
        await self._evaluate_health_alerts(health_results)
        
        return health_results
    
    async def _check_database_health(self) -> HealthCheck:
        """Check database health.
        
        Returns:
            Database health check result
        """
        try:
            with self.db_connection.cursor() as cursor:
                # Test basic connectivity
                cursor.execute("SELECT 1")
                
                # Check connection count
                cursor.execute("""
                    SELECT count(*) as active_connections
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """)
                active_connections = cursor.fetchone()[0]
                
                # Check database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                """)
                db_size = cursor.fetchone()[0]
                
                # Check recent activity
                cursor.execute("""
                    SELECT COUNT(*) as recent_inserts
                    FROM feature_computation_log 
                    WHERE computed_at >= NOW() - INTERVAL '1 hour'
                """)
                recent_activity = cursor.fetchone()[0]
                
                metrics = {
                    'active_connections': active_connections,
                    'database_size': db_size,
                    'recent_activity': recent_activity
                }
                
                # Determine status
                if active_connections > 50:
                    status = HealthStatus.WARNING
                    message = f"High connection count: {active_connections}"
                elif active_connections > 80:
                    status = HealthStatus.CRITICAL
                    message = f"Critical connection count: {active_connections}"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Database is healthy"
                
                return HealthCheck(
                    component="database",
                    status=status,
                    message=message,
                    timestamp=datetime.now(),
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheck(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_api_quota_health(self) -> HealthCheck:
        """Check API quota health.
        
        Returns:
            API quota health check result
        """
        try:
            with self.db_connection.cursor() as cursor:
                # Check daily API usage
                cursor.execute("""
                    SELECT 
                        COALESCE(SUM(requests_made), 0) as daily_requests,
                        COALESCE(SUM(quota_used), 0) as daily_quota_used
                    FROM api_usage_log 
                    WHERE date_trunc('day', created_at) = date_trunc('day', NOW())
                """)
                
                result = cursor.fetchone()
                daily_requests = result[0]
                daily_quota_used = result[1]
                
                # Get quota limits from config
                daily_limit = self.config.feature_computation.api_usage.daily_quota_limit
                
                # Calculate usage percentage
                usage_percentage = (daily_requests / daily_limit) * 100 if daily_limit > 0 else 0
                
                metrics = {
                    'daily_requests': daily_requests,
                    'daily_quota_used': daily_quota_used,
                    'daily_limit': daily_limit,
                    'usage_percentage': usage_percentage
                }
                
                # Determine status
                if usage_percentage >= 95:
                    status = HealthStatus.CRITICAL
                    message = f"API quota critical: {usage_percentage:.1f}% used"
                elif usage_percentage >= 80:
                    status = HealthStatus.WARNING
                    message = f"API quota warning: {usage_percentage:.1f}% used"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"API quota healthy: {usage_percentage:.1f}% used"
                
                return HealthCheck(
                    component="api_quota",
                    status=status,
                    message=message,
                    timestamp=datetime.now(),
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheck(
                component="api_quota",
                status=HealthStatus.UNKNOWN,
                message=f"API quota check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_feature_computation_health(self) -> HealthCheck:
        """Check feature computation health.
        
        Returns:
            Feature computation health check result
        """
        try:
            with self.db_connection.cursor() as cursor:
                # Check recent computation success rate
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_computations,
                        COUNT(*) FILTER (WHERE status = 'completed') as successful,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed,
                        AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration
                    FROM feature_computation_log 
                    WHERE computed_at >= NOW() - INTERVAL '24 hours'
                """)
                
                result = cursor.fetchone()
                total = result[0] or 0
                successful = result[1] or 0
                failed = result[2] or 0
                avg_duration = result[3] or 0
                
                success_rate = (successful / total * 100) if total > 0 else 0
                
                # Check pending computations
                cursor.execute("""
                    SELECT COUNT(*) as pending_count
                    FROM feature_computation_queue 
                    WHERE status = 'pending'
                """)
                pending_count = cursor.fetchone()[0]
                
                metrics = {
                    'total_computations_24h': total,
                    'successful_computations': successful,
                    'failed_computations': failed,
                    'success_rate': success_rate,
                    'avg_duration_seconds': avg_duration,
                    'pending_computations': pending_count
                }
                
                # Determine status
                if success_rate < 70 or pending_count > 1000:
                    status = HealthStatus.CRITICAL
                    message = f"Feature computation critical: {success_rate:.1f}% success rate, {pending_count} pending"
                elif success_rate < 85 or pending_count > 500:
                    status = HealthStatus.WARNING
                    message = f"Feature computation warning: {success_rate:.1f}% success rate, {pending_count} pending"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Feature computation healthy: {success_rate:.1f}% success rate"
                
                return HealthCheck(
                    component="feature_computation",
                    status=status,
                    message=message,
                    timestamp=datetime.now(),
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheck(
                component="feature_computation",
                status=HealthStatus.UNKNOWN,
                message=f"Feature computation check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_disk_space_health(self) -> HealthCheck:
        """Check disk space health.
        
        Returns:
            Disk space health check result
        """
        try:
            # Get disk usage
            disk_usage = psutil.disk_usage('/')
            
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            usage_percentage = (used_gb / total_gb) * 100
            
            metrics = {
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'free_gb': round(free_gb, 2),
                'usage_percentage': round(usage_percentage, 2)
            }
            
            # Determine status based on thresholds
            if usage_percentage >= 95.0:  # Critical threshold
                status = HealthStatus.CRITICAL
                message = f"Disk space critical: {usage_percentage:.1f}% used"
            elif usage_percentage >= 85.0:  # Warning threshold
                status = HealthStatus.WARNING
                message = f"Disk space warning: {usage_percentage:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space healthy: {usage_percentage:.1f}% used"
            
            return HealthCheck(
                component="disk_space",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheck(
                component="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Disk space check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_memory_health(self) -> HealthCheck:
        """Check memory health.
        
        Returns:
            Memory health check result
        """
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            
            total_gb = memory.total / (1024**3)
            used_gb = memory.used / (1024**3)
            available_gb = memory.available / (1024**3)
            usage_percentage = memory.percent
            
            metrics = {
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'available_gb': round(available_gb, 2),
                'usage_percentage': round(usage_percentage, 2)
            }
            
            # Determine status
            if usage_percentage >= 90.0:  # Critical threshold
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {usage_percentage:.1f}%"
            elif usage_percentage >= 80.0:  # Warning threshold
                status = HealthStatus.WARNING
                message = f"Memory usage warning: {usage_percentage:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage healthy: {usage_percentage:.1f}%"
            
            return HealthCheck(
                component="memory",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheck(
                component="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_cpu_health(self) -> HealthCheck:
        """Check CPU health.
        
        Returns:
            CPU health check result
        """
        try:
            # Get CPU usage (average over 1 second)
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            metrics = {
                'cpu_percent': round(cpu_percent, 2),
                'cpu_count': cpu_count,
                'load_avg_1m': round(load_avg[0], 2),
                'load_avg_5m': round(load_avg[1], 2),
                'load_avg_15m': round(load_avg[2], 2)
            }
            
            # Determine status
            if cpu_percent >= 95.0:  # Critical threshold
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent >= 85.0:  # Warning threshold
                status = HealthStatus.WARNING
                message = f"CPU usage warning: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage healthy: {cpu_percent:.1f}%"
            
            return HealthCheck(
                component="cpu",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheck(
                component="cpu",
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_data_quality_health(self) -> HealthCheck:
        """Check data quality health.
        
        Returns:
            Data quality health check result
        """
        try:
            with self.db_connection.cursor() as cursor:
                # Check average quality score
                cursor.execute("""
                    SELECT 
                        AVG(quality_score) as avg_quality,
                        AVG(feature_completeness) as avg_completeness,
                        COUNT(*) as total_features,
                        COUNT(*) FILTER (WHERE quality_score >= 0.8) as high_quality_count
                    FROM pre_computed_features 
                    WHERE computed_at >= NOW() - INTERVAL '7 days'
                    AND computation_status = 'completed'
                """)
                
                result = cursor.fetchone()
                avg_quality = result[0] or 0
                avg_completeness = result[1] or 0
                total_features = result[2] or 0
                high_quality_count = result[3] or 0
                
                high_quality_percentage = (high_quality_count / total_features * 100) if total_features > 0 else 0
                
                metrics = {
                    'avg_quality_score': round(avg_quality, 3),
                    'avg_completeness': round(avg_completeness, 3),
                    'total_features': total_features,
                    'high_quality_percentage': round(high_quality_percentage, 2)
                }
                
                # Determine status
                if avg_quality < 0.6 or high_quality_percentage < 60:
                    status = HealthStatus.CRITICAL
                    message = f"Data quality critical: {avg_quality:.3f} avg quality, {high_quality_percentage:.1f}% high quality"
                elif avg_quality < 0.75 or high_quality_percentage < 80:
                    status = HealthStatus.WARNING
                    message = f"Data quality warning: {avg_quality:.3f} avg quality, {high_quality_percentage:.1f}% high quality"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Data quality healthy: {avg_quality:.3f} avg quality"
                
                return HealthCheck(
                    component="data_quality",
                    status=status,
                    message=message,
                    timestamp=datetime.now(),
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheck(
                component="data_quality",
                status=HealthStatus.UNKNOWN,
                message=f"Data quality check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_training_performance_health(self) -> HealthCheck:
        """Check training performance health.
        
        Returns:
            Training performance health check result
        """
        try:
            with self.db_connection.cursor() as cursor:
                # Check recent training metrics (if available)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as training_runs,
                        AVG(CASE WHEN result->>'validation_rmse' IS NOT NULL 
                            THEN (result->>'validation_rmse')::float END) as avg_rmse,
                        AVG(CASE WHEN result->>'validation_r2' IS NOT NULL 
                            THEN (result->>'validation_r2')::float END) as avg_r2
                    FROM feature_computation_queue 
                    WHERE task_type = 'model_training'
                    AND status = 'completed'
                    AND completed_at >= NOW() - INTERVAL '30 days'
                """)
                
                result = cursor.fetchone()
                training_runs = result[0] or 0
                avg_rmse = result[1]
                avg_r2 = result[2]
                
                metrics = {
                    'training_runs_30d': training_runs,
                    'avg_rmse': round(avg_rmse, 4) if avg_rmse else None,
                    'avg_r2': round(avg_r2, 4) if avg_r2 else None
                }
                
                # Determine status
                if training_runs == 0:
                    status = HealthStatus.WARNING
                    message = "No recent training runs"
                elif avg_rmse and avg_rmse > 1.5:  # Threshold from config
                    status = HealthStatus.WARNING
                    message = f"Training performance warning: RMSE {avg_rmse:.4f}"
                elif avg_r2 and avg_r2 < 0.3:  # Threshold from config
                    status = HealthStatus.WARNING
                    message = f"Training performance warning: R² {avg_r2:.4f}"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Training performance healthy"
                
                return HealthCheck(
                    component="training_performance",
                    status=status,
                    message=message,
                    timestamp=datetime.now(),
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheck(
                component="training_performance",
                status=HealthStatus.UNKNOWN,
                message=f"Training performance check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _persist_health_checks(self, health_checks: Dict[str, HealthCheck]) -> None:
        """Persist health check results to database.
        
        Args:
            health_checks: Health check results to persist
        """
        try:
            with self.db_connection.cursor() as cursor:
                for component, health_check in health_checks.items():
                    cursor.execute("""
                        INSERT INTO data_quality_metrics 
                        (component, status, message, metrics, response_time_ms, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        health_check.component,
                        health_check.status.value,
                        health_check.message,
                        json.dumps(health_check.metrics),
                        health_check.response_time_ms,
                        health_check.timestamp
                    ))
                
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to persist health checks: {str(e)}")
    
    async def _evaluate_health_alerts(self, health_checks: Dict[str, HealthCheck]) -> None:
        """Evaluate health checks and generate alerts.
        
        Args:
            health_checks: Health check results to evaluate
        """
        for component, health_check in health_checks.items():
            alert_id = f"{component}_health"
            
            # Check if we should create an alert
            if health_check.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                # Check if alert already exists
                if alert_id not in self.active_alerts or self.active_alerts[alert_id].resolved:
                    severity = AlertSeverity.CRITICAL if health_check.status == HealthStatus.CRITICAL else AlertSeverity.WARNING
                    
                    alert = Alert(
                        id=alert_id,
                        component=component,
                        severity=severity,
                        title=f"{component.title()} Health Alert",
                        message=health_check.message,
                        timestamp=datetime.now(),
                        metadata=health_check.metrics
                    )
                    
                    self.active_alerts[alert_id] = alert
                    await self._send_alert(alert)
                    
            elif health_check.status == HealthStatus.HEALTHY:
                # Resolve existing alert if it exists
                if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
                    self.active_alerts[alert_id].resolved = True
                    self.active_alerts[alert_id].resolved_at = datetime.now()
                    
                    # Send resolution notification
                    resolution_alert = Alert(
                        id=f"{alert_id}_resolved",
                        component=component,
                        severity=AlertSeverity.INFO,
                        title=f"{component.title()} Health Restored",
                        message=f"{component} is now healthy: {health_check.message}",
                        timestamp=datetime.now()
                    )
                    
                    await self._send_alert(resolution_alert)
    
    async def _send_alert(self, alert: Alert) -> None:
        """Send alert notification.
        
        Args:
            alert: Alert to send
        """
        try:
            # Log alert
            self.logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
            
            # Send email notification if configured
            if self.config.notifications.email.enabled:
                await self._send_email_alert(alert)
            
            # Send SMS notification for critical alerts if configured
            if (alert.severity == AlertSeverity.CRITICAL and 
                self.config.notifications.sms.enabled):
                await self._send_sms_alert(alert)
            
            # Persist alert to database
            await self._persist_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to send alert {alert.id}: {str(e)}")
    
    async def _send_email_alert(self, alert: Alert) -> None:
        """Send email alert.
        
        Args:
            alert: Alert to send via email
        """
        try:
            email_config = self.config.notifications.email
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config.from_address
            msg['To'] = ', '.join(email_config.to_addresses)
            msg['Subject'] = f"[FormFinder2] {alert.title}"
            
            # Create email body
            body = f"""
            Alert Details:
            
            Component: {alert.component}
            Severity: {alert.severity.value.upper()}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Message: {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}
            
            --
            FormFinder2 Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(email_config.smtp_server, email_config.smtp_port) as server:
                if email_config.use_tls:
                    server.starttls()
                if email_config.username and email_config.password:
                    server.login(email_config.username, email_config.password)
                
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent for {alert.id}")
            
        except Exception as e:
            raise NotificationError(f"Failed to send email alert: {str(e)}")
    
    async def _send_sms_alert(self, alert: Alert) -> None:
        """Send SMS alert.
        
        Args:
            alert: Alert to send via SMS
        """
        try:
            # SMS implementation would go here
            # This is a placeholder for SMS functionality
            self.logger.info(f"SMS alert would be sent for {alert.id}")
            
        except Exception as e:
            raise NotificationError(f"Failed to send SMS alert: {str(e)}")
    
    async def _persist_alert(self, alert: Alert) -> None:
        """Persist alert to database.
        
        Args:
            alert: Alert to persist
        """
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO notification_logs 
                    (alert_id, component, severity, title, message, timestamp, resolved, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (alert_id) DO UPDATE SET
                        resolved = EXCLUDED.resolved,
                        resolved_at = EXCLUDED.resolved_at
                """, (
                    alert.id,
                    alert.component,
                    alert.severity.value,
                    alert.title,
                    alert.message,
                    alert.timestamp,
                    alert.resolved,
                    json.dumps(alert.metadata)
                ))
                
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to persist alert {alert.id}: {str(e)}")
    
    def collect_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect current performance metrics.
        
        Returns:
            List of performance metrics
        """
        metrics = []
        timestamp = datetime.now()
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics.extend([
                PerformanceMetric(
                    name="cpu_usage_percent",
                    value=cpu_percent,
                    unit="percent",
                    timestamp=timestamp,
                    component="system"
                ),
                PerformanceMetric(
                    name="memory_usage_percent",
                    value=memory.percent,
                    unit="percent",
                    timestamp=timestamp,
                    component="system"
                ),
                PerformanceMetric(
                    name="disk_usage_percent",
                    value=(disk.used / disk.total) * 100,
                    unit="percent",
                    timestamp=timestamp,
                    component="system"
                )
            ])
            
            # Database metrics (if connected)
            if self.db_connection:
                with self.db_connection.cursor() as cursor:
                    # Active connections
                    cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                    active_connections = cursor.fetchone()[0]
                    
                    metrics.append(PerformanceMetric(
                        name="active_database_connections",
                        value=active_connections,
                        unit="count",
                        timestamp=timestamp,
                        component="database"
                    ))
            
            # Store metrics
            self.performance_metrics.extend(metrics)
            
            # Keep only recent metrics (last 1000)
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-1000:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {str(e)}")
            return []
    
    async def start_monitoring(self, check_interval: int = 300) -> None:
        """Start continuous monitoring.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.is_monitoring = True
        self.logger.info(f"Starting continuous monitoring (interval: {check_interval}s)")
        
        try:
            while self.is_monitoring:
                # Run health checks
                await self.run_health_checks()
                
                # Collect performance metrics
                self.collect_performance_metrics()
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
        except Exception as e:
            self.logger.error(f"Monitoring error: {str(e)}")
            raise HealthCheckError(f"Monitoring failed: {str(e)}")
        
        finally:
            self.is_monitoring = False
            self.logger.info("Monitoring stopped")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.is_monitoring = False
        self.logger.info("Monitoring stop requested")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns:
            System status information
        """
        # Overall health
        health_statuses = [check.status for check in self.health_checks.values()]
        
        if HealthStatus.CRITICAL in health_statuses:
            overall_health = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in health_statuses:
            overall_health = HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in health_statuses:
            overall_health = HealthStatus.UNKNOWN
        else:
            overall_health = HealthStatus.HEALTHY
        
        # Alert summary
        active_alert_count = sum(1 for alert in self.active_alerts.values() if not alert.resolved)
        critical_alert_count = sum(
            1 for alert in self.active_alerts.values() 
            if not alert.resolved and alert.severity == AlertSeverity.CRITICAL
        )
        
        return {
            'overall_health': overall_health.value,
            'timestamp': datetime.now().isoformat(),
            'is_monitoring': self.is_monitoring,
            'health_checks': {
                component: {
                    'status': check.status.value,
                    'message': check.message,
                    'response_time_ms': check.response_time_ms,
                    'timestamp': check.timestamp.isoformat()
                }
                for component, check in self.health_checks.items()
            },
            'alerts': {
                'active_count': active_alert_count,
                'critical_count': critical_alert_count,
                'total_count': len(self.active_alerts)
            },
            'performance_metrics_count': len(self.performance_metrics)
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_to_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_database_connection()


class MetricsCollector:
    """Simple metrics collector for tracking system metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.counters = {}
    
    def increment(self, metric_name: str, value: int = 1):
        """Increment a counter metric."""
        if metric_name not in self.counters:
            self.counters[metric_name] = 0
        self.counters[metric_name] += value
    
    def gauge(self, metric_name: str, value: float):
        """Set a gauge metric value."""
        self.metrics[metric_name] = value
    
    def get_metric(self, metric_name: str):
        """Get a metric value."""
        return self.metrics.get(metric_name) or self.counters.get(metric_name, 0)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {**self.metrics, **self.counters}


async def main():
    """Main monitoring entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FormFinder2 System Monitor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run health checks once and exit")
    
    args = parser.parse_args()
    
    with SystemMonitor(args.config) as monitor:
        if args.once:
            # Run health checks once
            health_results = await monitor.run_health_checks()
            
            print("\nHealth Check Results:")
            print("=" * 50)
            for component, result in health_results.items():
                print(f"{component}: {result.status.value} - {result.message}")
            
            # Print system status
            status = monitor.get_system_status()
            print(f"\nOverall Health: {status['overall_health']}")
            print(f"Active Alerts: {status['alerts']['active_count']}")
        else:
            # Start continuous monitoring
            await monitor.start_monitoring(check_interval=args.interval)


if __name__ == "__main__":
    asyncio.run(main())