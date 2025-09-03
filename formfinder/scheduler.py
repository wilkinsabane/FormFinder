"""Scheduler Module for FormFinder2

This module provides comprehensive scheduling capabilities for the FormFinder2 system,
including cron-based task scheduling, job management, and workflow coordination.

Author: FormFinder2 Team
Created: 2025-01-01
Purpose: Task scheduling and automation
"""

import logging
import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
from croniter import croniter
import psycopg2
from psycopg2.extras import RealDictCursor

from .config import load_config
from .exceptions import (
    SchedulerError, TaskExecutionError, DatabaseError,
    ConfigurationError
)
from .feature_precomputer import FeaturePrecomputer
from .training_engine import TrainingEngine
from .monitoring import SystemMonitor
from .orchestrator import WorkflowOrchestrator, TaskPriority


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class JobType(Enum):
    """Types of scheduled jobs."""
    FEATURE_COMPUTATION = "feature_computation"
    HISTORICAL_BACKFILL = "historical_backfill"
    MODEL_TRAINING = "model_training"
    DATA_QUALITY_CHECK = "data_quality_check"
    CACHE_MAINTENANCE = "cache_maintenance"
    HEALTH_CHECK = "health_check"
    MONITORING = "monitoring"
    CLEANUP = "cleanup"
    BACKUP = "backup"
    CUSTOM = "custom"


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""
    id: str
    name: str
    job_type: JobType
    cron_expression: str
    function: Callable
    enabled: bool = True
    max_runtime_minutes: int = 60
    retry_count: int = 3
    retry_delay_minutes: int = 5
    timeout_minutes: int = 30
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Runtime state
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    current_attempt: int = 0
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Calculate next run time."""
        if self.enabled:
            self.calculate_next_run()
    
    def calculate_next_run(self, base_time: Optional[datetime] = None) -> None:
        """Calculate the next run time based on cron expression.
        
        Args:
            base_time: Base time for calculation (defaults to now)
        """
        if not self.enabled:
            self.next_run = None
            return
        
        base = base_time or datetime.now()
        try:
            cron = croniter(self.cron_expression, base)
            self.next_run = cron.get_next(datetime)
        except Exception as e:
            raise SchedulerError(f"Invalid cron expression '{self.cron_expression}': {str(e)}")
    
    def is_due(self, current_time: Optional[datetime] = None) -> bool:
        """Check if job is due for execution.
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            True if job is due for execution
        """
        if not self.enabled or not self.next_run:
            return False
        
        current = current_time or datetime.now()
        return current >= self.next_run
    
    def can_run(self) -> bool:
        """Check if job can run (not already running, within retry limits).
        
        Returns:
            True if job can run
        """
        return (
            self.enabled and
            self.status != JobStatus.RUNNING and
            self.current_attempt < self.retry_count
        )
    
    def reset_for_retry(self) -> None:
        """Reset job state for retry."""
        self.status = JobStatus.PENDING
        self.error_message = None


@dataclass
class JobExecution:
    """Represents a job execution record."""
    job_id: str
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: JobStatus = JobStatus.RUNNING
    attempt_number: int = 1
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Advanced task scheduler with cron support and job management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the task scheduler.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.db_connection = None
        
        # Scheduler state
        self.jobs: Dict[str, ScheduledJob] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_executions: List[JobExecution] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Components
        self.feature_precomputer = None
        self.training_engine = None
        self.system_monitor = None
        self.orchestrator = None
        
        # Scheduler configuration
        self.scheduler_config = self.config.scheduler
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize jobs from configuration
        self._initialize_jobs()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
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
            self.logger.info("Database connection established for scheduler")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to database: {str(e)}")
    
    def close_database_connection(self) -> None:
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.logger.info("Database connection closed")
    
    def _initialize_components(self) -> None:
        """Initialize system components."""
        try:
            self.feature_precomputer = FeaturePrecomputer()
            self.training_engine = TrainingEngine()
            self.system_monitor = SystemMonitor()
            self.orchestrator = WorkflowOrchestrator()
            
            self.logger.info("System components initialized")
        except Exception as e:
            raise SchedulerError(f"Failed to initialize components: {str(e)}")
    
    def _initialize_jobs(self) -> None:
        """Initialize jobs from configuration."""
        try:
            # Feature computation job
            self.add_job(ScheduledJob(
                id="feature_computation",
                name="Daily Feature Computation",
                job_type=JobType.FEATURE_COMPUTATION,
                cron_expression=self.scheduler_config.feature_computation_cron,
                function=self._run_feature_computation,
                max_runtime_minutes=120,
                timeout_minutes=90,
                tags=["daily", "features"]
            ))
            
            # Historical backfill job
            self.add_job(ScheduledJob(
                id="historical_backfill",
                name="Historical Data Backfill",
                job_type=JobType.HISTORICAL_BACKFILL,
                cron_expression=self.scheduler_config.historical_backfill_cron,
                function=self._run_historical_backfill,
                max_runtime_minutes=240,
                timeout_minutes=180,
                tags=["weekly", "backfill"]
            ))
            
            # Model training job
            self.add_job(ScheduledJob(
                id="model_training",
                name="Model Training",
                job_type=JobType.MODEL_TRAINING,
                cron_expression="0 4 * * 0",  # Weekly on Sunday at 4 AM
                function=self._run_model_training,
                max_runtime_minutes=60,
                timeout_minutes=45,
                tags=["training", "model"],
                dependencies=["feature_computation"]
            ))
            
            # Data quality check job
            self.add_job(ScheduledJob(
                id="data_quality_check",
                name="Data Quality Check",
                job_type=JobType.DATA_QUALITY_CHECK,
                cron_expression=self.scheduler_config.quality_check_cron,
                function=self._run_data_quality_check,
                max_runtime_minutes=30,
                timeout_minutes=20,
                tags=["quality", "validation"]
            ))
            
            # Cache maintenance job
            self.add_job(ScheduledJob(
                id="cache_maintenance",
                name="Cache Maintenance",
                job_type=JobType.CACHE_MAINTENANCE,
                cron_expression=self.scheduler_config.cache_cleanup_cron,
                function=self._run_cache_maintenance,
                max_runtime_minutes=45,
                timeout_minutes=30,
                tags=["maintenance", "cache"]
            ))
            
            # Health check job
            self.add_job(ScheduledJob(
                id="health_check",
                name="System Health Check",
                job_type=JobType.HEALTH_CHECK,
                cron_expression="*/5 * * * *",  # Every 5 minutes
                function=self._run_health_check,
                max_runtime_minutes=15,
                timeout_minutes=10,
                tags=["health", "monitoring"]
            ))
            
            # Monitoring job
            self.add_job(ScheduledJob(
                id="monitoring",
                name="System Monitoring",
                job_type=JobType.MONITORING,
                cron_expression=self.scheduler_config.metrics_aggregation_cron,
                function=self._run_monitoring,
                max_runtime_minutes=10,
                timeout_minutes=5,
                tags=["monitoring", "metrics"]
            ))
            
            # Cleanup job
            self.add_job(ScheduledJob(
                id="cleanup",
                name="System Cleanup",
                job_type=JobType.CLEANUP,
                cron_expression="0 0 * * 0",  # Weekly on Sunday at midnight
                function=self._run_cleanup,
                max_runtime_minutes=30,
                timeout_minutes=20,
                tags=["cleanup", "maintenance"]
            ))
            
            self.logger.info(f"Initialized {len(self.jobs)} scheduled jobs")
            
        except Exception as e:
            raise SchedulerError(f"Failed to initialize jobs: {str(e)}")
    
    def add_job(self, job: ScheduledJob) -> None:
        """Add a job to the scheduler.
        
        Args:
            job: Job to add
        """
        if job.id in self.jobs:
            raise SchedulerError(f"Job with ID '{job.id}' already exists")
        
        self.jobs[job.id] = job
        self.logger.info(f"Added job '{job.id}' with schedule '{job.cron_expression}'")
    
    def remove_job(self, job_id: str) -> None:
        """Remove a job from the scheduler.
        
        Args:
            job_id: ID of job to remove
        """
        if job_id not in self.jobs:
            raise SchedulerError(f"Job with ID '{job_id}' not found")
        
        # Cancel if running
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            del self.running_jobs[job_id]
        
        del self.jobs[job_id]
        self.logger.info(f"Removed job '{job_id}'")
    
    def enable_job(self, job_id: str) -> None:
        """Enable a job.
        
        Args:
            job_id: ID of job to enable
        """
        if job_id not in self.jobs:
            raise SchedulerError(f"Job with ID '{job_id}' not found")
        
        job = self.jobs[job_id]
        job.enabled = True
        job.calculate_next_run()
        
        self.logger.info(f"Enabled job '{job_id}'")
    
    def disable_job(self, job_id: str) -> None:
        """Disable a job.
        
        Args:
            job_id: ID of job to disable
        """
        if job_id not in self.jobs:
            raise SchedulerError(f"Job with ID '{job_id}' not found")
        
        job = self.jobs[job_id]
        job.enabled = False
        job.next_run = None
        
        # Cancel if running
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            del self.running_jobs[job_id]
        
        self.logger.info(f"Disabled job '{job_id}'")
    
    async def run_job_now(self, job_id: str, force: bool = False) -> JobExecution:
        """Run a job immediately.
        
        Args:
            job_id: ID of job to run
            force: Force run even if job is disabled or already running
            
        Returns:
            Job execution record
        """
        if job_id not in self.jobs:
            raise SchedulerError(f"Job with ID '{job_id}' not found")
        
        job = self.jobs[job_id]
        
        if not force:
            if not job.enabled:
                raise SchedulerError(f"Job '{job_id}' is disabled")
            
            if job_id in self.running_jobs:
                raise SchedulerError(f"Job '{job_id}' is already running")
        
        return await self._execute_job(job)
    
    async def _execute_job(self, job: ScheduledJob) -> JobExecution:
        """Execute a job.
        
        Args:
            job: Job to execute
            
        Returns:
            Job execution record
        """
        execution_id = f"{job.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution = JobExecution(
            job_id=job.id,
            execution_id=execution_id,
            started_at=datetime.now(),
            attempt_number=job.current_attempt + 1
        )
        
        self.job_executions.append(execution)
        job.current_attempt += 1
        job.status = JobStatus.RUNNING
        
        self.logger.info(f"Starting job '{job.id}' (attempt {execution.attempt_number})")
        
        try:
            # Create task with timeout
            task = asyncio.create_task(self._run_job_with_timeout(job, execution))
            self.running_jobs[job.id] = task
            
            # Wait for completion
            await task
            
        except asyncio.CancelledError:
            execution.status = JobStatus.CANCELLED
            execution.error_message = "Job was cancelled"
            job.status = JobStatus.CANCELLED
            
            self.logger.warning(f"Job '{job.id}' was cancelled")
            
        except Exception as e:
            execution.status = JobStatus.FAILED
            execution.error_message = str(e)
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            
            self.logger.error(f"Job '{job.id}' failed: {str(e)}")
            self.logger.debug(f"Job '{job.id}' traceback: {traceback.format_exc()}")
            
        finally:
            # Cleanup
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]
            
            execution.completed_at = datetime.now()
            execution.execution_time_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            
            job.execution_time_seconds = execution.execution_time_seconds
            job.last_run = execution.started_at
            
            # Calculate next run if successful or max retries reached
            if execution.status == JobStatus.COMPLETED or job.current_attempt >= job.retry_count:
                job.current_attempt = 0
                job.calculate_next_run()
                
                if execution.status == JobStatus.COMPLETED:
                    job.status = JobStatus.PENDING
                    job.error_message = None
            
            # Persist execution record
            await self._persist_job_execution(execution)
            
            self.logger.info(
                f"Job '{job.id}' completed with status {execution.status.value} "
                f"in {execution.execution_time_seconds:.2f}s"
            )
        
        return execution
    
    async def _run_job_with_timeout(self, job: ScheduledJob, execution: JobExecution) -> None:
        """Run job with timeout.
        
        Args:
            job: Job to run
            execution: Execution record
        """
        try:
            # Run job function with timeout
            await asyncio.wait_for(
                job.function(**job.parameters),
                timeout=job.timeout_minutes * 60
            )
            
            execution.status = JobStatus.COMPLETED
            job.status = JobStatus.COMPLETED
            
        except asyncio.TimeoutError:
            raise TaskExecutionError(f"Job '{job.id}' timed out after {job.timeout_minutes} minutes")
        except Exception as e:
            raise TaskExecutionError(f"Job '{job.id}' execution failed: {str(e)}")
    
    async def _persist_job_execution(self, execution: JobExecution) -> None:
        """Persist job execution record to database.
        
        Args:
            execution: Execution record to persist
        """
        try:
            if not self.db_connection:
                return
            
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO feature_computation_queue 
                    (task_id, task_type, priority, status, parameters, 
                     created_at, started_at, completed_at, error_message, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        completed_at = EXCLUDED.completed_at,
                        error_message = EXCLUDED.error_message,
                        metadata = EXCLUDED.metadata
                """, (
                    execution.execution_id,
                    'scheduled_job',
                    'normal',
                    execution.status.value,
                    json.dumps({'job_id': execution.job_id}),
                    execution.started_at,
                    execution.started_at,
                    execution.completed_at,
                    execution.error_message,
                    json.dumps(execution.metadata)
                ))
                
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to persist job execution {execution.execution_id}: {str(e)}")
    
    # Job implementation methods
    async def _run_feature_computation(self, **kwargs) -> None:
        """Run feature computation job."""
        if not self.feature_precomputer:
            self._initialize_components()
        
        # Get fixtures that need feature computation
        with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT f.id, f.home_team_id, f.away_team_id, f.utc_date, f.league_id
                FROM fixtures f
                LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
                WHERE f.utc_date >= CURRENT_DATE
                AND f.utc_date <= CURRENT_DATE + INTERVAL '7 days'
                AND (pcf.fixture_id IS NULL OR pcf.computation_status != 'completed')
                ORDER BY f.utc_date
                LIMIT 100
            """)
            
            fixtures = cursor.fetchall()
        
        if not fixtures:
            self.logger.info("No fixtures require feature computation")
            return
        
        self.logger.info(f"Computing features for {len(fixtures)} fixtures")
        
        # Process fixtures in batches
        batch_size = self.config.feature_computation.batch_processing.batch_size
        
        for i in range(0, len(fixtures), batch_size):
            batch = fixtures[i:i + batch_size]
            await self.feature_precomputer.compute_features_batch(batch)
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        self.logger.info("Feature computation completed")
    
    async def _run_historical_backfill(self, **kwargs) -> None:
        """Run historical data backfill job."""
        if not self.feature_precomputer:
            self._initialize_components()
        
        # Get historical fixtures that need backfill
        days_back = kwargs.get('days_back', 30)
        
        with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT f.id, f.home_team_id, f.away_team_id, f.utc_date, f.league_id
                FROM fixtures f
                LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
                WHERE f.utc_date >= CURRENT_DATE - INTERVAL '%s days'
                AND f.utc_date < CURRENT_DATE
                AND f.status = 'finished'
                AND (pcf.fixture_id IS NULL OR pcf.computation_status != 'completed')
                ORDER BY f.utc_date DESC
                LIMIT 500
            """, (days_back,))
            
            fixtures = cursor.fetchall()
        
        if not fixtures:
            self.logger.info("No historical fixtures require backfill")
            return
        
        self.logger.info(f"Backfilling features for {len(fixtures)} historical fixtures")
        
        # Process in smaller batches for historical data
        batch_size = self.config.feature_computation.batch_processing.batch_size // 2
        
        for i in range(0, len(fixtures), batch_size):
            batch = fixtures[i:i + batch_size]
            await self.feature_precomputer.compute_features_batch(batch)
            
            # Longer delay for historical processing
            await asyncio.sleep(2)
        
        self.logger.info("Historical backfill completed")
    
    async def _run_model_training(self, **kwargs) -> None:
        """Run model training job."""
        if not self.training_engine:
            self._initialize_components()
        
        self.logger.info("Starting model training")
        
        # Train the model
        model, metrics = await asyncio.to_thread(
            self.training_engine.train_model_pipeline
        )
        
        self.logger.info(f"Model training completed with RMSE: {metrics.get('validation_rmse', 'N/A')}")
    
    async def _run_data_quality_check(self, **kwargs) -> None:
        """Run data quality check job."""
        self.logger.info("Running data quality checks")
        
        with self.db_connection.cursor() as cursor:
            # Check for data quality issues
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_features,
                    AVG(quality_score) as avg_quality,
                    COUNT(*) FILTER (WHERE quality_score < 0.5) as low_quality_count
                FROM pre_computed_features 
                WHERE computed_at >= NOW() - INTERVAL '24 hours'
            """)
            
            result = cursor.fetchone()
            total_features = result[0] or 0
            avg_quality = result[1] or 0
            low_quality_count = result[2] or 0
            
            # Log quality metrics
            self.logger.info(
                f"Data quality check: {total_features} features, "
                f"avg quality {avg_quality:.3f}, {low_quality_count} low quality"
            )
            
            # Alert if quality is poor
            if avg_quality < 0.7 or (low_quality_count / total_features > 0.2 if total_features > 0 else False):
                self.logger.warning("Data quality issues detected")
    
    async def _run_cache_maintenance(self, **kwargs) -> None:
        """Run cache maintenance job."""
        self.logger.info("Running cache maintenance")
        
        with self.db_connection.cursor() as cursor:
            # Clean old cache entries
            cursor.execute("""
                DELETE FROM h2h_cache 
                WHERE last_fetched_at < NOW() - INTERVAL '30 days'
            """)
            
            deleted_h2h = cursor.rowcount
            
            # Clean old computation logs
            cursor.execute("""
                DELETE FROM feature_computation_log 
                WHERE computed_at < NOW() - INTERVAL '90 days'
            """)
            
            deleted_logs = cursor.rowcount
            
            # Clean old API usage logs
            cursor.execute("""
                DELETE FROM api_usage_log 
                WHERE created_at < NOW() - INTERVAL '60 days'
            """)
            
            deleted_api_logs = cursor.rowcount
            
            self.db_connection.commit()
            
            self.logger.info(
                f"Cache maintenance completed: {deleted_h2h} H2H entries, "
                f"{deleted_logs} computation logs, {deleted_api_logs} API logs deleted"
            )
    
    async def _run_health_check(self, **kwargs) -> None:
        """Run health check job."""
        if not self.system_monitor:
            self._initialize_components()
        
        self.logger.info("Running system health check")
        
        # Run health checks
        health_results = await self.system_monitor.run_health_checks()
        
        # Log results
        for component, result in health_results.items():
            self.logger.info(f"Health check {component}: {result.status.value}")
    
    async def _run_monitoring(self, **kwargs) -> None:
        """Run monitoring job."""
        if not self.system_monitor:
            self._initialize_components()
        
        # Collect performance metrics
        metrics = self.system_monitor.collect_performance_metrics()
        
        self.logger.info(f"Collected {len(metrics)} performance metrics")
    
    async def _run_cleanup(self, **kwargs) -> None:
        """Run cleanup job."""
        self.logger.info("Running system cleanup")
        
        with self.db_connection.cursor() as cursor:
            # Clean old job executions
            cursor.execute("""
                DELETE FROM feature_computation_queue 
                WHERE task_type = 'scheduled_job'
                AND completed_at < NOW() - INTERVAL '30 days'
            """)
            
            deleted_executions = cursor.rowcount
            
            # Clean old notification logs
            cursor.execute("""
                DELETE FROM notification_logs 
                WHERE timestamp < NOW() - INTERVAL '90 days'
            """)
            
            deleted_notifications = cursor.rowcount
            
            self.db_connection.commit()
            
            self.logger.info(
                f"Cleanup completed: {deleted_executions} job executions, "
                f"{deleted_notifications} notifications deleted"
            )
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self.is_running:
            raise SchedulerError("Scheduler is already running")
        
        self.logger.info("Starting task scheduler")
        
        # Connect to database
        if not self.db_connection:
            self.connect_to_database()
        
        # Initialize components
        self._initialize_components()
        
        self.is_running = True
        
        try:
            # Main scheduler loop
            while self.is_running:
                current_time = datetime.now()
                
                # Check for due jobs
                due_jobs = [
                    job for job in self.jobs.values()
                    if job.is_due(current_time) and job.can_run()
                ]
                
                # Execute due jobs
                for job in due_jobs:
                    if job.id not in self.running_jobs:
                        self.logger.info(f"Job '{job.id}' is due for execution")
                        
                        # Check dependencies
                        if self._check_job_dependencies(job):
                            task = asyncio.create_task(self._execute_job(job))
                            # Don't await here - let jobs run concurrently
                        else:
                            self.logger.warning(f"Job '{job.id}' dependencies not met, skipping")
                            job.status = JobStatus.SKIPPED
                            job.calculate_next_run()
                
                # Clean up completed tasks
                completed_tasks = [
                    job_id for job_id, task in self.running_jobs.items()
                    if task.done()
                ]
                
                for job_id in completed_tasks:
                    del self.running_jobs[job_id]
                
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break
                
                # Sleep for check interval
                await asyncio.sleep(self.scheduler_config.job_management.check_interval_seconds)
                
        except Exception as e:
            self.logger.error(f"Scheduler error: {str(e)}")
            raise SchedulerError(f"Scheduler failed: {str(e)}")
        
        finally:
            await self._cleanup_running_jobs()
            self.is_running = False
            self.logger.info("Task scheduler stopped")
    
    def _check_job_dependencies(self, job: ScheduledJob) -> bool:
        """Check if job dependencies are satisfied.
        
        Args:
            job: Job to check dependencies for
            
        Returns:
            True if all dependencies are satisfied
        """
        if not job.dependencies:
            return True
        
        for dep_job_id in job.dependencies:
            if dep_job_id not in self.jobs:
                self.logger.warning(f"Dependency job '{dep_job_id}' not found")
                return False
            
            dep_job = self.jobs[dep_job_id]
            
            # Check if dependency job has run recently and successfully
            if (not dep_job.last_run or 
                dep_job.status != JobStatus.COMPLETED or
                dep_job.last_run < datetime.now() - timedelta(hours=24)):
                return False
        
        return True
    
    async def _cleanup_running_jobs(self) -> None:
        """Cancel all running jobs."""
        if self.running_jobs:
            self.logger.info(f"Cancelling {len(self.running_jobs)} running jobs")
            
            for job_id, task in self.running_jobs.items():
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.running_jobs.values(), return_exceptions=True)
            
            self.running_jobs.clear()
    
    async def shutdown(self) -> None:
        """Shutdown the scheduler gracefully."""
        self.logger.info("Shutting down scheduler")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for main loop to finish
        await asyncio.sleep(1)
        
        # Cleanup
        await self._cleanup_running_jobs()
        self.close_database_connection()
        
        self.logger.info("Scheduler shutdown complete")
    
    def get_job_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get job status information.
        
        Args:
            job_id: Specific job ID (if None, returns all jobs)
            
        Returns:
            Job status information
        """
        if job_id:
            if job_id not in self.jobs:
                raise SchedulerError(f"Job with ID '{job_id}' not found")
            
            job = self.jobs[job_id]
            return {
                'id': job.id,
                'name': job.name,
                'type': job.job_type.value,
                'enabled': job.enabled,
                'status': job.status.value,
                'cron_expression': job.cron_expression,
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'current_attempt': job.current_attempt,
                'max_retries': job.retry_count,
                'execution_time_seconds': job.execution_time_seconds,
                'error_message': job.error_message,
                'is_running': job_id in self.running_jobs
            }
        else:
            return {
                'scheduler_status': 'running' if self.is_running else 'stopped',
                'total_jobs': len(self.jobs),
                'running_jobs': len(self.running_jobs),
                'enabled_jobs': sum(1 for job in self.jobs.values() if job.enabled),
                'jobs': {
                    job_id: self.get_job_status(job_id)
                    for job_id in self.jobs.keys()
                }
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_to_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_database_connection()


async def main():
    """Main scheduler entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FormFinder2 Task Scheduler")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--job", help="Run specific job immediately")
    parser.add_argument("--status", action="store_true", help="Show job status")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    scheduler = TaskScheduler(args.config)
    
    try:
        if args.status:
            # Show status
            status = scheduler.get_job_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.job:
            # Run specific job
            with scheduler:
                execution = await scheduler.run_job_now(args.job)
                print(f"Job '{args.job}' completed with status: {execution.status.value}")
                
        elif args.daemon:
            # Run as daemon
            with scheduler:
                await scheduler.start()
        else:
            # Show help
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())