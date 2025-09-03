#!/usr/bin/env python3
"""
FormFinder Scheduler

Provides automated scheduling for data collection, feature computation,
and model training workflows. Supports cron-based scheduling with
robust error handling and monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import json
import signal
import sys
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from sqlalchemy import create_engine, text

from formfinder.config import get_config
from formfinder.database import get_db_session
from orchestrator import WorkflowOrchestrator, create_daily_workflow
from data_quality_checker import DataQualityChecker


class JobStatus(Enum):
    """Job execution status."""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobType(Enum):
    """Types of scheduled jobs."""
    DATA_COLLECTION = "data_collection"
    FEATURE_COMPUTATION = "feature_computation"
    MODEL_TRAINING = "model_training"
    DATA_QUALITY_CHECK = "data_quality_check"
    FULL_WORKFLOW = "full_workflow"
    MAINTENANCE = "maintenance"


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""
    id: str
    name: str
    job_type: JobType
    function: Callable
    cron_expression: str
    enabled: bool = True
    max_instances: int = 1
    timeout_minutes: int = 60
    retry_count: int = 3
    last_run: Optional[datetime] = None
    last_status: Optional[JobStatus] = None
    last_error: Optional[str] = None


class FormFinderScheduler:
    """Main scheduler for FormFinder workflows."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.scheduler = AsyncIOScheduler()
        self.jobs: Dict[str, ScheduledJob] = {}
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup job event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
    
    def add_job(self, job: ScheduledJob) -> None:
        """Add a scheduled job."""
        if not job.enabled:
            self.logger.info(f"Job {job.name} is disabled, skipping")
            return
        
        try:
            # Parse cron expression
            trigger = CronTrigger.from_crontab(job.cron_expression)
            
            # Add job to scheduler
            self.scheduler.add_job(
                func=self._execute_job_wrapper,
                trigger=trigger,
                args=[job],
                id=job.id,
                name=job.name,
                max_instances=job.max_instances,
                replace_existing=True
            )
            
            self.jobs[job.id] = job
            self.logger.info(f"Added scheduled job: {job.name} ({job.cron_expression})")
            
        except Exception as e:
            self.logger.error(f"Failed to add job {job.name}: {e}")
    
    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job."""
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self.jobs:
                del self.jobs[job_id]
            self.logger.info(f"Removed job: {job_id}")
        except Exception as e:
            self.logger.error(f"Failed to remove job {job_id}: {e}")
    
    async def start(self) -> None:
        """Start the scheduler."""
        try:
            self.logger.info("Starting FormFinder Scheduler")
            self.scheduler.start()
            self.running = True
            
            # Log scheduled jobs
            for job in self.scheduler.get_jobs():
                next_run = job.next_run_time
                self.logger.info(f"Job '{job.name}' next run: {next_run}")
            
            # Keep the scheduler running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self.logger.info("Stopping FormFinder Scheduler")
        self.running = False
        self.scheduler.shutdown(wait=True)
    
    async def _execute_job_wrapper(self, job: ScheduledJob) -> None:
        """Wrapper for job execution with error handling and logging."""
        job.last_run = datetime.now()
        
        try:
            self.logger.info(f"Starting job: {job.name}")
            
            # Set timeout
            timeout_seconds = job.timeout_minutes * 60
            
            # Execute job with timeout
            if asyncio.iscoroutinefunction(job.function):
                result = await asyncio.wait_for(
                    job.function(), timeout=timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(job.function), timeout=timeout_seconds
                )
            
            job.last_status = JobStatus.COMPLETED
            job.last_error = None
            
            self.logger.info(f"Job {job.name} completed successfully")
            
            # Log job execution
            await self._log_job_execution(job, JobStatus.COMPLETED, result=result)
            
        except asyncio.TimeoutError:
            error_msg = f"Job {job.name} timed out after {job.timeout_minutes} minutes"
            job.last_status = JobStatus.FAILED
            job.last_error = error_msg
            self.logger.error(error_msg)
            await self._log_job_execution(job, JobStatus.FAILED, error=error_msg)
            
        except Exception as e:
            error_msg = f"Job {job.name} failed: {str(e)}"
            job.last_status = JobStatus.FAILED
            job.last_error = error_msg
            self.logger.error(error_msg)
            await self._log_job_execution(job, JobStatus.FAILED, error=error_msg)
    
    def _job_executed(self, event) -> None:
        """Handle job execution events."""
        job_id = event.job_id
        if job_id in self.jobs:
            self.logger.debug(f"Job {job_id} executed successfully")
    
    def _job_error(self, event) -> None:
        """Handle job error events."""
        job_id = event.job_id
        exception = event.exception
        if job_id in self.jobs:
            self.logger.error(f"Job {job_id} failed with exception: {exception}")
    
    async def _log_job_execution(self, job: ScheduledJob, status: JobStatus, 
                                result: Any = None, error: str = None) -> None:
        """Log job execution to database."""
        try:
            with get_db_session() as session:
                log_entry = {
                    'job_id': job.id,
                    'job_name': job.name,
                    'job_type': job.job_type.value,
                    'status': status.value,
                    'start_time': job.last_run,
                    'end_time': datetime.now(),
                    'duration': (datetime.now() - job.last_run).total_seconds(),
                    'result': json.dumps(result, default=str) if result else None,
                    'error_message': error
                }
                
                query = text("""
                    INSERT INTO job_executions 
                    (job_id, job_name, job_type, status, start_time, end_time, 
                     duration, result, error_message)
                    VALUES (:job_id, :job_name, :job_type, :status, :start_time, 
                            :end_time, :duration, :result, :error_message)
                """)
                
                session.execute(query, log_entry)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log job execution: {e}")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        scheduler_job = self.scheduler.get_job(job_id)
        
        return {
            'id': job.id,
            'name': job.name,
            'type': job.job_type.value,
            'enabled': job.enabled,
            'cron_expression': job.cron_expression,
            'last_run': job.last_run,
            'last_status': job.last_status.value if job.last_status else None,
            'last_error': job.last_error,
            'next_run': scheduler_job.next_run_time if scheduler_job else None
        }
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs with their status."""
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]


def create_default_schedule() -> FormFinderScheduler:
    """Create the default FormFinder schedule."""
    scheduler = FormFinderScheduler()
    
    # Daily full workflow at 2 AM
    daily_workflow = ScheduledJob(
        id="daily_workflow",
        name="Daily Data Collection and Training",
        job_type=JobType.FULL_WORKFLOW,
        function=run_daily_workflow,
        cron_expression="0 2 * * *",  # 2 AM daily
        timeout_minutes=120
    )
    
    # Data quality check every 6 hours
    quality_check = ScheduledJob(
        id="quality_check",
        name="Data Quality Check",
        job_type=JobType.DATA_QUALITY_CHECK,
        function=run_quality_check,
        cron_expression="0 */6 * * *",  # Every 6 hours
        timeout_minutes=15
    )
    
    # Feature computation every 4 hours during active periods
    feature_computation = ScheduledJob(
        id="feature_computation",
        name="Feature Computation",
        job_type=JobType.FEATURE_COMPUTATION,
        function=run_feature_computation,
        cron_expression="0 */4 * * *",  # Every 4 hours
        timeout_minutes=60
    )
    
    # Model training twice daily
    model_training = ScheduledJob(
        id="model_training",
        name="Model Training",
        job_type=JobType.MODEL_TRAINING,
        function=run_model_training,
        cron_expression="0 6,18 * * *",  # 6 AM and 6 PM
        timeout_minutes=30
    )
    
    # Weekly maintenance on Sunday at 1 AM
    maintenance = ScheduledJob(
        id="weekly_maintenance",
        name="Weekly Maintenance",
        job_type=JobType.MAINTENANCE,
        function=run_maintenance,
        cron_expression="0 1 * * 0",  # Sunday 1 AM
        timeout_minutes=60
    )
    
    scheduler.add_job(daily_workflow)
    scheduler.add_job(quality_check)
    scheduler.add_job(feature_computation)
    scheduler.add_job(model_training)
    scheduler.add_job(maintenance)
    
    return scheduler


async def run_daily_workflow() -> Dict[str, Any]:
    """Run the complete daily workflow."""
    try:
        orchestrator = create_daily_workflow()
        result = await orchestrator.execute_workflow()
        return result
    except Exception as e:
        logging.error(f"Daily workflow failed: {e}")
        raise


def run_quality_check() -> Dict[str, Any]:
    """Run data quality check."""
    try:
        with get_db_session() as session:
            checker = DataQualityChecker(session)
            results = checker.run_comprehensive_check()
            return {'status': 'completed', 'results': results}
    except Exception as e:
        logging.error(f"Quality check failed: {e}")
        raise


def run_feature_computation() -> Dict[str, Any]:
    """Run feature computation."""
    try:
        from precompute_features import main as precompute_main
        result = precompute_main()
        return {'status': 'completed', 'result': result}
    except Exception as e:
        logging.error(f"Feature computation failed: {e}")
        raise


def run_model_training() -> Dict[str, Any]:
    """Run model training."""
    try:
        from train_model import main as train_main
        result = train_main()
        return {'status': 'completed', 'result': result}
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise


def run_maintenance() -> Dict[str, Any]:
    """Run weekly maintenance tasks."""
    try:
        tasks_completed = []
        
        # Clean old logs
        with get_db_session() as session:
            # Remove job execution logs older than 30 days
            cleanup_query = text("""
                DELETE FROM job_executions 
                WHERE start_time < :cutoff_date
            """)
            cutoff_date = datetime.now() - timedelta(days=30)
            result = session.execute(cleanup_query, {'cutoff_date': cutoff_date})
            session.commit()
            tasks_completed.append(f"Cleaned {result.rowcount} old job execution logs")
            
            # Remove workflow execution logs older than 30 days
            workflow_cleanup = text("""
                DELETE FROM workflow_executions 
                WHERE timestamp < :cutoff_date
            """)
            result = session.execute(workflow_cleanup, {'cutoff_date': cutoff_date})
            session.commit()
            tasks_completed.append(f"Cleaned {result.rowcount} old workflow execution logs")
        
        return {
            'status': 'completed',
            'tasks_completed': tasks_completed
        }
        
    except Exception as e:
        logging.error(f"Maintenance failed: {e}")
        raise


async def main():
    """Main scheduler entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start scheduler
    scheduler = create_default_schedule()
    
    try:
        await scheduler.start()
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    finally:
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())