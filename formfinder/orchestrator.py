"""Orchestration Module for FormFinder2

This module coordinates the data collection and training processes,
managing the workflow between feature pre-computation and model training.

Author: FormFinder2 Team
Created: 2025-01-01
Purpose: Workflow orchestration and coordination
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor

from .config import load_config
from .feature_precomputer import FeaturePrecomputer
from .training_engine import TrainingEngine, train_model_pipeline
from .exceptions import (
    WorkflowError, DatabaseError, FeatureComputationError,
    TrainingError, ConfigurationError
)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a workflow task."""
    id: str
    name: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    parameters: Dict[str, Any] = None
    dependencies: List[str] = None
    result: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.result is None:
            self.result = {}


class WorkflowOrchestrator:
    """Orchestrates data collection and training workflows."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.db_connection = None
        self.tasks: Dict[str, Task] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize task handlers
        self._register_task_handlers()
        
        # Workflow state
        self.is_running = False
        self.shutdown_requested = False
        
    def _register_task_handlers(self) -> None:
        """Register task handlers for different task types."""
        self.task_handlers = {
            'feature_computation': self._handle_feature_computation,
            'model_training': self._handle_model_training,
            'data_quality_check': self._handle_data_quality_check,
            'cache_maintenance': self._handle_cache_maintenance,
            'historical_backfill': self._handle_historical_backfill,
            'health_check': self._handle_health_check,
            'cleanup': self._handle_cleanup
        }
    
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
            self.logger.info("Database connection established")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to database: {str(e)}")
    
    def close_database_connection(self) -> None:
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.logger.info("Database connection closed")
    
    def create_task(self, 
                   name: str,
                   task_type: str,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   parameters: Dict[str, Any] = None,
                   dependencies: List[str] = None) -> str:
        """Create a new task.
        
        Args:
            name: Task name
            task_type: Type of task
            priority: Task priority
            parameters: Task parameters
            dependencies: List of task IDs this task depends on
            
        Returns:
            Task ID
        """
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.tasks)}"
        
        task = Task(
            id=task_id,
            name=name,
            task_type=task_type,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            parameters=parameters or {},
            dependencies=dependencies or []
        )
        
        self.tasks[task_id] = task
        self._persist_task(task)
        
        self.logger.info(f"Created task {task_id}: {name}")
        return task_id
    
    def _persist_task(self, task: Task) -> None:
        """Persist task to database.
        
        Args:
            task: Task to persist
        """
        if not self.db_connection:
            self.connect_to_database()
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO feature_computation_queue 
                    (task_id, task_name, task_type, priority, status, created_at, 
                     parameters, dependencies, retry_count, max_retries)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        started_at = EXCLUDED.started_at,
                        completed_at = EXCLUDED.completed_at,
                        error_message = EXCLUDED.error_message,
                        retry_count = EXCLUDED.retry_count,
                        result = EXCLUDED.result
                """, (
                    task.id, task.name, task.task_type, task.priority.value,
                    task.status.value, task.created_at,
                    json.dumps(task.parameters), json.dumps(task.dependencies),
                    task.retry_count, task.max_retries
                ))
            self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to persist task {task.id}: {str(e)}")
    
    def load_tasks_from_database(self) -> None:
        """Load pending tasks from database."""
        if not self.db_connection:
            self.connect_to_database()
        
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM feature_computation_queue 
                    WHERE status IN ('pending', 'running', 'retrying')
                    ORDER BY priority DESC, created_at ASC
                """)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    task = Task(
                        id=row['task_id'],
                        name=row['task_name'],
                        task_type=row['task_type'],
                        priority=TaskPriority(row['priority']),
                        status=TaskStatus(row['status']),
                        created_at=row['created_at'],
                        started_at=row.get('started_at'),
                        completed_at=row.get('completed_at'),
                        error_message=row.get('error_message'),
                        retry_count=row['retry_count'],
                        max_retries=row['max_retries'],
                        parameters=json.loads(row['parameters']) if row['parameters'] else {},
                        dependencies=json.loads(row['dependencies']) if row['dependencies'] else [],
                        result=json.loads(row['result']) if row['result'] else {}
                    )
                    self.tasks[task.id] = task
                
                self.logger.info(f"Loaded {len(rows)} tasks from database")
                
        except Exception as e:
            raise DatabaseError(f"Failed to load tasks from database: {str(e)}")
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute.
        
        Returns:
            List of ready tasks sorted by priority
        """
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    dependencies_met = False
                    break
                if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_tasks.append(task)
        
        # Sort by priority (highest first) then by creation time
        ready_tasks.sort(key=lambda t: (-t.priority.value, t.created_at))
        
        return ready_tasks
    
    async def execute_task(self, task: Task) -> None:
        """Execute a single task.
        
        Args:
            task: Task to execute
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self._persist_task(task)
        
        self.logger.info(f"Starting task {task.id}: {task.name}")
        
        try:
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise WorkflowError(f"No handler for task type: {task.task_type}")
            
            # Execute task
            result = await handler(task)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result or {}
            
            self.logger.info(f"Completed task {task.id}: {task.name}")
            
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {str(e)}")
            
            task.error_message = str(e)
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.RETRYING
                self.logger.info(f"Task {task.id} will be retried ({task.retry_count}/{task.max_retries})")
            else:
                task.status = TaskStatus.FAILED
                self.logger.error(f"Task {task.id} failed permanently after {task.retry_count} retries")
        
        finally:
            self._persist_task(task)
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _handle_feature_computation(self, task: Task) -> Dict[str, Any]:
        """Handle feature computation task.
        
        Args:
            task: Feature computation task
            
        Returns:
            Task result
        """
        try:
            precomputer = FeaturePrecomputer()
            
            # Get parameters
            fixture_ids = task.parameters.get('fixture_ids', [])
            batch_size = task.parameters.get('batch_size', 50)
            force_recompute = task.parameters.get('force_recompute', False)
            
            if not fixture_ids:
                # Get fixtures that need computation
                fixture_ids = precomputer.get_fixtures_needing_computation()
            
            # Process in batches
            total_processed = 0
            total_failed = 0
            
            for i in range(0, len(fixture_ids), batch_size):
                batch = fixture_ids[i:i + batch_size]
                results = await precomputer.compute_features_batch(
                    batch, force_recompute=force_recompute
                )
                
                total_processed += results['processed']
                total_failed += results['failed']
                
                # Update progress
                progress = (i + len(batch)) / len(fixture_ids) * 100
                self.logger.info(f"Feature computation progress: {progress:.1f}%")
            
            return {
                'total_fixtures': len(fixture_ids),
                'processed': total_processed,
                'failed': total_failed,
                'success_rate': total_processed / len(fixture_ids) if fixture_ids else 0
            }
            
        except Exception as e:
            raise FeatureComputationError(f"Feature computation failed: {str(e)}")
    
    async def _handle_model_training(self, task: Task) -> Dict[str, Any]:
        """Handle model training task.
        
        Args:
            task: Model training task
            
        Returns:
            Task result
        """
        try:
            # Get parameters
            start_date = task.parameters.get('start_date')
            end_date = task.parameters.get('end_date')
            model_save_path = task.parameters.get('model_save_path')
            
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            
            # Train model
            results = train_model_pipeline(
                start_date=start_date,
                end_date=end_date,
                model_save_path=model_save_path
            )
            
            return {
                'training_samples': results['model_metadata']['training_samples'],
                'validation_samples': results['model_metadata']['validation_samples'],
                'validation_rmse': results['validation_metrics']['rmse'],
                'validation_r2': results['validation_metrics']['r2'],
                'cv_rmse_mean': results['cross_validation']['cv_rmse_mean'],
                'model_saved': results.get('model_saved')
            }
            
        except Exception as e:
            raise TrainingError(f"Model training failed: {str(e)}")
    
    async def _handle_data_quality_check(self, task: Task) -> Dict[str, Any]:
        """Handle data quality check task.
        
        Args:
            task: Data quality check task
            
        Returns:
            Task result
        """
        try:
            if not self.db_connection:
                self.connect_to_database()
            
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Check feature completeness
                cursor.execute("""
                    SELECT 
                        AVG(quality_score) as avg_quality_score,
                        AVG(feature_completeness) as avg_completeness,
                        COUNT(*) as total_features,
                        COUNT(*) FILTER (WHERE quality_score >= 0.8) as high_quality_count
                    FROM pre_computed_features
                    WHERE computation_status = 'completed'
                    AND computed_at >= NOW() - INTERVAL '7 days'
                """)
                
                quality_stats = cursor.fetchone()
                
                # Check for missing features
                cursor.execute("""
                    SELECT COUNT(*) as missing_features
                    FROM fixtures f
                    LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
                    WHERE f.utc_date >= NOW() - INTERVAL '30 days'
                    AND f.status = 'finished'
                    AND pcf.fixture_id IS NULL
                """)
                
                missing_stats = cursor.fetchone()
                
                return {
                    'avg_quality_score': float(quality_stats['avg_quality_score'] or 0),
                    'avg_completeness': float(quality_stats['avg_completeness'] or 0),
                    'total_features': quality_stats['total_features'],
                    'high_quality_count': quality_stats['high_quality_count'],
                    'missing_features': missing_stats['missing_features']
                }
                
        except Exception as e:
            raise WorkflowError(f"Data quality check failed: {str(e)}")
    
    async def _handle_cache_maintenance(self, task: Task) -> Dict[str, Any]:
        """Handle cache maintenance task.
        
        Args:
            task: Cache maintenance task
            
        Returns:
            Task result
        """
        try:
            if not self.db_connection:
                self.connect_to_database()
            
            with self.db_connection.cursor() as cursor:
                # Clean old cache entries
                cursor.execute("""
                    DELETE FROM h2h_cache_enhanced 
                    WHERE last_updated < NOW() - INTERVAL '30 days'
                """)
                old_h2h_deleted = cursor.rowcount
                
                # Clean old computation logs
                cursor.execute("""
                    DELETE FROM feature_computation_log 
                    WHERE computed_at < NOW() - INTERVAL '90 days'
                """)
                old_logs_deleted = cursor.rowcount
                
                # Clean completed tasks
                cursor.execute("""
                    DELETE FROM feature_computation_queue 
                    WHERE status = 'completed' 
                    AND completed_at < NOW() - INTERVAL '7 days'
                """)
                old_tasks_deleted = cursor.rowcount
                
                self.db_connection.commit()
                
                return {
                    'h2h_cache_cleaned': old_h2h_deleted,
                    'logs_cleaned': old_logs_deleted,
                    'tasks_cleaned': old_tasks_deleted
                }
                
        except Exception as e:
            raise WorkflowError(f"Cache maintenance failed: {str(e)}")
    
    async def _handle_historical_backfill(self, task: Task) -> Dict[str, Any]:
        """Handle historical data backfill task.
        
        Args:
            task: Historical backfill task
            
        Returns:
            Task result
        """
        try:
            start_date = task.parameters.get('start_date')
            end_date = task.parameters.get('end_date')
            
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            
            # Get fixtures in date range that need backfill
            if not self.db_connection:
                self.connect_to_database()
            
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT f.id
                    FROM fixtures f
                    LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
                    WHERE f.utc_date BETWEEN %s AND %s
                    AND f.status = 'finished'
                    AND pcf.fixture_id IS NULL
                    ORDER BY f.utc_date DESC
                """, (start_date, end_date))
                
                fixture_ids = [row[0] for row in cursor.fetchall()]
            
            if fixture_ids:
                # Create feature computation task
                computation_task_id = self.create_task(
                    name=f"Historical backfill computation ({len(fixture_ids)} fixtures)",
                    task_type="feature_computation",
                    priority=TaskPriority.LOW,
                    parameters={
                        'fixture_ids': fixture_ids,
                        'batch_size': 25,
                        'force_recompute': False
                    }
                )
                
                return {
                    'fixtures_found': len(fixture_ids),
                    'computation_task_created': computation_task_id
                }
            else:
                return {
                    'fixtures_found': 0,
                    'message': 'No fixtures need backfill'
                }
                
        except Exception as e:
            raise WorkflowError(f"Historical backfill failed: {str(e)}")
    
    async def _handle_health_check(self, task: Task) -> Dict[str, Any]:
        """Handle system health check task.
        
        Args:
            task: Health check task
            
        Returns:
            Task result
        """
        try:
            health_status = {
                'database': 'unknown',
                'feature_computation': 'unknown',
                'api_quota': 'unknown',
                'disk_space': 'unknown'
            }
            
            # Check database connection
            try:
                if not self.db_connection:
                    self.connect_to_database()
                
                with self.db_connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    health_status['database'] = 'healthy'
            except Exception:
                health_status['database'] = 'unhealthy'
            
            # Check recent feature computation
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM feature_computation_log 
                        WHERE computed_at >= NOW() - INTERVAL '24 hours'
                        AND status = 'completed'
                    """)
                    recent_computations = cursor.fetchone()[0]
                    
                    if recent_computations > 0:
                        health_status['feature_computation'] = 'healthy'
                    else:
                        health_status['feature_computation'] = 'warning'
            except Exception:
                health_status['feature_computation'] = 'unhealthy'
            
            # Check API quota usage
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT SUM(requests_made) as daily_requests
                        FROM api_usage_log 
                        WHERE date_trunc('day', created_at) = date_trunc('day', NOW())
                    """)
                    daily_requests = cursor.fetchone()[0] or 0
                    
                    quota_limit = self.config.feature_computation.api_usage.daily_quota_limit
                    if daily_requests < quota_limit * 0.8:
                        health_status['api_quota'] = 'healthy'
                    elif daily_requests < quota_limit:
                        health_status['api_quota'] = 'warning'
                    else:
                        health_status['api_quota'] = 'critical'
            except Exception:
                health_status['api_quota'] = 'unknown'
            
            # Check disk space (simplified)
            try:
                import shutil
                total, used, free = shutil.disk_usage("/")
                free_percentage = free / total * 100
                
                if free_percentage > 20:
                    health_status['disk_space'] = 'healthy'
                elif free_percentage > 10:
                    health_status['disk_space'] = 'warning'
                else:
                    health_status['disk_space'] = 'critical'
            except Exception:
                health_status['disk_space'] = 'unknown'
            
            # Overall health
            unhealthy_count = sum(1 for status in health_status.values() if status == 'unhealthy')
            critical_count = sum(1 for status in health_status.values() if status == 'critical')
            
            if critical_count > 0 or unhealthy_count > 1:
                overall_health = 'unhealthy'
            elif unhealthy_count > 0 or 'warning' in health_status.values():
                overall_health = 'warning'
            else:
                overall_health = 'healthy'
            
            return {
                'overall_health': overall_health,
                'components': health_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise WorkflowError(f"Health check failed: {str(e)}")
    
    async def _handle_cleanup(self, task: Task) -> Dict[str, Any]:
        """Handle cleanup task.
        
        Args:
            task: Cleanup task
            
        Returns:
            Task result
        """
        try:
            # Clean up completed and failed tasks
            completed_tasks = [t for t in self.tasks.values() 
                             if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                             and t.completed_at 
                             and t.completed_at < datetime.now() - timedelta(hours=24)]
            
            for task_to_clean in completed_tasks:
                del self.tasks[task_to_clean.id]
            
            return {
                'tasks_cleaned': len(completed_tasks)
            }
            
        except Exception as e:
            raise WorkflowError(f"Cleanup failed: {str(e)}")
    
    async def run_workflow(self, max_concurrent_tasks: int = 3) -> None:
        """Run the workflow orchestrator.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self.is_running = True
        self.logger.info("Starting workflow orchestrator")
        
        try:
            # Load existing tasks
            self.load_tasks_from_database()
            
            while self.is_running and not self.shutdown_requested:
                # Get ready tasks
                ready_tasks = self.get_ready_tasks()
                
                # Start new tasks if we have capacity
                while (len(self.running_tasks) < max_concurrent_tasks and 
                       ready_tasks and not self.shutdown_requested):
                    
                    task = ready_tasks.pop(0)
                    
                    # Start task
                    task_coroutine = self.execute_task(task)
                    asyncio_task = asyncio.create_task(task_coroutine)
                    self.running_tasks[task.id] = asyncio_task
                
                # Wait for some tasks to complete
                if self.running_tasks:
                    done, pending = await asyncio.wait(
                        self.running_tasks.values(),
                        timeout=10.0,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Clean up completed tasks
                    for task_future in done:
                        for task_id, future in list(self.running_tasks.items()):
                            if future == task_future:
                                del self.running_tasks[task_id]
                                break
                else:
                    # No running tasks, wait a bit
                    await asyncio.sleep(30)
                
                # Periodic cleanup
                if len(self.tasks) > 1000:
                    await self._handle_cleanup(Task(
                        id="cleanup", name="Periodic cleanup", task_type="cleanup",
                        priority=TaskPriority.LOW, status=TaskStatus.RUNNING,
                        created_at=datetime.now()
                    ))
        
        except Exception as e:
            self.logger.error(f"Workflow orchestrator error: {str(e)}")
            raise WorkflowError(f"Workflow execution failed: {str(e)}")
        
        finally:
            # Wait for running tasks to complete
            if self.running_tasks:
                await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
            
            self.is_running = False
            self.logger.info("Workflow orchestrator stopped")
    
    def shutdown(self) -> None:
        """Request graceful shutdown."""
        self.logger.info("Shutdown requested")
        self.shutdown_requested = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status.
        
        Returns:
            Status information
        """
        task_counts = {}
        for status in TaskStatus:
            task_counts[status.value] = sum(
                1 for task in self.tasks.values() if task.status == status
            )
        
        return {
            'is_running': self.is_running,
            'shutdown_requested': self.shutdown_requested,
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.running_tasks),
            'task_counts': task_counts,
            'database_connected': self.db_connection is not None
        }
    
    def schedule_daily_tasks(self) -> None:
        """Schedule daily recurring tasks."""
        # Feature computation for recent fixtures
        self.create_task(
            name="Daily feature computation",
            task_type="feature_computation",
            priority=TaskPriority.HIGH,
            parameters={
                'batch_size': 100,
                'force_recompute': False
            }
        )
        
        # Data quality check
        self.create_task(
            name="Daily data quality check",
            task_type="data_quality_check",
            priority=TaskPriority.NORMAL
        )
        
        # Health check
        self.create_task(
            name="Daily health check",
            task_type="health_check",
            priority=TaskPriority.NORMAL
        )
        
        # Cache maintenance (weekly)
        if datetime.now().weekday() == 0:  # Monday
            self.create_task(
                name="Weekly cache maintenance",
                task_type="cache_maintenance",
                priority=TaskPriority.LOW
            )
    
    def schedule_training_task(self, 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              model_save_path: Optional[str] = None) -> str:
        """Schedule a model training task.
        
        Args:
            start_date: Training data start date
            end_date: Training data end date
            model_save_path: Path to save the model
            
        Returns:
            Task ID
        """
        # Ensure features are computed first
        feature_task_id = self.create_task(
            name="Pre-training feature computation",
            task_type="feature_computation",
            priority=TaskPriority.HIGH,
            parameters={
                'batch_size': 50,
                'force_recompute': False
            }
        )
        
        # Schedule training task with dependency
        training_task_id = self.create_task(
            name="Model training",
            task_type="model_training",
            priority=TaskPriority.HIGH,
            parameters={
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'model_save_path': model_save_path
            },
            dependencies=[feature_task_id]
        )
        
        return training_task_id
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_to_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_database_connection()


async def main():
    """Main orchestrator entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FormFinder2 Workflow Orchestrator")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent tasks")
    parser.add_argument("--schedule-daily", action="store_true", help="Schedule daily tasks")
    
    args = parser.parse_args()
    
    with WorkflowOrchestrator(args.config) as orchestrator:
        if args.schedule_daily:
            orchestrator.schedule_daily_tasks()
        
        await orchestrator.run_workflow(max_concurrent_tasks=args.max_concurrent)


if __name__ == "__main__":
    asyncio.run(main())