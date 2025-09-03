#!/usr/bin/env python3
"""
Workflow Orchestrator

Orchestrates data collection, feature computation, and model training workflows
for the FormFinder system. Provides task management, dependency resolution,
and automated workflow execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json
import traceback

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text

from formfinder.config import get_config
from formfinder.database import get_db_session
from precompute_features import main as precompute_main
from train_model import main as train_main
from data_quality_checker import DataQualityChecker


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a workflow task."""
    id: str
    name: str
    function: Callable
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_minutes: int = 30
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    result: Any = None


class WorkflowOrchestrator:
    """Orchestrates workflow execution with dependency management."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.tasks: Dict[str, Task] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        self.tasks[task.id] = task
        self.logger.info(f"Added task: {task.name} (ID: {task.id})")
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task from the workflow."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.logger.info(f"Removed task: {task_id}")
    
    async def execute_workflow(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute workflow tasks in dependency order."""
        start_time = datetime.now()
        
        try:
            # Determine which tasks to execute
            if task_ids is None:
                tasks_to_execute = list(self.tasks.keys())
            else:
                tasks_to_execute = task_ids
            
            # Validate dependencies
            self._validate_dependencies(tasks_to_execute)
            
            # Sort tasks by dependencies and priority
            execution_order = self._resolve_dependencies(tasks_to_execute)
            
            self.logger.info(f"Executing workflow with {len(execution_order)} tasks")
            
            # Execute tasks in order
            results = {}
            for task_id in execution_order:
                task = self.tasks[task_id]
                result = await self._execute_task(task)
                results[task_id] = result
                
                # Stop execution if critical task fails
                if task.status == TaskStatus.FAILED and task.priority == TaskPriority.CRITICAL:
                    self.logger.error(f"Critical task {task_id} failed, stopping workflow")
                    break
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Generate execution summary
            summary = self._generate_execution_summary(results, duration)
            
            # Log execution
            self._log_execution(summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'failed',
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task with retry logic."""
        task.start_time = datetime.now()
        task.status = TaskStatus.RUNNING
        
        self.logger.info(f"Starting task: {task.name}")
        
        for attempt in range(task.max_retries + 1):
            try:
                # Set timeout
                timeout_seconds = task.timeout_minutes * 60
                
                # Execute task function
                if asyncio.iscoroutinefunction(task.function):
                    task.result = await asyncio.wait_for(
                        task.function(), timeout=timeout_seconds
                    )
                else:
                    task.result = await asyncio.wait_for(
                        asyncio.to_thread(task.function), timeout=timeout_seconds
                    )
                
                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()
                
                duration = (task.end_time - task.start_time).total_seconds()
                self.logger.info(f"Task {task.name} completed in {duration:.2f}s")
                
                return {
                    'status': 'completed',
                    'duration': duration,
                    'result': task.result
                }
                
            except asyncio.TimeoutError:
                task.retry_count += 1
                error_msg = f"Task {task.name} timed out after {task.timeout_minutes} minutes"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1}/{task.max_retries + 1})")
                
                if attempt < task.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = error_msg
                    task.end_time = datetime.now()
                    
            except Exception as e:
                task.retry_count += 1
                error_msg = f"Task {task.name} failed: {str(e)}"
                self.logger.error(f"{error_msg} (attempt {attempt + 1}/{task.max_retries + 1})")
                
                if attempt < task.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = error_msg
                    task.end_time = datetime.now()
        
        duration = (task.end_time - task.start_time).total_seconds()
        return {
            'status': 'failed',
            'duration': duration,
            'error': task.error_message,
            'retry_count': task.retry_count
        }
    
    def _validate_dependencies(self, task_ids: List[str]) -> None:
        """Validate that all dependencies are available."""
        for task_id in task_ids:
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"Task {task_id} depends on missing task {dep_id}")
                if dep_id not in task_ids:
                    raise ValueError(f"Task {task_id} depends on {dep_id} which is not in execution list")
    
    def _resolve_dependencies(self, task_ids: List[str]) -> List[str]:
        """Resolve task dependencies and return execution order."""
        # Topological sort with priority consideration
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def visit(task_id: str):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving task {task_id}")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            # Visit dependencies first
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id in task_ids:
                    visit(dep_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            execution_order.append(task_id)
        
        # Sort by priority first, then resolve dependencies
        sorted_tasks = sorted(task_ids, key=lambda tid: self.tasks[tid].priority.value, reverse=True)
        
        for task_id in sorted_tasks:
            if task_id not in visited:
                visit(task_id)
        
        return execution_order
    
    def _generate_execution_summary(self, results: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Generate workflow execution summary."""
        completed = sum(1 for r in results.values() if r.get('status') == 'completed')
        failed = sum(1 for r in results.values() if r.get('status') == 'failed')
        total = len(results)
        
        return {
            'status': 'completed' if failed == 0 else 'partial' if completed > 0 else 'failed',
            'total_tasks': total,
            'completed_tasks': completed,
            'failed_tasks': failed,
            'duration': duration,
            'start_time': datetime.now() - timedelta(seconds=duration),
            'end_time': datetime.now(),
            'task_results': results
        }
    
    def _log_execution(self, summary: Dict[str, Any]) -> None:
        """Log workflow execution to database."""
        try:
            with get_db_session() as session:
                log_entry = {
                    'timestamp': datetime.now(),
                    'workflow_status': summary['status'],
                    'total_tasks': summary['total_tasks'],
                    'completed_tasks': summary['completed_tasks'],
                    'failed_tasks': summary['failed_tasks'],
                    'duration': summary['duration'],
                    'details': json.dumps(summary['task_results'], default=str)
                }
                
                query = text("""
                    INSERT INTO workflow_executions 
                    (timestamp, workflow_status, total_tasks, completed_tasks, 
                     failed_tasks, duration, details)
                    VALUES (:timestamp, :workflow_status, :total_tasks, :completed_tasks,
                            :failed_tasks, :duration, :details)
                """)
                
                session.execute(query, log_entry)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log workflow execution: {e}")


def create_daily_workflow() -> WorkflowOrchestrator:
    """Create the standard daily data collection and training workflow."""
    orchestrator = WorkflowOrchestrator()
    
    # Data quality check task
    quality_task = Task(
        id="data_quality_check",
        name="Data Quality Check",
        function=lambda: run_data_quality_check(),
        priority=TaskPriority.HIGH,
        timeout_minutes=10
    )
    
    # Feature precomputation task
    precompute_task = Task(
        id="precompute_features",
        name="Precompute Features",
        function=lambda: run_feature_precomputation(),
        priority=TaskPriority.HIGH,
        dependencies=["data_quality_check"],
        timeout_minutes=60
    )
    
    # Model training task
    training_task = Task(
        id="train_model",
        name="Train Model",
        function=lambda: run_model_training(),
        priority=TaskPriority.CRITICAL,
        dependencies=["precompute_features"],
        timeout_minutes=30
    )
    
    orchestrator.add_task(quality_task)
    orchestrator.add_task(precompute_task)
    orchestrator.add_task(training_task)
    
    return orchestrator


def run_data_quality_check() -> Dict[str, Any]:
    """Run data quality check."""
    try:
        with get_db_session() as session:
            checker = DataQualityChecker(session)
            results = checker.run_comprehensive_check()
            return {'status': 'completed', 'results': results}
    except Exception as e:
        logging.error(f"Data quality check failed: {e}")
        raise


def run_feature_precomputation() -> Dict[str, Any]:
    """Run feature precomputation."""
    try:
        result = precompute_main()
        return {'status': 'completed', 'result': result}
    except Exception as e:
        logging.error(f"Feature precomputation failed: {e}")
        raise


def run_model_training() -> Dict[str, Any]:
    """Run model training."""
    try:
        result = train_main()
        return {'status': 'completed', 'result': result}
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise


async def main():
    """Main orchestrator entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and execute daily workflow
    orchestrator = create_daily_workflow()
    summary = await orchestrator.execute_workflow()
    
    print(f"Workflow completed with status: {summary['status']}")
    print(f"Duration: {summary['duration']:.2f} seconds")
    print(f"Tasks: {summary['completed_tasks']}/{summary['total_tasks']} completed")
    
    return summary


if __name__ == "__main__":
    asyncio.run(main())