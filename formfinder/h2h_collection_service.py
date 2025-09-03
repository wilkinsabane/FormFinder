#!/usr/bin/env python3
"""
Standalone H2H Collection Service

This service handles bulk H2H data collection with intelligent batching,
rate limiting, and priority-based processing to optimize API usage.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import time
from pathlib import Path
import json

try:
    # Try relative imports first (when run as module)
    from .database import get_db_session
    from .clients.api_client import SoccerDataAPIClient
    from .h2h_manager import H2HManager
    from .config import get_config
    from .logger import get_logger
    from .monitoring import MetricsCollector
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from formfinder.database import get_db_session
    from formfinder.clients.api_client import SoccerDataAPIClient
    from formfinder.h2h_manager import H2HManager
    from formfinder.config import get_config
    from formfinder.logger import get_logger
    from formfinder.monitoring import MetricsCollector


class Priority(Enum):
    """Priority levels for H2H data collection."""
    CRITICAL = 1    # Upcoming matches within 24 hours
    HIGH = 2        # Upcoming matches within 7 days
    MEDIUM = 3      # Recent matches or popular leagues
    LOW = 4         # Historical data or lower leagues


@dataclass
class H2HCollectionTask:
    """Represents a single H2H data collection task."""
    team1_id: int
    team2_id: int
    league_id: int
    priority: Priority
    scheduled_time: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.scheduled_time < other.scheduled_time
    
    @property
    def task_id(self) -> str:
        """Unique identifier for this task."""
        return f"{self.team1_id}_{self.team2_id}_{self.league_id}"


class H2HCollectionService:
    """Standalone service for efficient H2H data collection."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Rate limiting configuration with safe access
        if hasattr(self.config, 'get'):
            # Config is a dict-like object
            self.api_calls_per_minute = self.config.get('h2h_collection.api_calls_per_minute', 30)
            self.batch_size = self.config.get('h2h_collection.batch_size', 10)
            self.concurrent_workers = self.config.get('h2h_collection.concurrent_workers', 3)
            self.cache_ttl_hours = self.config.get('h2h_collection.cache_ttl_hours', 168)  # 7 days
            self.priority_cache_ttl_hours = self.config.get('h2h_collection.priority_cache_ttl_hours', 24)  # 1 day
        else:
            # Config is a FormFinderConfig object or doesn't have get method, use defaults
            self.api_calls_per_minute = 30
            self.batch_size = 10
            self.concurrent_workers = 3
            self.cache_ttl_hours = 168  # 7 days
            self.priority_cache_ttl_hours = 24  # 1 day
        
        # Task management
        self.task_queue: List[H2HCollectionTask] = []
        self.processed_tasks: Set[str] = set()
        self.failed_tasks: Dict[str, H2HCollectionTask] = {}
        
        # Rate limiting state
        self.last_api_call = 0.0
        self.api_call_count = 0
        self.rate_limit_window_start = time.time()
        
        # Service state
        self.is_running = False
        self.workers: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the collection service."""
        self.logger.info("Initializing H2H Collection Service")
        
        # Initialize database session and API client
        with get_db_session() as session:
            self.api_client = SoccerDataAPIClient(session)
            self.h2h_manager = H2HManager(session, self.api_client)
        
        self.logger.info("H2H Collection Service initialized successfully")
    
    async def add_task(self, team1_id: int, team2_id: int, league_id: int, 
                      priority: Priority = Priority.MEDIUM) -> bool:
        """Add a new H2H collection task to the queue."""
        task = H2HCollectionTask(
            team1_id=team1_id,
            team2_id=team2_id,
            league_id=league_id,
            priority=priority
        )
        
        # Check if task already exists or was recently processed
        if task.task_id in self.processed_tasks:
            self.logger.debug(f"Task {task.task_id} already processed, skipping")
            return False
        
        # Check if we need to collect this data based on cache freshness
        if await self._is_cache_fresh(task):
            self.logger.debug(f"Cache is fresh for task {task.task_id}, skipping")
            return False
        
        heapq.heappush(self.task_queue, task)
        self.logger.info(f"Added H2H collection task: {task.task_id} (priority: {priority.name})")
        return True
    
    async def add_bulk_tasks(self, tasks: List[Tuple[int, int, int, Priority]]) -> int:
        """Add multiple tasks in bulk."""
        added_count = 0
        for team1_id, team2_id, league_id, priority in tasks:
            if await self.add_task(team1_id, team2_id, league_id, priority):
                added_count += 1
        
        self.logger.info(f"Added {added_count}/{len(tasks)} bulk H2H collection tasks")
        return added_count
    
    async def start(self):
        """Start the H2H collection service."""
        if self.is_running:
            self.logger.warning("H2H Collection Service is already running")
            return
        
        self.is_running = True
        self.logger.info(f"Starting H2H Collection Service with {self.concurrent_workers} workers")
        
        # Start worker tasks
        for i in range(self.concurrent_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor())
        self.workers.append(monitor_task)
        
        self.logger.info("H2H Collection Service started successfully")
    
    async def stop(self):
        """Stop the H2H collection service."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping H2H Collection Service")
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        self.logger.info("H2H Collection Service stopped")
    
    async def _worker(self, worker_name: str):
        """Worker coroutine for processing H2H collection tasks."""
        self.logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get next task from queue
                if not self.task_queue:
                    await asyncio.sleep(1)
                    continue
                
                task = heapq.heappop(self.task_queue)
                
                # Check rate limiting
                await self._enforce_rate_limit()
                
                # Process the task
                success = await self._process_task(task, worker_name)
                
                if success:
                    self.processed_tasks.add(task.task_id)
                    self.metrics.increment('h2h_collection.tasks.success')
                else:
                    # Handle failed task
                    task.attempts += 1
                    if task.attempts < task.max_attempts:
                        # Reschedule with exponential backoff
                        task.scheduled_time = datetime.now() + timedelta(minutes=2 ** task.attempts)
                        heapq.heappush(self.task_queue, task)
                        self.logger.warning(f"Rescheduling failed task {task.task_id} (attempt {task.attempts})")
                    else:
                        self.failed_tasks[task.task_id] = task
                        self.metrics.increment('h2h_collection.tasks.failed')
                        self.logger.error(f"Task {task.task_id} failed permanently after {task.attempts} attempts")
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} encountered error: {e}")
                await asyncio.sleep(5)
        
        self.logger.info(f"Worker {worker_name} stopped")
    
    async def _process_task(self, task: H2HCollectionTask, worker_name: str) -> bool:
        """Process a single H2H collection task."""
        try:
            self.logger.debug(f"Worker {worker_name} processing task {task.task_id}")
            
            # Collect H2H data
            h2h_data = await self.h2h_manager.get_or_compute_h2h(
                task.team1_id, task.team2_id, task.league_id
            )
            
            if h2h_data:
                self.logger.info(f"Successfully collected H2H data for {task.task_id}")
                self.metrics.increment('h2h_collection.api_calls.success')
                return True
            else:
                task.last_error = "No H2H data returned"
                self.logger.warning(f"No H2H data returned for {task.task_id}")
                return False
                
        except Exception as e:
            task.last_error = str(e)
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            self.metrics.increment('h2h_collection.api_calls.error')
            return False
    
    async def _enforce_rate_limit(self):
        """Enforce API rate limiting."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.rate_limit_window_start >= 60:
            self.api_call_count = 0
            self.rate_limit_window_start = current_time
        
        # Check if we've hit the rate limit
        if self.api_call_count >= self.api_calls_per_minute:
            sleep_time = 60 - (current_time - self.rate_limit_window_start)
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
                self.api_call_count = 0
                self.rate_limit_window_start = time.time()
        
        # Ensure minimum time between API calls
        min_interval = 60 / self.api_calls_per_minute
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < min_interval:
            await asyncio.sleep(min_interval - time_since_last_call)
        
        self.api_call_count += 1
        self.last_api_call = time.time()
    
    async def _is_cache_fresh(self, task: H2HCollectionTask) -> bool:
        """Check if cached H2H data is still fresh."""
        try:
            # Use shorter TTL for high priority tasks
            ttl_hours = (self.priority_cache_ttl_hours 
                        if task.priority in [Priority.CRITICAL, Priority.HIGH] 
                        else self.cache_ttl_hours)
            
            cached_data = await self.h2h_manager._get_cached_h2h(
                task.team1_id, task.team2_id, task.league_id
            )
            
            if cached_data:
                cache_age = datetime.now() - cached_data.get('created_at', datetime.min)
                return cache_age < timedelta(hours=ttl_hours)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking cache freshness for {task.task_id}: {e}")
            return False
    
    async def _monitor(self):
        """Monitor service health and performance."""
        self.logger.info("H2H Collection Service monitor started")
        
        while self.is_running:
            try:
                # Log queue status
                queue_size = len(self.task_queue)
                processed_count = len(self.processed_tasks)
                failed_count = len(self.failed_tasks)
                
                self.logger.info(
                    f"H2H Collection Status - Queue: {queue_size}, "
                    f"Processed: {processed_count}, Failed: {failed_count}"
                )
                
                # Update metrics
                self.metrics.gauge('h2h_collection.queue_size', queue_size)
                self.metrics.gauge('h2h_collection.processed_count', processed_count)
                self.metrics.gauge('h2h_collection.failed_count', failed_count)
                
                # Sleep for monitoring interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)
        
        self.logger.info("H2H Collection Service monitor stopped")
    
    async def get_status(self) -> Dict:
        """Get current service status."""
        return {
            'is_running': self.is_running,
            'queue_size': len(self.task_queue),
            'processed_count': len(self.processed_tasks),
            'failed_count': len(self.failed_tasks),
            'workers_count': len(self.workers),
            'api_calls_per_minute': self.api_calls_per_minute,
            'current_api_call_count': self.api_call_count,
            'rate_limit_window_remaining': max(0, 60 - (time.time() - self.rate_limit_window_start))
        }
    
    async def get_failed_tasks(self) -> List[Dict]:
        """Get list of failed tasks for debugging."""
        return [
            {
                'task_id': task.task_id,
                'team1_id': task.team1_id,
                'team2_id': task.team2_id,
                'league_id': task.league_id,
                'priority': task.priority.name,
                'attempts': task.attempts,
                'last_error': task.last_error,
                'scheduled_time': task.scheduled_time.isoformat()
            }
            for task in self.failed_tasks.values()
        ]
    
    async def retry_failed_tasks(self) -> int:
        """Retry all failed tasks."""
        retry_count = 0
        for task_id, task in list(self.failed_tasks.items()):
            task.attempts = 0
            task.last_error = None
            task.scheduled_time = datetime.now()
            heapq.heappush(self.task_queue, task)
            del self.failed_tasks[task_id]
            retry_count += 1
        
        self.logger.info(f"Retrying {retry_count} failed H2H collection tasks")
        return retry_count


async def create_h2h_collection_service(config=None) -> H2HCollectionService:
    """Factory function to create and initialize H2H collection service."""
    # Get config if not provided
    if config is None:
        try:
            config = get_config()
        except RuntimeError:
            # Config not loaded, use None and let service handle defaults
            config = None
    
    service = H2HCollectionService(config)
    await service.initialize()
    return service


def load_team_pairs_from_file(file_path: str) -> List[Tuple[int, int, int, Priority]]:
    """Load team pairs from a configuration file.
    
    Expected file format (JSON):
    {
        "leagues": {
            "203": {
                "name": "Premier League",
                "team_pairs": [
                    {"team1_id": 2689, "team2_id": 2693, "priority": "HIGH"},
                    {"team1_id": 3258, "team2_id": 3259, "priority": "MEDIUM"}
                ]
            }
        }
    }
    """
    import json
    import os
    
    if not os.path.exists(file_path):
        print(f"⚠️  Configuration file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        team_pairs = []
        leagues = config.get('leagues', {})
        
        for league_id, league_data in leagues.items():
            league_name = league_data.get('name', f'League {league_id}')
            pairs = league_data.get('team_pairs', [])
            
            print(f"📋 Loading {len(pairs)} team pairs from {league_name} (ID: {league_id})")
            
            for pair in pairs:
                try:
                    team1_id = int(pair['team1_id'])
                    team2_id = int(pair['team2_id'])
                    priority_str = pair.get('priority', 'MEDIUM').upper()
                    priority = Priority[priority_str] if priority_str in Priority.__members__ else Priority.MEDIUM
                    
                    team_pairs.append((team1_id, team2_id, int(league_id), priority))
                    
                except (KeyError, ValueError) as e:
                    print(f"⚠️  Skipping invalid team pair in {league_name}: {pair} - {e}")
        
        print(f"✅ Loaded {len(team_pairs)} total team pairs from configuration")
        return team_pairs
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing configuration file: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading configuration file: {e}")
        return []


def get_manual_team_input() -> List[Tuple[int, int, int, Priority]]:
    """Get team pairs from manual user input."""
    team_pairs = []
    
    print("\n📝 Manual Team ID Input Mode")
    print("Enter team pairs (press Enter with empty input to finish)")
    print("Format: team1_id,team2_id,league_id,priority")
    print("Priority options: CRITICAL, HIGH, MEDIUM, LOW")
    print("Example: 2689,2693,203,HIGH")
    
    while True:
        try:
            user_input = input("\nEnter team pair (or press Enter to finish): ").strip()
            
            if not user_input:
                break
            
            parts = [p.strip() for p in user_input.split(',')]
            
            if len(parts) < 3:
                print("❌ Invalid format. Need at least: team1_id,team2_id,league_id")
                continue
            
            team1_id = int(parts[0])
            team2_id = int(parts[1])
            league_id = int(parts[2])
            priority_str = parts[3].upper() if len(parts) > 3 else 'MEDIUM'
            
            if priority_str not in Priority.__members__:
                print(f"⚠️  Invalid priority '{priority_str}', using MEDIUM")
                priority = Priority.MEDIUM
            else:
                priority = Priority[priority_str]
            
            team_pairs.append((team1_id, team2_id, league_id, priority))
            print(f"✅ Added: Team {team1_id} vs {team2_id} (League: {league_id}, Priority: {priority.name})")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}. Please use numeric IDs.")
        except KeyboardInterrupt:
            print("\n🛑 Input cancelled")
            break
    
    return team_pairs


if __name__ == "__main__":
    """Allow running the service standalone with flexible team ID input."""
    import sys
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description='H2H Collection Service')
        parser.add_argument('--config-file', '-f', type=str, 
                          help='Path to team configuration JSON file')
        parser.add_argument('--manual', '-m', action='store_true',
                          help='Enable manual team ID input mode')
        parser.add_argument('--duration', '-d', type=int, default=60,
                          help='Service run duration in seconds (default: 60)')
        
        args = parser.parse_args()
        
        try:
            # Import and load configuration
            from formfinder.config import load_config
            load_config()
            
            # Create and start the service
            service = await create_h2h_collection_service()
            
            print("🚀 H2H Collection Service started successfully!")
            
            # Determine team pairs source
            team_pairs = []
            
            if args.config_file:
                print(f"📁 Loading team pairs from file: {args.config_file}")
                team_pairs.extend(load_team_pairs_from_file(args.config_file))
            
            if args.manual or (not args.config_file and not team_pairs):
                if not args.config_file:
                    print("ℹ️  No configuration file specified, switching to manual input")
                team_pairs.extend(get_manual_team_input())
            
            # Fallback to default test pairs if no input provided
            if not team_pairs:
                print("⚠️  No team pairs provided, using default test pairs")
                team_pairs = [
                    (2689, 2693, 203, Priority.HIGH),    # Real teams from fixtures
                    (3258, 3259, 203, Priority.MEDIUM),  # Velez vs Zeljeznicar
                    (2920, 2928, 203, Priority.LOW)      # Dinamo Zagreb vs HNK Gorica
                ]
            
            # Add tasks to the service
            print(f"\n📋 Adding {len(team_pairs)} H2H collection tasks...")
            for team1_id, team2_id, league_id, priority in team_pairs:
                success = await service.add_task(team1_id, team2_id, league_id, priority)
                if success:
                    print(f"✅ Queued: Team {team1_id} vs {team2_id} (League: {league_id}, Priority: {priority.name})")
                else:
                    print(f"⚠️  Skipped: Team {team1_id} vs {team2_id} (already processed or cache fresh)")
            
            print("\n🔄 Starting H2H collection service...")
            # Start the service
            await service.start()
            
            print(f"\n⏱️  Service running for {args.duration} seconds. Press Ctrl+C to stop early.")
            try:
                await asyncio.sleep(args.duration)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
            finally:
                await service.stop()
                print("✅ Service stopped.")
                
        except Exception as e:
            print(f"❌ Error running H2H Collection Service: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the main function
    asyncio.run(main())