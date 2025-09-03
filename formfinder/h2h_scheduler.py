#!/usr/bin/env python3
"""
H2H Data Collection Scheduler

Automated scheduler for H2H data collection with intelligent prioritization and rate limiting.
Integrates with the H2H collection service to optimize API usage and ensure data freshness.
Includes monitoring integration and comprehensive scheduling strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import json
from pathlib import Path

from .database import get_db_session
from .clients.api_client import SoccerDataAPIClient
from .h2h_collection_service import H2HCollectionService, Priority
from .h2h_monitoring import H2HMonitor
from .config import get_config
from .logger import get_logger
from .monitoring import MetricsCollector


@dataclass
class LeagueConfig:
    """Configuration for league-specific H2H collection."""
    league_id: int
    name: str
    priority: Priority
    collection_interval_hours: int = 24
    lookahead_days: int = 14
    enabled: bool = True


class H2HScheduler:
    """Automated scheduler for H2H data collection with monitoring integration."""
    
    def __init__(self, collection_service: H2HCollectionService, config=None, monitor: H2HMonitor = None):
        self.config = config or get_config()
        self.collection_service = collection_service
        self.monitor = monitor
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Scheduler configuration
        self.schedule_interval_minutes = config.get('h2h_scheduler.interval_minutes', 60)
        self.max_tasks_per_run = config.get('h2h_scheduler.max_tasks_per_run', 100)
        self.adaptive_scheduling = config.get('h2h_scheduler.adaptive_scheduling', True)
        self.emergency_throttle_threshold = config.get('h2h_scheduler.emergency_throttle_threshold', 0.2)
        
        # League configurations
        self.league_configs = self._load_league_configs()
        
        # Scheduler state
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.last_run: Optional[datetime] = None
        self.emergency_throttle = False
        self.last_health_check = datetime.min
        
    def _load_league_configs(self) -> Dict[int, LeagueConfig]:
        """Load league configurations from config."""
        configs = {}
        
        # Default league configurations
        default_leagues = [
            # Premier League
            LeagueConfig(203, "Premier League", Priority.HIGH, 12, 21),
            # La Liga
            LeagueConfig(204, "La Liga", Priority.HIGH, 12, 21),
            # Bundesliga
            LeagueConfig(205, "Bundesliga", Priority.HIGH, 12, 21),
            # Serie A
            LeagueConfig(206, "Serie A", Priority.HIGH, 12, 21),
            # Ligue 1
            LeagueConfig(207, "Ligue 1", Priority.HIGH, 12, 21),
            # Champions League
            LeagueConfig(235, "Champions League", Priority.CRITICAL, 6, 28),
            # Europa League
            LeagueConfig(236, "Europa League", Priority.HIGH, 12, 21),
            # Championship
            LeagueConfig(208, "Championship", Priority.MEDIUM, 24, 14),
            # MLS
            LeagueConfig(242, "MLS", Priority.MEDIUM, 24, 14),
            # Eredivisie
            LeagueConfig(209, "Eredivisie", Priority.MEDIUM, 24, 14),
        ]
        
        # Load from config if available
        config_leagues = self.config.get('h2h_scheduler.leagues', [])
        if config_leagues:
            for league_data in config_leagues:
                config = LeagueConfig(
                    league_id=league_data['id'],
                    name=league_data.get('name', f"League {league_data['id']}"),
                    priority=Priority[league_data.get('priority', 'MEDIUM')],
                    collection_interval_hours=league_data.get('interval_hours', 24),
                    lookahead_days=league_data.get('lookahead_days', 14),
                    enabled=league_data.get('enabled', True)
                )
                configs[config.league_id] = config
        else:
            # Use defaults
            for config in default_leagues:
                configs[config.league_id] = config
        
        enabled_count = sum(1 for c in configs.values() if c.enabled)
        self.logger.info(f"Loaded {len(configs)} league configurations ({enabled_count} enabled)")
        
        return configs
    
    async def initialize(self):
        """Initialize the scheduler."""
        self.logger.info("Initializing H2H Scheduler")
        
        # Initialize database session
        self.db_session = await get_db_session()
        
        # Initialize API client
        self.api_client = SoccerDataAPIClient(self.config)
        
        self.logger.info("H2H Scheduler initialized successfully")
    
    async def start(self):
        """Start the scheduler with health monitoring."""
        if self.is_running:
            self.logger.warning("H2H Scheduler is already running")
            return
        
        self.is_running = True
        self.logger.info(f"Starting H2H Scheduler with monitoring integration (interval: {self.schedule_interval_minutes} minutes)")
        
        try:
            # Start both scheduler and health monitoring loops
            scheduler_task = asyncio.create_task(self._scheduler_loop())
            health_task = asyncio.create_task(self._health_monitor_loop())
            
            self.scheduler_task = scheduler_task
            
            # Wait for either task to complete (or fail)
            done, pending = await asyncio.wait(
                [scheduler_task, health_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except asyncio.CancelledError:
            self.logger.info("H2H Scheduler cancelled")
        except Exception as e:
            self.logger.error(f"H2H Scheduler error: {e}")
        finally:
            self.is_running = False
        
        self.logger.info("H2H Scheduler started successfully")
    
    async def stop(self):
        """Stop the H2H scheduler."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping H2H Scheduler")
        self.is_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("H2H Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop with adaptive scheduling."""
        self.logger.info("H2H Scheduler loop started")
        
        while self.is_running:
            try:
                start_time = datetime.now()
                
                # Check if emergency throttle is active
                if self.emergency_throttle:
                    self.logger.warning("Emergency throttle active, skipping collection cycle")
                    await asyncio.sleep(self.schedule_interval_minutes * 60 * 2)  # Double interval
                    continue
                
                # Run scheduling logic
                tasks_scheduled = await self._schedule_h2h_tasks()
                
                # Update metrics
                self.metrics.increment('h2h_scheduler.runs')
                self.metrics.gauge('h2h_scheduler.tasks_scheduled', tasks_scheduled)
                
                # Update last run time
                self.last_run = start_time
                
                # Log completion
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.info(
                    f"H2H Scheduler run completed: {tasks_scheduled} tasks scheduled in {duration:.1f}s"
                )
                
                # Adaptive sleep interval based on system health
                sleep_interval = await self._calculate_adaptive_interval()
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                self.logger.error(f"H2H Scheduler error: {e}")
                self.metrics.increment('h2h_scheduler.errors')
                await asyncio.sleep(300)  # 5 minute error backoff
        
        self.logger.info("H2H Scheduler loop stopped")
    
    async def _schedule_h2h_tasks(self) -> int:
        """Schedule H2H collection tasks based on upcoming fixtures."""
        total_tasks_scheduled = 0
        
        for league_config in self.league_configs.values():
            if not league_config.enabled:
                continue
            
            try:
                # Check if it's time to schedule for this league
                if not await self._should_schedule_league(league_config):
                    continue
                
                # Get upcoming fixtures for this league
                fixtures = await self._get_upcoming_fixtures(league_config)
                
                if not fixtures:
                    self.logger.debug(f"No upcoming fixtures found for {league_config.name}")
                    continue
                
                # Generate H2H tasks from fixtures
                tasks = await self._generate_h2h_tasks(fixtures, league_config)
                
                # Add tasks to collection service
                if tasks:
                    scheduled_count = await self.collection_service.add_bulk_tasks(tasks)
                    total_tasks_scheduled += scheduled_count
                    
                    self.logger.info(
                        f"Scheduled {scheduled_count} H2H tasks for {league_config.name} "
                        f"from {len(fixtures)} fixtures"
                    )
                
            except Exception as e:
                self.logger.error(f"Error scheduling H2H tasks for {league_config.name}: {e}")
                continue
        
        return total_tasks_scheduled
    
    async def _should_schedule_league(self, league_config: LeagueConfig) -> bool:
        """Check if it's time to schedule H2H tasks for a league."""
        if not self.last_run:
            return True  # First run
        
        time_since_last_run = datetime.now() - self.last_run
        return time_since_last_run >= timedelta(hours=league_config.collection_interval_hours)
    
    async def _get_upcoming_fixtures(self, league_config: LeagueConfig) -> List[Dict]:
        """Get upcoming fixtures for a league."""
        try:
            # Calculate date range
            start_date = datetime.now().date()
            end_date = start_date + timedelta(days=league_config.lookahead_days)
            
            # Query database for upcoming fixtures
            query = """
                SELECT DISTINCT 
                    home_team_id,
                    away_team_id,
                    league_id,
                    fixture_date,
                    fixture_id
                FROM fixtures 
                WHERE league_id = %s 
                    AND fixture_date BETWEEN %s AND %s
                    AND status IN ('Not Started', 'TBD')
                ORDER BY fixture_date ASC
                LIMIT 50
            """
            
            result = await self.db_session.execute(
                query, (league_config.league_id, start_date, end_date)
            )
            
            fixtures = []
            for row in result.fetchall():
                fixtures.append({
                    'home_team_id': row[0],
                    'away_team_id': row[1],
                    'league_id': row[2],
                    'fixture_date': row[3],
                    'fixture_id': row[4]
                })
            
            return fixtures
            
        except Exception as e:
            self.logger.error(f"Error fetching fixtures for {league_config.name}: {e}")
            return []
    
    async def _generate_h2h_tasks(self, fixtures: List[Dict], 
                                 league_config: LeagueConfig) -> List[tuple]:
        """Generate H2H collection tasks from fixtures."""
        tasks = []
        
        for fixture in fixtures:
            # Determine priority based on fixture timing
            fixture_date = fixture['fixture_date']
            if isinstance(fixture_date, str):
                fixture_date = datetime.fromisoformat(fixture_date).date()
            
            days_until_fixture = (fixture_date - datetime.now().date()).days
            
            # Adjust priority based on timing
            if days_until_fixture <= 1:
                priority = Priority.CRITICAL
            elif days_until_fixture <= 3:
                priority = Priority.HIGH
            elif days_until_fixture <= 7:
                priority = Priority.MEDIUM
            else:
                priority = league_config.priority
            
            # Create task tuple
            task = (
                fixture['home_team_id'],
                fixture['away_team_id'],
                fixture['league_id'],
                priority
            )
            
            tasks.append(task)
        
        return tasks
    
    async def schedule_immediate_h2h(self, team1_id: int, team2_id: int, 
                                   league_id: int, priority: Priority = Priority.HIGH) -> bool:
        """Schedule an immediate H2H collection task."""
        try:
            success = await self.collection_service.add_task(
                team1_id, team2_id, league_id, priority
            )
            
            if success:
                self.logger.info(
                    f"Scheduled immediate H2H task: {team1_id} vs {team2_id} "
                    f"(league: {league_id}, priority: {priority.name})"
                )
                self.metrics.increment('h2h_scheduler.immediate_tasks')
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error scheduling immediate H2H task: {e}")
            return False
    
    async def schedule_league_bulk(self, league_id: int, 
                                  priority: Priority = Priority.MEDIUM) -> int:
        """Schedule H2H collection for all upcoming fixtures in a league."""
        try:
            # Find league config or create temporary one
            league_config = self.league_configs.get(
                league_id,
                LeagueConfig(league_id, f"League {league_id}", priority)
            )
            
            # Get fixtures and generate tasks
            fixtures = await self._get_upcoming_fixtures(league_config)
            tasks = await self._generate_h2h_tasks(fixtures, league_config)
            
            # Schedule tasks
            if tasks:
                scheduled_count = await self.collection_service.add_bulk_tasks(tasks)
                self.logger.info(
                    f"Bulk scheduled {scheduled_count} H2H tasks for league {league_id}"
                )
                self.metrics.increment('h2h_scheduler.bulk_tasks', scheduled_count)
                return scheduled_count
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error in bulk H2H scheduling for league {league_id}: {e}")
            return 0
    
    async def get_status(self) -> Dict:
        """Get comprehensive scheduler status."""
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'emergency_throttle': self.emergency_throttle,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check != datetime.min else None,
            'schedule_interval_minutes': self.schedule_interval_minutes,
            'enabled_leagues': [
                {
                    'id': config.league_id,
                    'name': config.name,
                    'priority': config.priority.name,
                    'interval_hours': config.collection_interval_hours,
                    'lookahead_days': config.lookahead_days
                }
                for config in self.league_configs.values()
                if config.enabled
            ],
            'total_leagues': len(self.league_configs),
            'enabled_leagues_count': sum(1 for c in self.league_configs.values() if c.enabled),
            'config': {
                'max_tasks_per_run': self.max_tasks_per_run,
                'adaptive_scheduling': self.adaptive_scheduling,
                'emergency_throttle_threshold': self.emergency_throttle_threshold
            }
        }
    
    async def update_league_config(self, league_id: int, **kwargs) -> bool:
        """Update configuration for a specific league."""
        try:
            if league_id not in self.league_configs:
                self.logger.error(f"League {league_id} not found in configurations")
                return False
            
            config = self.league_configs[league_id]
            
            # Update allowed fields
            if 'priority' in kwargs:
                config.priority = Priority[kwargs['priority']]
            if 'collection_interval_hours' in kwargs:
                config.collection_interval_hours = kwargs['collection_interval_hours']
            if 'lookahead_days' in kwargs:
                config.lookahead_days = kwargs['lookahead_days']
            if 'enabled' in kwargs:
                config.enabled = kwargs['enabled']
            
            self.logger.info(f"Updated configuration for league {league_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating league config for {league_id}: {e}")
            return False


    async def _health_monitor_loop(self):
        """Monitor system health and adjust scheduling accordingly."""
        if not self.monitor:
            self.logger.info("No monitor provided, skipping health monitoring")
            return
        
        health_check_interval = self.config.get('health_check_interval', 300)  # 5 minutes default
        
        while self.is_running:
            try:
                await asyncio.sleep(health_check_interval)
                
                # Get current system health
                health_status = self.monitor.get_health_status()
                self.last_health_check = datetime.now()
                
                # Check for emergency conditions
                if health_status['status'] == 'unhealthy':
                    # Check API failure rate specifically
                    metrics = health_status.get('metrics', {})
                    total_requests = metrics.get('api_requests_total', 0)
                    failed_requests = metrics.get('api_requests_failed', 0)
                    
                    if total_requests > 0:
                        failure_rate = failed_requests / total_requests
                        if failure_rate > self.emergency_throttle_threshold:
                            if not self.emergency_throttle:
                                self.logger.warning(
                                    f"Activating emergency throttle due to {failure_rate:.1%} API failure rate"
                                )
                                self.emergency_throttle = True
                        else:
                            if self.emergency_throttle:
                                self.logger.info("Deactivating emergency throttle - system health improved")
                                self.emergency_throttle = False
                    
                elif health_status['status'] == 'healthy' and self.emergency_throttle:
                    self.logger.info("Deactivating emergency throttle - system healthy")
                    self.emergency_throttle = False
                
                # Log health status periodically
                if self.last_health_check.minute % 15 == 0:  # Every 15 minutes
                    self.logger.info(f"System health: {health_status['status']}")
                    
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_adaptive_interval(self) -> float:
        """Calculate adaptive sleep interval based on system performance."""
        base_interval = self.schedule_interval_minutes * 60
        
        if not self.adaptive_scheduling or not self.monitor:
            return base_interval
        
        try:
            # Get current metrics
            metrics = self.monitor.get_current_metrics(self.collection_service)
            
            # Adjust interval based on queue size
            queue_size = metrics.get('collection_queue_size', 0)
            if queue_size > 500:  # High queue backlog
                return base_interval * 1.5  # Slow down scheduling
            elif queue_size < 50:  # Low queue
                return base_interval * 0.8  # Speed up scheduling
            
            # Adjust based on API performance
            avg_response_time = metrics.get('avg_response_time', 1.0)
            if avg_response_time > 3.0:  # Slow API responses
                return base_interval * 1.2
            elif avg_response_time < 1.0:  # Fast API responses
                return base_interval * 0.9
            
            return base_interval
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive interval: {e}")
            return base_interval


async def create_h2h_scheduler(config=None, 
                              collection_service: H2HCollectionService = None) -> H2HScheduler:
    """Factory function to create and initialize H2H scheduler."""
    scheduler = H2HScheduler(collection_service, config)
    await scheduler.initialize()
    return scheduler