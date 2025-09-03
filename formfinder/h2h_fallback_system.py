#!/usr/bin/env python3
"""
H2H Smart Fallback System

Provides intelligent fallback mechanisms for real-time H2H data fetching
when cache misses occur, with circuit breaker patterns and adaptive strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict, deque

from .database import get_db_session
from .clients.api_client import SoccerDataAPIClient
from .h2h_collection_service import Priority
from .config import get_config
from .logger import get_logger
from .monitoring import MetricsCollector


class FallbackStrategy(Enum):
    """Fallback strategies for H2H data retrieval."""
    IMMEDIATE = "immediate"        # Fetch immediately
    QUEUE_PRIORITY = "queue_priority"  # Add to priority queue
    QUEUE_NORMAL = "queue_normal"   # Add to normal queue
    SKIP = "skip"                  # Skip if not critical


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service is back


@dataclass
class CircuitBreaker:
    """Circuit breaker for API calls."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_calls: int = field(default=0)
    
    def can_execute(self) -> bool:
        """Check if circuit allows execution."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_calls < self.half_open_max_calls
    
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1


@dataclass
class FallbackRequest:
    """Represents a fallback H2H data request."""
    team1_id: int
    team2_id: int
    league_id: int
    priority: Priority
    strategy: FallbackStrategy
    requested_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    
    @property
    def request_id(self) -> str:
        """Unique identifier for this request."""
        return f"{self.team1_id}_{self.team2_id}_{self.league_id}"


class H2HFallbackSystem:
    """Smart fallback system for H2H data retrieval."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Fallback configuration
        self.immediate_timeout = config.get('h2h_fallback.immediate_timeout', 10)
        self.max_concurrent_immediate = config.get('h2h_fallback.max_concurrent_immediate', 3)
        self.rate_limit_per_minute = config.get('h2h_fallback.rate_limit_per_minute', 10)
        
        # Circuit breaker configuration
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('h2h_fallback.circuit_failure_threshold', 5),
            recovery_timeout=config.get('h2h_fallback.circuit_recovery_timeout', 60),
            half_open_max_calls=config.get('h2h_fallback.circuit_half_open_calls', 3)
        )
        
        # Request tracking
        self.pending_requests: Dict[str, FallbackRequest] = {}
        self.recent_requests: deque = deque(maxlen=100)
        self.request_times: deque = deque(maxlen=self.rate_limit_per_minute)
        
        # Concurrent execution tracking
        self.active_immediate_requests: int = 0
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_immediate)
        
        # Performance tracking
        self.success_rate_window = deque(maxlen=50)
        self.response_times = deque(maxlen=20)
        
    async def initialize(self):
        """Initialize the fallback system."""
        self.logger.info("Initializing H2H Fallback System")
        
        # Initialize database session
        self.db_session = await get_db_session()
        
        # Initialize API client
        self.api_client = SoccerDataAPIClient(self.config)
        
        self.logger.info("H2H Fallback System initialized successfully")
    
    async def request_h2h_data(self, team1_id: int, team2_id: int, league_id: int,
                              priority: Priority = Priority.MEDIUM,
                              timeout: Optional[float] = None) -> Optional[Dict]:
        """Request H2H data with intelligent fallback strategy."""
        request = FallbackRequest(
            team1_id=team1_id,
            team2_id=team2_id,
            league_id=league_id,
            priority=priority,
            strategy=self._determine_strategy(priority)
        )
        
        self.logger.debug(f"H2H fallback request: {request.request_id} (strategy: {request.strategy.value})")
        
        # Track request
        self.pending_requests[request.request_id] = request
        self.recent_requests.append(request.request_id)
        
        try:
            # Execute based on strategy
            if request.strategy == FallbackStrategy.IMMEDIATE:
                return await self._execute_immediate(request, timeout)
            elif request.strategy == FallbackStrategy.QUEUE_PRIORITY:
                return await self._execute_queue_priority(request)
            elif request.strategy == FallbackStrategy.QUEUE_NORMAL:
                return await self._execute_queue_normal(request)
            else:  # SKIP
                self.logger.debug(f"Skipping H2H request {request.request_id} due to strategy")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in H2H fallback request {request.request_id}: {e}")
            self.metrics.increment('h2h_fallback.errors')
            return None
        finally:
            # Clean up
            self.pending_requests.pop(request.request_id, None)
    
    def _determine_strategy(self, priority: Priority) -> FallbackStrategy:
        """Determine the best fallback strategy based on current conditions."""
        # Check circuit breaker state
        if self.circuit_breaker.state == CircuitState.OPEN:
            if priority in [Priority.CRITICAL, Priority.HIGH]:
                return FallbackStrategy.QUEUE_PRIORITY
            else:
                return FallbackStrategy.SKIP
        
        # Check rate limiting
        if self._is_rate_limited():
            if priority == Priority.CRITICAL:
                return FallbackStrategy.QUEUE_PRIORITY
            elif priority == Priority.HIGH:
                return FallbackStrategy.QUEUE_NORMAL
            else:
                return FallbackStrategy.SKIP
        
        # Check concurrent request limits
        if self.active_immediate_requests >= self.max_concurrent_immediate:
            if priority in [Priority.CRITICAL, Priority.HIGH]:
                return FallbackStrategy.QUEUE_PRIORITY
            else:
                return FallbackStrategy.QUEUE_NORMAL
        
        # Check recent success rate
        success_rate = self._get_recent_success_rate()
        if success_rate < 0.5:  # Less than 50% success rate
            if priority == Priority.CRITICAL:
                return FallbackStrategy.IMMEDIATE
            else:
                return FallbackStrategy.QUEUE_PRIORITY
        
        # Default strategy based on priority
        if priority == Priority.CRITICAL:
            return FallbackStrategy.IMMEDIATE
        elif priority == Priority.HIGH:
            return FallbackStrategy.IMMEDIATE
        else:
            return FallbackStrategy.QUEUE_NORMAL
    
    async def _execute_immediate(self, request: FallbackRequest, 
                               timeout: Optional[float] = None) -> Optional[Dict]:
        """Execute immediate H2H data fetch."""
        if not self.circuit_breaker.can_execute():
            self.logger.warning(f"Circuit breaker open, cannot execute immediate request {request.request_id}")
            return None
        
        timeout = timeout or self.immediate_timeout
        
        async with self.request_semaphore:
            self.active_immediate_requests += 1
            start_time = time.time()
            
            try:
                # Execute with timeout
                h2h_data = await asyncio.wait_for(
                    self._fetch_h2h_data(request),
                    timeout=timeout
                )
                
                # Record success
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                self.success_rate_window.append(True)
                self.circuit_breaker.record_success()
                
                self.metrics.increment('h2h_fallback.immediate.success')
                self.metrics.histogram('h2h_fallback.response_time', response_time)
                
                self.logger.info(f"Immediate H2H fetch successful: {request.request_id} ({response_time:.2f}s)")
                return h2h_data
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Immediate H2H fetch timeout: {request.request_id}")
                self.success_rate_window.append(False)
                self.circuit_breaker.record_failure()
                self.metrics.increment('h2h_fallback.immediate.timeout')
                return None
                
            except Exception as e:
                self.logger.error(f"Immediate H2H fetch error: {request.request_id}: {e}")
                self.success_rate_window.append(False)
                self.circuit_breaker.record_failure()
                self.metrics.increment('h2h_fallback.immediate.error')
                return None
                
            finally:
                self.active_immediate_requests -= 1
    
    async def _execute_queue_priority(self, request: FallbackRequest) -> Optional[Dict]:
        """Execute priority queue strategy."""
        # This would integrate with the H2H collection service
        # For now, we'll implement a simplified version
        self.logger.info(f"Adding H2H request to priority queue: {request.request_id}")
        
        # Add to collection service if available
        if hasattr(self, 'collection_service') and self.collection_service:
            await self.collection_service.add_task(
                request.team1_id, request.team2_id, request.league_id, Priority.HIGH
            )
        
        self.metrics.increment('h2h_fallback.queue_priority')
        return None  # Queued for later processing
    
    async def _execute_queue_normal(self, request: FallbackRequest) -> Optional[Dict]:
        """Execute normal queue strategy."""
        self.logger.info(f"Adding H2H request to normal queue: {request.request_id}")
        
        # Add to collection service if available
        if hasattr(self, 'collection_service') and self.collection_service:
            await self.collection_service.add_task(
                request.team1_id, request.team2_id, request.league_id, request.priority
            )
        
        self.metrics.increment('h2h_fallback.queue_normal')
        return None  # Queued for later processing
    
    async def _fetch_h2h_data(self, request: FallbackRequest) -> Optional[Dict]:
        """Fetch H2H data from API."""
        try:
            # Record API call timing
            self.request_times.append(time.time())
            
            # Make API call
            h2h_data = await self.api_client.get_h2h_stats(
                request.team1_id, request.team2_id, request.league_id
            )
            
            if h2h_data:
                # Cache the data
                await self._cache_h2h_data(request, h2h_data)
                return h2h_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching H2H data for {request.request_id}: {e}")
            raise
    
    async def _cache_h2h_data(self, request: FallbackRequest, h2h_data: Dict):
        """Cache H2H data in database."""
        try:
            # This would integrate with the H2H manager's caching logic
            cache_data = {
                'team1_id': request.team1_id,
                'team2_id': request.team2_id,
                'league_id': request.league_id,
                'h2h_data': json.dumps(h2h_data),
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=24)
            }
            
            # Insert or update cache
            query = """
                INSERT INTO h2h_cache (team1_id, team2_id, league_id, h2h_data, created_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (team1_id, team2_id, league_id) 
                DO UPDATE SET 
                    h2h_data = EXCLUDED.h2h_data,
                    created_at = EXCLUDED.created_at,
                    expires_at = EXCLUDED.expires_at
            """
            
            await self.db_session.execute(query, (
                cache_data['team1_id'],
                cache_data['team2_id'],
                cache_data['league_id'],
                cache_data['h2h_data'],
                cache_data['created_at'],
                cache_data['expires_at']
            ))
            
            await self.db_session.commit()
            self.logger.debug(f"Cached H2H data for {request.request_id}")
            
        except Exception as e:
            self.logger.error(f"Error caching H2H data for {request.request_id}: {e}")
    
    def _is_rate_limited(self) -> bool:
        """Check if we're currently rate limited."""
        current_time = time.time()
        
        # Remove old requests outside the window
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        return len(self.request_times) >= self.rate_limit_per_minute
    
    def _get_recent_success_rate(self) -> float:
        """Get recent success rate."""
        if not self.success_rate_window:
            return 1.0
        
        successes = sum(1 for success in self.success_rate_window if success)
        return successes / len(self.success_rate_window)
    
    def _get_average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        
        return sum(self.response_times) / len(self.response_times)
    
    async def get_status(self) -> Dict:
        """Get current fallback system status."""
        return {
            'circuit_breaker': {
                'state': self.circuit_breaker.state.value,
                'failure_count': self.circuit_breaker.failure_count,
                'last_failure_time': self.circuit_breaker.last_failure_time
            },
            'performance': {
                'success_rate': self._get_recent_success_rate(),
                'average_response_time': self._get_average_response_time(),
                'active_immediate_requests': self.active_immediate_requests,
                'pending_requests': len(self.pending_requests)
            },
            'rate_limiting': {
                'requests_in_window': len(self.request_times),
                'rate_limit_per_minute': self.rate_limit_per_minute,
                'is_rate_limited': self._is_rate_limited()
            },
            'configuration': {
                'immediate_timeout': self.immediate_timeout,
                'max_concurrent_immediate': self.max_concurrent_immediate,
                'rate_limit_per_minute': self.rate_limit_per_minute
            }
        }
    
    async def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self.circuit_breaker.state = CircuitState.CLOSED
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.last_failure_time = None
        self.circuit_breaker.half_open_calls = 0
        
        self.logger.info("Circuit breaker manually reset")
        self.metrics.increment('h2h_fallback.circuit_breaker.manual_reset')
    
    def set_collection_service(self, collection_service):
        """Set the collection service for queue strategies."""
        self.collection_service = collection_service
        self.logger.info("H2H collection service linked to fallback system")


async def create_h2h_fallback_system(config=None) -> H2HFallbackSystem:
    """Factory function to create and initialize H2H fallback system."""
    fallback_system = H2HFallbackSystem(config)
    await fallback_system.initialize()
    return fallback_system