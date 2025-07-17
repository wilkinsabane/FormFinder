"""

Enhanced Soccer Data Fetcher

===========================

A robust, modular soccer data fetching system with advanced features:
- Rate limiting and parallel processing
- Comprehensive caching mechanism
- Advanced error handling and logging
- Unit testing support
- Configuration management
- Data transformation and validation

Author: Enhanced DataFetcher
Version: 2.0.0
"""

import asyncio
import aiohttp
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm.asyncio import tqdm
import threading
from functools import wraps
import hashlib
import pickle
from pathlib import Path
import yaml
from pydantic import BaseModel, ValidationError, Field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



# Configuration Models
class APIConfig(BaseModel):
    """API configuration validation model."""
    auth_token: str = Field(..., min_length=1)
    base_url: str = Field(default="https://api.soccerdataapi.com")
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_period: int = Field(default=3600, ge=1)
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=5, ge=1)

class ProcessingConfig(BaseModel):
    """Processing configuration validation model."""
    league_ids: List[int] = Field(..., min_items=1)
    season_year: str = Field(..., pattern=r'^(\d{4}|\d{4}-\d{4})$')  # Fixed: regex -> pattern
    max_concurrent_requests: int = Field(default=5, ge=1, le=20)
    inter_league_delay: int = Field(default=10, ge=0)
    cache_ttl_hours: int = Field(default=24, ge=1)

class DataFetcherConfig(BaseModel):
    """Main configuration model."""
    api: APIConfig
    processing: ProcessingConfig
   
    @classmethod
    def from_file(cls, config_path: str) -> 'DataFetcherConfig':
        """Load configuration from file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
     
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
      
        return cls(**data)

# Rate Limiter
class RateLimiter:
    """Thread-safe rate limiter with token bucket algorithm."""
    """Much more efficient rate limiter."""
   
    def __init__(self, max_requests: int = 300, time_window: int = 60):  # 300 requests per minute instead of 100 per hour
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
  
    def acquire(self) -> bool:
        """Acquire permission to make a request."""
        with self.lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests
                           if now - req_time < self.time_window]
         
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    async def wait_if_needed(self):
        """Wait if rate limit is exceeded - much shorter waits."""
        while not self.acquire():
            await asyncio.sleep(0.2)  # Wait only 200ms instead of 1 second

# Cache System

class CacheManager:
    """File-based cache manager with TTL support."""
   
    def __init__(self, cache_dir: str, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
    
    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            stat = cache_path.stat()
            if time.time() - stat.st_mtime > self.ttl_seconds:
                cache_path.unlink()  # Remove expired cache
                return None
            
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Cache read error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any):
        """Store item in cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logging.warning(f"Cache write error for key {key}: {e}")
    
    def clear_expired(self):
        """Clear expired cache entries."""
        now = time.time()
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                if now - cache_file.stat().st_mtime > self.ttl_seconds:
                    cache_file.unlink()
            except Exception as e:
                logging.warning(f"Error clearing cache file {cache_file}: {e}")


# Enhanced Logger
class EnhancedLogger:
    """Centralized logger with rotation and structured formatting."""
    
    @staticmethod
    def setup_logger(name: str, log_dir: str = "data/logs", level: int = logging.INFO) -> logging.Logger:
        """Setup enhanced logger with rotation."""
        from logging.handlers import RotatingFileHandler
        
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

class DataFetcher:
    """Main class for fetching soccer data."""
    pass


# Data Models
@dataclass
class Match:
    """Match data model with validation."""
    id: str
    date: str
    time: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    status: str
    home_score: Optional[str] = None
    away_score: Optional[str] = None
    league_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and transform data after initialization."""
        self.date = self._standardize_date(self.date)
        self.time = self._standardize_time(self.time)
    
    def _standardize_date(self, date_str: str) -> str:
        """Standardize date format."""
        if not date_str or date_str == 'N/A':
            return 'N/A'
        
        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return date_str
        except Exception:
            return date_str
    
    def _standardize_time(self, time_str: str) -> str:
        """Standardize time format."""
        if not time_str or time_str == 'N/A':
            return 'N/A'
        
        try:
            # Try common time formats
            for fmt in ['%H:%M', '%H:%M:%S', '%I:%M %p']:
                try:
                    dt = datetime.strptime(time_str, fmt)
                    return dt.strftime('%H:%M')
                except ValueError:
                    continue
            return time_str
        except Exception:
            return time_str


@dataclass
class Standing:
    """Standing data model with validation."""
    position: int
    team_id: str
    team_name: str
    games_played: int
    points: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    league_name: Optional[str] = None
    
    @property
    def goal_difference(self) -> int:
        """Calculate goal difference."""
        return self.goals_for - self.goals_against


# Enhanced Data Fetcher
class EnhancedDataFetcher:
    """Enhanced soccer data fetcher with advanced features."""
    
    def __init__(self, config: DataFetcherConfig):
        self.config = config
        self.logger = EnhancedLogger.setup_logger("DataFetcher")
        self.rate_limiter = RateLimiter(
            config.api.rate_limit_requests,
            config.api.rate_limit_period
        )
        self.cache = CacheManager("data/cache", config.processing.cache_ttl_hours)
        
        # Setup directories
        self.directories = {
            'logs': 'data/logs',
            'historical': 'data/historical',
            'fixtures': 'data/fixtures',
            'standings': 'data/standings',
            'cache': 'data/cache'
        }
        
        for directory in self.directories.values():
            os.makedirs(directory, exist_ok=True)
        
        # Setup session with retry strategy
        self.session = self._create_session()
        
        league_ids = self.config.processing.league_ids or []
        self.logger.info(f"Initialized EnhancedDataFetcher for {len(league_ids)} leagues")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.api.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimized request method with faster failures."""
        await self.rate_limiter.wait_if_needed()

        cache_key = f"{url}:{json.dumps(params, sort_keys=True)}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        max_retries = 2  # Reduce from 5 to 2 retries
        
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:  # Reduce timeout from 30 to 15
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.cache.set(cache_key, data)
                            return data
                        elif response.status == 429:
                            if attempt < max_retries:
                                # Much shorter wait for 429 errors
                                wait_time = min(int(response.headers.get('Retry-After', 10)), 30)  # Max 30 seconds
                                self.logger.warning(f"Rate limited. Waiting {wait_time}s (attempt {attempt + 1})")
                                await asyncio.sleep(wait_time)
                                continue
                        elif 500 <= response.status < 600:
                            if attempt < max_retries:
                                # Shorter exponential backoff
                                wait_time = min(5 * (2 ** attempt), 30)  # Max 30 seconds
                                self.logger.warning(f"Server error {response.status}. Waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        self.logger.error(f"HTTP {response.status} for {url}")
                        return None
            
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(2)  # Short wait before retry
                    continue
            except Exception as e:
                self.logger.error(f"Error for {url}: {e}")
                return None
        
        return None
    
    async def fetch_leagues(self) -> Dict[int, str]:
        """Fetch all available leagues."""
        self.logger.info("Fetching all leagues")
        
        url = f"{self.config.api.base_url}/league/"
        params = {'auth_token': self.config.api.auth_token}
        leagues = {}
        page = 1
        
        while True:
            params['page'] = page
            data = await self._make_request(url, params)
            
            if not data or 'results' not in data or not data['results']:
                break
            
            for league in data['results']:
                league_id = league.get('id')
                league_name = league.get('name', f"UnnamedLeague{league_id}")
                if league_id:
                    leagues[league_id] = league_name
            
            if data.get('next') is None:
                break
            page += 1
        
        self.logger.info(f"Fetched {len(leagues)} leagues")
        return leagues
    
    async def fetch_historical_matches(self, league_id: int, league_name: str) -> List[Match]:
        """Fetch historical matches for a league and season."""
        self.logger.info(f"Fetching historical matches for {league_name} (ID: {league_id}), Season: {self.config.processing.season_year}")
        
        url = f"{self.config.api.base_url}/matches/"
        params = {
            'auth_token': self.config.api.auth_token,
            'league_id': league_id,
            'season': self.config.processing.season_year
        }
        
        matches: List[Match] = [] # Initialize list to store Match objects
        
        # Make a single request for this league/season
        # The _make_request method handles caching and basic retries for 429 internally
        api_response_data = await self._make_request(url, params) 
        
        if not api_response_data:
            self.logger.warning(f"No data returned from API (or _make_request failed) for historical matches: {league_name} (ID: {league_id})")
            return []

        # Expecting API response structure: list -> dict -> 'stage' (list) -> dict -> 'matches' (list)
        # Example: api_response_data = [ { "league_id": ..., "stage": [ { "stage_name": ..., "matches": [ ... ] } ] } ]
        try:
            if isinstance(api_response_data, list) and len(api_response_data) > 0:
                league_data_item = api_response_data[0] # First item in the outer list

                if isinstance(league_data_item, dict) and \
                  'stage' in league_data_item and \
                  isinstance(league_data_item['stage'], list) and \
                  len(league_data_item['stage']) > 0:
                    
                    # Assuming relevant matches are in the first stage. 
                    # If multiple stages could contain historical matches, you might need to loop through league_data_item['stage'].
                    stage_item = league_data_item['stage'][0] 

                    if isinstance(stage_item, dict) and \
                      'matches' in stage_item and \
                      isinstance(stage_item['matches'], list):
                        
                        actual_matches_list = stage_item['matches']
                        num_api_matches = len(actual_matches_list)
                        self.logger.info(f"API returned {num_api_matches} match entries in stage '{stage_item.get('stage_name', 'N/A')}' for {league_name} before filtering.")
                        
                        for match_data_dict in actual_matches_list: # Iterate through the list of match dictionaries
                            if isinstance(match_data_dict, dict) and match_data_dict.get('status') == 'finished':
                                try:
                                    match_obj = self._create_match_from_data(match_data_dict, league_name)
                                    matches.append(match_obj)
                                except Exception as e:
                                    self.logger.warning(f"Error processing/validating individual match data for ID {match_data_dict.get('id', 'N/A')} in {league_name}: {e}", exc_info=True)
                    else:
                        self.logger.warning(f"No 'matches' list found in first stage, first stage is not a dict, or 'matches' list is not a list for {league_name}. Stage content snippet: {str(stage_item)[:200]}")
                else:
                    self.logger.warning(f"No 'stage' list found in league data, it's empty, not a list, or league_data_item is not a dict for {league_name}. League data keys: {list(league_data_item.keys()) if isinstance(league_data_item, dict) else 'League data item not a dict'}")
            else:
                self.logger.warning(f"Unexpected API response structure for historical matches: {league_name}. Expected a non-empty list. Received: {str(api_response_data)[:500]}")
                
        except Exception as e:
            self.logger.error(f"General error during parsing of historical matches for {league_name}: {e}", exc_info=True)
            
        self.logger.info(f"Fetched and filtered {len(matches)} finished historical matches for {league_name}")
        return matches
    
    def _create_match_from_data(self, match_data: Dict[str, Any], league_name: str) -> Match:
        """Create Match object from API data."""
        return Match(
            id=str(match_data.get('id', 'N/A')),
            date=match_data.get('date', 'N/A'),
            time=match_data.get('time', 'N/A'),
            home_team_id=str(match_data.get('teams', {}).get('home', {}).get('id', 'N/A')),
            home_team_name=match_data.get('teams', {}).get('home', {}).get('name', 'N/A'),
            away_team_id=str(match_data.get('teams', {}).get('away', {}).get('id', 'N/A')),
            away_team_name=match_data.get('teams', {}).get('away', {}).get('name', 'N/A'),
            status=match_data.get('status', 'N/A'),
            home_score=str(match_data.get('goals', {}).get('home_ft_goals', 'N/A')),
            away_score=str(match_data.get('goals', {}).get('away_ft_goals', 'N/A')),
            league_name=league_name
        )
    
    async def fetch_upcoming_fixtures(self) -> Dict[int, List[Match]]:
        """Fetch upcoming fixtures for all configured leagues."""
        self.logger.info("Fetching upcoming fixtures")
        
        url = f"{self.config.api.base_url}/match-previews-upcoming/"
        params = {'auth_token': self.config.api.auth_token}
        
        data = await self._make_request(url, params)
        if not data or 'results' not in data:
            self.logger.error("No upcoming fixtures data received")
            return {}
        
        fixtures_by_league = {}
        for league_fixture_data in data['results']:
            league_id = league_fixture_data.get('league_id')
            league_ids = self.config.processing.league_ids or []
            if league_id in league_ids and 'match_previews' in league_fixture_data:
                fixtures = []
                for preview in league_fixture_data['match_previews']:
                    try:
                        match = self._create_match_from_preview(preview)
                        fixtures.append(match)
                    except Exception as e:
                        self.logger.warning(f"Error processing fixture {preview.get('id')}: {e}")
                
                fixtures_by_league[league_id] = fixtures
        
        self.logger.info(f"Fetched upcoming fixtures for {len(fixtures_by_league)} leagues")
        return fixtures_by_league
    
    def _create_match_from_preview(self, preview_data: Dict[str, Any]) -> Match:
        """Create Match object from preview data."""
        return Match(
            id=str(preview_data.get('id', 'N/A')),
            date=preview_data.get('date', 'N/A'),
            time=preview_data.get('time', 'N/A'),
            home_team_id=str(preview_data.get('teams', {}).get('home', {}).get('id', 'N/A')),
            home_team_name=preview_data.get('teams', {}).get('home', {}).get('name', 'N/A'),
            away_team_id=str(preview_data.get('teams', {}).get('away', {}).get('id', 'N/A')),
            away_team_name=preview_data.get('teams', {}).get('away', {}).get('name', 'N/A'),
            status='scheduled',
            home_score='N/A',
            away_score='N/A'
        )
    
    async def fetch_standings_async(self, league_id: int, league_name: str) -> List[Standing]:
        """Fetch standings for a league."""
        self.logger.info(f"Fetching standings for {league_name} (ID: {league_id})")
       
        url = f"{self.config.api.base_url}/standing/"
        params = {
            'auth_token': self.config.api.auth_token,
            'league_id': league_id
        }
      
        data = await self._make_request(url, params)
        if not data:
            return []
       
        standings = []
       
        if 'stage' in data and data['stage']:
            for stage_data in data['stage']:
                if isinstance(stage_data, dict) and 'standings' in stage_data:
                    for standing_data in stage_data['standings']:
                        try:
                            standing = Standing(
                                position=int(standing_data.get('position', 0)),
                                team_id=str(standing_data.get('team_id', 'N/A')),
                                team_name=standing_data.get('team_name', 'N/A'),
                                games_played=int(standing_data.get('games_played', 0)),
                                points=int(standing_data.get('points', 0)),
                                wins=int(standing_data.get('wins', 0)),
                                draws=int(standing_data.get('draws', 0)),
                                losses=int(standing_data.get('losses', 0)),
                                goals_for=int(standing_data.get('goals_for', 0)),
                                goals_against=int(standing_data.get('goals_against', 0)),
                                league_name=league_name
                            )
                            standings.append(standing)
                        except Exception as e:
                            self.logger.warning(f"Error processing standing: {e}")
                    break  # Use first stage with standings
       
        self.logger.info(f"Fetched {len(standings)} standings for {league_name}")
        return standings
    
    def save_matches_to_csv(self, matches: List[Match], filename: str, mode: str = 'w'):
        """Save matches to CSV file."""
        if not matches:
            self.logger.warning(f"No matches to save to {filename}")
            return
        
        try:
            data = []
            for match in matches:
                data.append({
                    'league_name': match.league_name,
                    'match_id': match.id,
                    'date': match.date,
                    'time': match.time,
                    'home_team_id': match.home_team_id,
                    'home_team_name': match.home_team_name,
                    'away_team_id': match.away_team_id,
                    'away_team_name': match.away_team_name,
                    'status': match.status,
                    'home_score': match.home_score,
                    'away_score': match.away_score
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, mode=mode, header=(mode == 'w'), index=False)
            self.logger.info(f"Saved {len(data)} matches to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving matches to {filename}: {e}")
    
    def save_standings_to_csv(self, standings: List[Standing], filename: str):
        """Save standings to CSV file."""
        if not standings:
            self.logger.warning(f"No standings to save to {filename}")
            return
        
        try:
            data = []
            for standing in standings:
                data.append({
                    'league_name': standing.league_name,
                    'position': standing.position,
                    'team_id': standing.team_id,
                    'team_name': standing.team_name,
                    'games_played': standing.games_played,
                    'points': standing.points,
                    'wins': standing.wins,
                    'draws': standing.draws,
                    'losses': standing.losses,
                    'goals_for': standing.goals_for,
                    'goals_against': standing.goals_against,
                    'goal_difference': standing.goal_difference
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(data)} standings to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving standings to {filename}: {e}")
    
    async def process_league(self, league_id: int, league_name: str, semaphore: asyncio.Semaphore):
        """Process data for a single league."""
        async with semaphore:
            self.logger.info(f"Processing league: {league_name} (ID: {league_id})")
            
            # Define file paths
            historical_file = os.path.join(
                self.directories['historical'],
                f"league_{league_id}_{self.config.processing.season_year}_historical_matches.csv"
            )
            fixtures_file = os.path.join(
                self.directories['fixtures'],
                f"league_{league_id}_upcoming_fixtures.csv"
            )
            standings_file = os.path.join(
                self.directories['standings'],
                f"league_{league_id}_{self.config.processing.season_year}_standings.csv"
            )
            
            tasks = []
            
            # Fetch historical matches (with file existence check)
            if not (os.path.exists(historical_file) and os.path.getsize(historical_file) > 200):
                tasks.append(('historical', self.fetch_historical_matches(league_id, league_name)))
            else:
                self.logger.info(f"Historical data exists for {league_name}, skipping fetch")
            
            # Fetch standings
            tasks.append(('standings', self.fetch_standings(league_id, league_name)))
            
            # Execute tasks
            results = {}
            for task_name, task in tasks:
                try:
                    results[task_name] = await task
                except Exception as e:
                    self.logger.error(f"Error in {task_name} task for {league_name}: {e}")
                    results[task_name] = []
            
            # Save results
            if 'historical' in results and results['historical']:
                self.save_matches_to_csv(results['historical'], historical_file)
            
            if 'standings' in results and results['standings']:
                self.save_standings_to_csv(results['standings'], standings_file)
                
                
    # BONUS: Summary report generation
    async def generate_summary_report(self):
        """Generate a summary report of all fetched data."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'leagues': {},
                'total_files': 0,
                'total_size_mb': 0
            }
            
            league_ids = self.config.processing.league_ids or []
            for league_id in league_ids:
                league_summary = {
                    'historical_matches': 0,
                    'standings': 0,
                    'fixtures': 0,
                    'files_created': []
                }
                
                # Check historical matches file
                historical_file = os.path.join(
                    self.directories['historical'],
                    f"league_{league_id}_{self.config.processing.season_year}_historical_matches.csv"
                )
                if os.path.exists(historical_file):
                    league_summary['files_created'].append(os.path.basename(historical_file))
                    try:
                        df = pd.read_csv(historical_file)
                        league_summary['historical_matches'] = len(df)
                        summary['total_size_mb'] += os.path.getsize(historical_file) / (1024*1024)
                        summary['total_files'] += 1
                    except Exception as e:
                        self.logger.warning(f"Error reading historical file for league {league_id}: {e}")
                
                # Check standings file
                standings_file = os.path.join(
                    self.directories['standings'],
                    f"league_{league_id}_{self.config.processing.season_year}_standings.csv"
                )
                if os.path.exists(standings_file):
                    league_summary['files_created'].append(os.path.basename(standings_file))
                    try:
                        df = pd.read_csv(standings_file)
                        league_summary['standings'] = len(df)
                        summary['total_size_mb'] += os.path.getsize(standings_file) / (1024*1024)
                        summary['total_files'] += 1
                    except Exception as e:
                        self.logger.warning(f"Error reading standings file for league {league_id}: {e}")
                
                # Check fixtures file
                fixtures_file = os.path.join(
                    self.directories['fixtures'],
                    f"league_{league_id}_upcoming_fixtures.csv"
                )
                if os.path.exists(fixtures_file):
                    league_summary['files_created'].append(os.path.basename(fixtures_file))
                    try:
                        df = pd.read_csv(fixtures_file)
                        league_summary['fixtures'] = len(df)
                        summary['total_size_mb'] += os.path.getsize(fixtures_file) / (1024*1024)
                        summary['total_files'] += 1
                    except Exception as e:
                        self.logger.warning(f"Error reading fixtures file for league {league_id}: {e}")
                
                summary['leagues'][league_id] = league_summary
            
            # Save summary report
            summary_file = os.path.join(self.directories['logs'], 'fetch_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Log summary
            self.logger.info("=== FETCH SUMMARY ===")
            self.logger.info(f"Total files created: {summary['total_files']}")
            self.logger.info(f"Total data size: {summary['total_size_mb']:.2f} MB")
            
            for league_id, league_data in summary['leagues'].items():
                self.logger.info(f"League {league_id}: {league_data['historical_matches']} matches, "
                               f"{league_data['standings']} standings, {league_data['fixtures']} fixtures")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}", exc_info=True)


    # 4. SEPARATE TASK METHODS for better parallelization
    async def fetch_historical_matches_task(self, league_id: int, league_name: str, semaphore: asyncio.Semaphore):
        """Separate task for historical matches."""
        async with semaphore:
            try:
                historical_file = os.path.join(
                    self.directories['historical'],
                    f"league_{league_id}_{self.config.processing.season_year}_historical_matches.csv"
                )
                
                # Skip if file exists and is recent (less than 24 hours old)
                if os.path.exists(historical_file):
                    file_age = time.time() - os.path.getmtime(historical_file)
                    if file_age < 24 * 3600 and os.path.getsize(historical_file) > 200:
                        self.logger.info(f"Recent historical data exists for {league_name}, skipping")
                        return
                
                self.logger.info(f"Starting historical matches task for {league_name}")
                start_time = time.time()
                
                matches = await self.fetch_historical_matches(league_id, league_name)
                if matches:
                    self.save_matches_to_csv(matches, historical_file)
                    self.logger.info(f"Historical matches task completed for {league_name} in {time.time() - start_time:.2f}s")
                else:
                    self.logger.warning(f"No historical matches found for {league_name}")
                    
            except Exception as e:
                self.logger.error(f"Error in historical matches task for {league_name}: {e}", exc_info=True)


    async def fetch_standings_task(self, league_id: int, league_name: str, semaphore: asyncio.Semaphore):
        """Separate task for standings."""
        async with semaphore:
            try:
                standings_file = os.path.join(
                    self.directories['standings'],
                    f"league_{league_id}_{self.config.processing.season_year}_standings.csv"
                )
                
                # Check if file exists and is recent (less than 6 hours old for standings)
                if os.path.exists(standings_file):
                    file_age = time.time() - os.path.getmtime(standings_file)
                    if file_age < 6 * 3600 and os.path.getsize(standings_file) > 100:
                        self.logger.info(f"Recent standings data exists for {league_name}, skipping")
                        return
                
                self.logger.info(f"Starting standings task for {league_name}")
                start_time = time.time()
                
                standings = await self.fetch_standings(league_id, league_name)
                if standings:
                    self.save_standings_to_csv(standings, standings_file)
                    self.logger.info(f"Standings task completed for {league_name} in {time.time() - start_time:.2f}s")
                else:
                    self.logger.warning(f"No standings found for {league_name}")
                    
            except Exception as e:
                self.logger.error(f"Error in standings task for {league_name}: {e}", exc_info=True)


    async def fetch_fixtures_task(self, league_id: int, league_name: str, semaphore: asyncio.Semaphore):
        """Separate task for upcoming fixtures."""
        async with semaphore:
            try:
                fixtures_file = os.path.join(
                    self.directories['fixtures'],
                    f"league_{league_id}_upcoming_fixtures.csv"
                )
                
                # Check if file exists and is recent (less than 2 hours old for fixtures)
                if os.path.exists(fixtures_file):
                    file_age = time.time() - os.path.getmtime(fixtures_file)
                    if file_age < 2 * 3600 and os.path.getsize(fixtures_file) > 100:
                        self.logger.info(f"Recent fixtures data exists for {league_name}, skipping")
                        return
                
                self.logger.info(f"Starting fixtures task for {league_name}")
                start_time = time.time()
                
                # Fetch fixtures for this specific league
                fixtures = await self.fetch_upcoming_fixtures_for_league(league_id, league_name)
                if fixtures:
                    # Update league names for fixtures
                    for fixture in fixtures:
                        fixture.league_name = league_name
                    
                    self.save_matches_to_csv(fixtures, fixtures_file)
                    self.logger.info(f"Fixtures task completed for {league_name} in {time.time() - start_time:.2f}s")
                else:
                    self.logger.warning(f"No upcoming fixtures found for {league_name}")
                    
            except Exception as e:
                self.logger.error(f"Error in fixtures task for {league_name}: {e}", exc_info=True)


    async def fetch_upcoming_fixtures_for_league(self, league_id: int, league_name: str) -> List[Match]:
        """Fetch upcoming fixtures for a specific league."""
        self.logger.info(f"Fetching upcoming fixtures for {league_name} (ID: {league_id})")
        
        url = f"{self.config.api.base_url}/match-previews-upcoming/"
        params = {
            'auth_token': self.config.api.auth_token,
            'league_id': league_id  # Filter by specific league if API supports it
        }
        
        data = await self._make_request(url, params)
        if not data or 'results' not in data:
            self.logger.error(f"No upcoming fixtures data received for {league_name}")
            return []
        
        fixtures = []
        for league_fixture_data in data['results']:
            # Check if this is the league we want
            if league_fixture_data.get('league_id') == league_id and 'match_previews' in league_fixture_data:
                for preview in league_fixture_data['match_previews']:
                    try:
                        match = self._create_match_from_preview(preview)
                        match.league_name = league_name
                        fixtures.append(match)
                    except Exception as e:
                        self.logger.warning(f"Error processing fixture {preview.get('id')} for {league_name}: {e}")
                break
        
        self.logger.info(f"Fetched {len(fixtures)} upcoming fixtures for {league_name}")
        return fixtures


    # Alternative method if API doesn't support league filtering for fixtures
    async def fetch_all_fixtures_task(self, all_leagues: Dict[int, str], semaphore: asyncio.Semaphore):
        """Fetch all upcoming fixtures in one call and distribute by league."""
        async with semaphore:
            try:
                self.logger.info("Starting combined fixtures task for all leagues")
                start_time = time.time()
                
                # Fetch all upcoming fixtures
                url = f"{self.config.api.base_url}/match-previews-upcoming/"
                params = {'auth_token': self.config.api.auth_token}
                
                data = await self._make_request(url, params)
                if not data or 'results' not in data:
                    self.logger.error("No upcoming fixtures data received")
                    return
                
                # Group fixtures by league
                fixtures_by_league = {}
                for league_fixture_data in data['results']:
                    league_id = league_fixture_data.get('league_id')
                    league_ids = self.config.processing.league_ids or []
                    if league_id in league_ids and 'match_previews' in league_fixture_data:
                        fixtures = []
                        league_name = all_leagues.get(league_id, f"League-{league_id}")
                        
                        for preview in league_fixture_data['match_previews']:
                            try:
                                match = self._create_match_from_preview(preview)
                                match.league_name = league_name
                                fixtures.append(match)
                            except Exception as e:
                                self.logger.warning(f"Error processing fixture {preview.get('id')}: {e}")
                        
                        fixtures_by_league[league_id] = fixtures
                
                # Save fixtures for each league
                for league_id, fixtures in fixtures_by_league.items():
                    if fixtures:
                        league_name = all_leagues.get(league_id, f"League-{league_id}")
                        fixtures_file = os.path.join(
                            self.directories['fixtures'],
                            f"league_{league_id}_upcoming_fixtures.csv"
                        )
                        self.save_matches_to_csv(fixtures, fixtures_file)
                
                self.logger.info(f"Combined fixtures task completed for {len(fixtures_by_league)} leagues in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error in combined fixtures task: {e}", exc_info=True)
    
    async def run(self):
        """Main execution method with parallel processing."""
        self.logger.info("Starting enhanced data fetch process")
        
        try:
            # Clear expired cache
            self.cache.clear_expired()
            
            # Increase concurrent requests significantly
            max_concurrent = min(self.config.processing.max_concurrent_requests * 2, 12)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Fetch leagues first
            self.logger.info("Fetching all leagues...")
            all_leagues = await self.fetch_leagues()
            self.logger.info(f"Found {len(all_leagues)} total leagues")
            
            # Create separate tasks for each data type and league combination
            all_tasks = []
            
            # Option 1: Individual tasks per league (more granular control)
            league_ids = self.config.processing.league_ids or []
            for league_id in league_ids:
                league_name = all_leagues.get(league_id, f"League-{league_id}")
                self.logger.info(f"Scheduling tasks for {league_name} (ID: {league_id})")
                
                # Create separate tasks for each data type
                all_tasks.extend([
                    self.fetch_historical_matches_task(league_id, league_name, semaphore),
                    self.fetch_standings_task(league_id, league_name, semaphore)
                    # self.fetch_fixtures_task(league_id, league_name, semaphore)
                ])
            
            # Option 2: Combined fixtures task (uncomment if you prefer single fixtures call)
            all_tasks.append(self.fetch_all_fixtures_task(all_leagues, semaphore))
            
            self.logger.info(f"Executing {len(all_tasks)} tasks in parallel with max {max_concurrent} concurrent requests")
            
            # Execute ALL tasks in parallel
            start_time = time.time()
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Process results and count successes/failures
            successful_tasks = 0
            failed_tasks = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {i+1} failed: {result}")
                    failed_tasks += 1
                else:
                    successful_tasks += 1
            
            self.logger.info(f"OPTIMIZED data fetch completed in {total_time:.2f}s")
            self.logger.info(f"Results: {successful_tasks} successful, {failed_tasks} failed tasks")
            
            # Generate summary report
            await self.generate_summary_report()
            
        except Exception as e:
            self.logger.error(f"Critical error in optimized run: {e}", exc_info=True)
            raise
    
    # Synchronous wrapper methods for workflow compatibility
    def fetch_fixtures(self, league_id: int) -> List[Match]:
        """Synchronous wrapper for fetching fixtures for a specific league."""
        try:
            # Get league name from config or use default
            league_name = f"League-{league_id}"
            
            # Run async method synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.fetch_upcoming_fixtures_for_league(league_id, league_name)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in fetch_fixtures for league {league_id}: {e}")
            return []
    
    def fetch_standings_sync(self, league_id: int, league_name: str) -> List[Standing]:
        """Synchronous wrapper for fetching standings for a specific league."""
        try:
            # Run async method synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.fetch_standings(league_id, league_name)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in fetch_standings for league {league_id}: {e}")
            return []
    
    def fetch_standings(self, league_id: int, league_name: str = None) -> List[Standing]:
        """Synchronous method for fetching standings - workflow compatible."""
        if league_name is None:
            league_name = f"League-{league_id}"
        
        try:
            # Run async method synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.fetch_standings_async(league_id, league_name)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in fetch_standings for league {league_id}: {e}")
            return []
    
    def fetch_teams(self, league_id: int) -> List[Dict[str, Any]]:
        """Synchronous wrapper for fetching teams for a specific league."""
        try:
            # This method doesn't exist in async form, so return empty list for now
            # TODO: Implement async fetch_teams method if needed
            self.logger.warning(f"fetch_teams not implemented for league {league_id}")
            return []
        except Exception as e:
            self.logger.error(f"Error in fetch_teams for league {league_id}: {e}")
            return []





# Main execution
async def main():
    """Main entry point."""
    try:
        # Load configuration
        config = DataFetcherConfig.from_file('sdata_config.json')  # or config.json
        
        # Create and run fetcher
        fetcher = EnhancedDataFetcher(config)
        await fetcher.run()
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Alias for backward compatibility
DataFetcher = EnhancedDataFetcher

if __name__ == "__main__":
    asyncio.run(main())
