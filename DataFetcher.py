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
from datetime import datetime, timedelta, date
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

# Database imports
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
# Assuming database_setup.py is adjusted to use app_config.database_url or similar
# For now, direct import might work if database_setup.py doesn't rely on AppConfig at import time.
from database_setup import SessionLocal, League, Team, Match as DBMatch, Standing as DBStanding # engine might not be needed directly

# Configuration import
from config.config import app_config, DataFetcherAppConfig # Using the global app_config

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
    
    def __init__(self, fetcher_config: DataFetcherAppConfig, db_session: Session):
        self.config = fetcher_config # Changed to DataFetcherAppConfig
        self.logger = EnhancedLogger.setup_logger("DataFetcher", log_dir=str(self.config.log_dir))
        self.rate_limiter = RateLimiter(
            self.config.api.rate_limit_requests,
            self.config.api.rate_limit_period
        )
        self.cache = CacheManager(str(self.config.cache_dir), self.config.processing.cache_ttl_hours)
        
        # Ensure log and cache directories exist (Pydantic models ensure they are DirectoryPath objects)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)

        # Setup session with retry strategy
        self.session = self._create_session()
        self.db_session = db_session # Use passed-in session

        self.logger.info(f"Initialized EnhancedDataFetcher for {len(self.config.processing.league_ids)} leagues")

    def _get_or_create_league(self, api_league_id: int, league_name: str) -> League:
        league = self.db_session.query(League).filter_by(api_league_id=api_league_id).first()
        if not league:
            league = League(api_league_id=api_league_id, name=league_name)
            self.db_session.add(league)
            try:
                self.db_session.commit()
                self.db_session.refresh(league)
                self.logger.info(f"Created new league: {league_name} (API ID: {api_league_id})")
            except IntegrityError:
                self.db_session.rollback()
                league = self.db_session.query(League).filter_by(api_league_id=api_league_id).first()
                self.logger.warning(f"League {league_name} (API ID: {api_league_id}) already exists after rollback.")
            except Exception as e:
                self.db_session.rollback()
                self.logger.error(f"Error creating league {league_name}: {e}")
                raise
        return league

    def _get_or_create_team(self, api_team_id: str, team_name: str) -> Team:
        if not api_team_id or api_team_id == 'N/A':
            self.logger.warning(f"Attempted to get/create team with invalid API ID: {api_team_id}, Name: {team_name}")
            return None # Or raise an error, depending on desired strictness

        team = self.db_session.query(Team).filter_by(api_team_id=api_team_id).first()
        if not team:
            team = Team(api_team_id=api_team_id, name=team_name)
            self.db_session.add(team)
            try:
                self.db_session.commit()
                self.db_session.refresh(team)
                self.logger.info(f"Created new team: {team_name} (API ID: {api_team_id})")
            except IntegrityError:
                self.db_session.rollback()
                team = self.db_session.query(Team).filter_by(api_team_id=api_team_id).first()
                self.logger.warning(f"Team {team_name} (API ID: {api_team_id}) already exists after rollback.")
            except Exception as e:
                self.db_session.rollback()
                self.logger.error(f"Error creating team {team_name}: {e}")
                raise
        elif team.name != team_name: # Update team name if it changed
            team.name = team_name
            try:
                self.db_session.commit()
                self.logger.info(f"Updated team name for API ID {api_team_id} to {team_name}")
            except Exception as e:
                self.db_session.rollback()
                self.logger.error(f"Error updating team name {team_name}: {e}")
        return team

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
        
        self.logger.info(f"Fetched {len(leagues)} leagues from API")

        # Store/update leagues in DB
        for api_id, name in leagues.items():
            self._get_or_create_league(api_league_id=api_id, league_name=name)

        # Return a dict of DB league_id to name for internal use if needed, or just API IDs
        # For now, let's assume downstream tasks will use API IDs which are then mapped to DB IDs.
        return leagues # still returns API ID -> name mapping
    
    async def fetch_historical_matches(self, league_api_id: int, league_name: str) -> int:
        """Fetch historical matches for a league and season and save to DB."""
        self.logger.info(f"Fetching historical matches for {league_name} (API ID: {league_api_id}), Season: {self.config.processing.season_year}")
        
        db_league = self._get_or_create_league(api_league_id=league_api_id, league_name=league_name)
        if not db_league:
            self.logger.error(f"Could not get/create DB league for API ID {league_api_id}. Skipping historical matches.")
            return 0

        url = f"{self.config.api.base_url}/matches/"
        params = {
            'auth_token': self.config.api.auth_token,
            'league_id': league_id,
            'season': self.config.processing.season_year
        }
        matches_processed_count = 0
        
        # Make a single request for this league/season
        # The _make_request method handles caching and basic retries for 429 internally
        api_response_data = await self._make_request(url, params) 
        
        if not api_response_data:
            self.logger.warning(f"No data returned from API (or _make_request failed) for historical matches: {league_name} (API ID: {league_api_id})")
            return 0

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
                        
                        for match_data_dict in actual_matches_list:
                            if isinstance(match_data_dict, dict) and match_data_dict.get('status') == 'finished':
                                try:
                                    match_obj = self._create_match_from_data(match_data_dict, league_name) # This is the Pydantic model
                                    if match_obj:
                                        # Get/Create DB Team instances
                                        home_team_db = self._get_or_create_team(match_obj.home_team_id, match_obj.home_team_name)
                                        away_team_db = self._get_or_create_team(match_obj.away_team_id, match_obj.away_team_name)

                                        if not home_team_db or not away_team_db:
                                            self.logger.warning(f"Skipping match {match_obj.id} due to missing team data.")
                                            continue

                                        # Parse date and time
                                        try:
                                            match_datetime_str = f"{match_obj.date} {match_obj.time}"
                                            match_datetime = datetime.strptime(match_datetime_str, '%Y-%m-%d %H:%M')
                                        except ValueError:
                                            # Fallback if time is missing or format is unexpected
                                            try:
                                                match_datetime = datetime.strptime(match_obj.date, '%Y-%m-%d')
                                            except ValueError:
                                                self.logger.warning(f"Invalid date format for match {match_obj.id}: {match_obj.date}. Skipping.")
                                                continue

                                        # Check if match already exists
                                        existing_match = self.db_session.query(DBMatch).filter_by(api_match_id=str(match_obj.id)).first()
                                        if existing_match:
                                            # Optionally update if needed, for now, we skip
                                            # self.logger.debug(f"Match {match_obj.id} already exists. Skipping.")
                                            continue

                                        db_match_entry = DBMatch(
                                            api_match_id=str(match_obj.id),
                                            date=match_datetime.date(), # Store date part
                                            time=match_datetime.strftime('%H:%M') if match_obj.time and match_obj.time != 'N/A' else None,
                                            league_id=db_league.id,
                                            home_team_id=home_team_db.id,
                                            away_team_id=away_team_db.id,
                                            status=match_obj.status,
                                            home_score=int(match_obj.home_score) if match_obj.home_score and match_obj.home_score.isdigit() else None,
                                            away_score=int(match_obj.away_score) if match_obj.away_score and match_obj.away_score.isdigit() else None,
                                            is_fixture=0 # Historical match
                                        )
                                        self.db_session.add(db_match_entry)
                                        matches_processed_count += 1
                                except Exception as e:
                                    self.logger.warning(f"Error processing/saving individual match data for API match ID {match_data_dict.get('id', 'N/A')} in {league_name}: {e}", exc_info=True)

                        if matches_processed_count > 0:
                            try:
                                self.db_session.commit()
                                self.logger.info(f"Committed {matches_processed_count} new historical matches for {league_name} to DB.")
                            except Exception as e:
                                self.db_session.rollback()
                                self.logger.error(f"DB Error committing historical matches for {league_name}: {e}", exc_info=True)
                                matches_processed_count = 0 # Reset count as commit failed
                    else:
                        self.logger.warning(f"No 'matches' list found in first stage, first stage is not a dict, or 'matches' list is not a list for {league_name}. Stage content snippet: {str(stage_item)[:200]}")
                else:
                    self.logger.warning(f"No 'stage' list found in league data, it's empty, not a list, or league_data_item is not a dict for {league_name}. League data keys: {list(league_data_item.keys()) if isinstance(league_data_item, dict) else 'League data item not a dict'}")
            else:
                self.logger.warning(f"Unexpected API response structure for historical matches: {league_name}. Expected a non-empty list. Received: {str(api_response_data)[:500]}")
                
        except Exception as e:
            self.logger.error(f"General error during parsing of historical matches for {league_name}: {e}", exc_info=True)

        self.logger.info(f"Processed {matches_processed_count} finished historical matches for {league_name} into DB.")
        return matches_processed_count
    
    def _create_match_from_data(self, match_data: Dict[str, Any], league_name: str) -> Match: # Returns Pydantic Model
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
            league_name=league_name # This is Pydantic model's league_name, not DB one
        )
    
    async def fetch_upcoming_fixtures_for_league(self, league_api_id: int, league_name: str) -> int:
        """Fetch upcoming fixtures for a specific league and save to DB."""
        self.logger.info(f"Fetching upcoming fixtures for {league_name} (API ID: {league_api_id})")

        db_league = self.db_session.query(League).filter_by(api_league_id=league_api_id).first()
        if not db_league:
            # Attempt to create if it wasn't picked up by initial league scan
            db_league = self._get_or_create_league(api_league_id=league_api_id, league_name=league_name)
            if not db_league:
                self.logger.error(f"Could not get/create DB league for API ID {league_api_id}. Skipping fixtures.")
                return 0
        
        url = f"{self.config.api.base_url}/match-previews-upcoming/"
        params = {
            'auth_token': self.config.api.auth_token,
            'league_id': league_api_id
        }
        
        data = await self._make_request(url, params)
        if not data or 'results' not in data:
            self.logger.error(f"No upcoming fixtures data received for {league_name}")
            return 0
        
        fixtures_processed_count = 0
        for league_fixture_data_item in data['results']: # API returns a list of dicts, one per league
            if league_fixture_data_item.get('league_id') == league_api_id and 'match_previews' in league_fixture_data_item:
                for preview_data in league_fixture_data_item['match_previews']:
                    try:
                        match_obj = self._create_match_from_preview(preview_data, league_name) # Pydantic model
                        if match_obj:
                            home_team_db = self._get_or_create_team(match_obj.home_team_id, match_obj.home_team_name)
                            away_team_db = self._get_or_create_team(match_obj.away_team_id, match_obj.away_team_name)

                            if not home_team_db or not away_team_db:
                                self.logger.warning(f"Skipping fixture {match_obj.id} due to missing team data.")
                                continue

                            try:
                                fixture_datetime_str = f"{match_obj.date} {match_obj.time}"
                                fixture_datetime = datetime.strptime(fixture_datetime_str, '%Y-%m-%d %H:%M')
                            except ValueError:
                                try:
                                    fixture_datetime = datetime.strptime(match_obj.date, '%Y-%m-%d')
                                except ValueError:
                                    self.logger.warning(f"Invalid date format for fixture {match_obj.id}: {match_obj.date}. Skipping.")
                                    continue

                            existing_fixture = self.db_session.query(DBMatch).filter_by(api_match_id=str(match_obj.id), is_fixture=1).first()
                            if existing_fixture:
                                # Optionally update status, time, etc.
                                # For now, skip if exists to avoid duplicates before proper update logic
                                # self.logger.debug(f"Fixture {match_obj.id} already exists. Skipping.")
                                continue

                            db_fixture_entry = DBMatch(
                                api_match_id=str(match_obj.id),
                                date=fixture_datetime.date(),
                                time=fixture_datetime.strftime('%H:%M') if match_obj.time and match_obj.time != 'N/A' else None,
                                league_id=db_league.id,
                                home_team_id=home_team_db.id,
                                away_team_id=away_team_db.id,
                                status=match_obj.status, # Should be 'scheduled' or similar
                                is_fixture=1 # Mark as fixture
                            )
                            self.db_session.add(db_fixture_entry)
                            fixtures_processed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error processing/saving fixture {preview_data.get('id')} for {league_name}: {e}", exc_info=True)
                break # Found the league, no need to check other items in 'results'
        
        if fixtures_processed_count > 0:
            try:
                self.db_session.commit()
                self.logger.info(f"Committed {fixtures_processed_count} new upcoming fixtures for {league_name} to DB.")
            except Exception as e:
                self.db_session.rollback()
                self.logger.error(f"DB Error committing upcoming fixtures for {league_name}: {e}", exc_info=True)
                fixtures_processed_count = 0

        self.logger.info(f"Fetched and processed {fixtures_processed_count} upcoming fixtures for {league_name}")
        return fixtures_processed_count
    
    def _create_match_from_preview(self, preview_data: Dict[str, Any], league_name_for_pydantic: str) -> Match: # Returns Pydantic Model
        """Create Match object from preview data."""
        return Match(
            id=str(preview_data.get('id', 'N/A')),
            date=preview_data.get('date', 'N/A'),
            time=preview_data.get('time', 'N/A'),
            home_team_id=str(preview_data.get('teams', {}).get('home', {}).get('id', 'N/A')),
            home_team_name=preview_data.get('teams', {}).get('home', {}).get('name', 'N/A'),
            away_team_id=str(preview_data.get('teams', {}).get('away', {}).get('id', 'N/A')),
            away_team_name=preview_data.get('teams', {}).get('away', {}).get('name', 'N/A'),
            status='scheduled', # Pydantic model field
            home_score='N/A',   # Pydantic model field
            away_score='N/A',   # Pydantic model field
            league_name=league_name_for_pydantic # Pydantic model field
        )

    async def fetch_standings(self, league_api_id: int, league_name: str) -> int:
        """Fetch standings for a league and save to DB."""
        self.logger.info(f"Fetching standings for {league_name} (API ID: {league_api_id})")

        db_league = self._get_or_create_league(api_league_id=league_api_id, league_name=league_name)
        if not db_league:
            self.logger.error(f"Could not get/create DB league for API ID {league_api_id}. Skipping standings.")
            return 0

        url = f"{self.config.api.base_url}/standing/"
        params = {
            'auth_token': self.config.api.auth_token,
            'league_id': league_api_id
        }
      
        data = await self._make_request(url, params)
        if not data:
            self.logger.warning(f"No standings data received from API for {league_name}")
            return 0
       
        standings_processed_count = 0
       
        if 'stage' in data and data['stage']:
            for stage_data in data['stage']: # API returns list of stages
                if isinstance(stage_data, dict) and 'standings' in stage_data:
                    for standing_data_dict in stage_data['standings']:
                        try:
                            # Use Pydantic model for validation if desired, or directly populate DB model
                            # For simplicity, directly populating DB model here.
                            team_api_id = str(standing_data_dict.get('team_id', 'N/A'))
                            team_name = standing_data_dict.get('team_name', 'N/A')

                            db_team = self._get_or_create_team(api_team_id=team_api_id, team_name=team_name)
                            if not db_team:
                                self.logger.warning(f"Could not get/create team {team_name} (API ID: {team_api_id}) for standings. Skipping.")
                                continue

                            # Upsert logic: Check if standing for this league, team, season already exists
                            existing_standing = self.db_session.query(DBStanding).filter_by(
                                league_id=db_league.id,
                                team_id=db_team.id,
                                season_year=self.config.processing.season_year
                            ).first()

                            if existing_standing:
                                # Update existing record
                                existing_standing.position = int(standing_data_dict.get('position', 0))
                                existing_standing.games_played = int(standing_data_dict.get('games_played', 0))
                                existing_standing.points = int(standing_data_dict.get('points', 0))
                                existing_standing.wins = int(standing_data_dict.get('wins', 0))
                                existing_standing.draws = int(standing_data_dict.get('draws', 0))
                                existing_standing.losses = int(standing_data_dict.get('losses', 0))
                                existing_standing.goals_for = int(standing_data_dict.get('goals_for', 0))
                                existing_standing.goals_against = int(standing_data_dict.get('goals_against', 0))
                                existing_standing.fetched_at = datetime.utcnow() # Update timestamp
                                self.logger.debug(f"Updating existing standing for team {db_team.name} in league {db_league.name}")
                            else:
                                # Create new record
                                db_standing_entry = DBStanding(
                                    league_id=db_league.id,
                                    team_id=db_team.id,
                                    season_year=self.config.processing.season_year,
                                    position=int(standing_data_dict.get('position', 0)),
                                    games_played=int(standing_data_dict.get('games_played', 0)),
                                    points=int(standing_data_dict.get('points', 0)),
                                    wins=int(standing_data_dict.get('wins', 0)),
                                    draws=int(standing_data_dict.get('draws', 0)),
                                    losses=int(standing_data_dict.get('losses', 0)),
                                    goals_for=int(standing_data_dict.get('goals_for', 0)),
                                    goals_against=int(standing_data_dict.get('goals_against', 0))
                                    # fetched_at has default func.now()
                                )
                                self.db_session.add(db_standing_entry)
                                self.logger.debug(f"Adding new standing for team {db_team.name} in league {db_league.name}")

                            standings_processed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Error processing/saving standing data: {e}", exc_info=True)
                    break  # Use first stage with standings, common for most leagues
        
        if standings_processed_count > 0:
            try:
                self.db_session.commit()
                self.logger.info(f"Committed/Updated {standings_processed_count} standings for {league_name} to DB.")
            except Exception as e:
                self.db_session.rollback()
                self.logger.error(f"DB Error committing standings for {league_name}: {e}", exc_info=True)
                standings_processed_count = 0

        self.logger.info(f"Fetched and processed {standings_processed_count} standings for {league_name}")
        return standings_processed_count
    
    # Removed save_matches_to_csv and save_standings_to_csv as data goes to DB

    # BONUS: Summary report generation (can be adapted to read from DB or just count API calls)
    async def generate_summary_report(self):
        """Generate a summary report of data fetching activities."""
        # This will be simpler as we don't have individual files to count/size anymore.
        # It can report on number of leagues processed, items added/updated per type.
        # For now, let's keep it minimal or rely on logging.
        # A more advanced summary would query the DB for counts.
        self.logger.info("Summary report generation needs to be adapted for DB persistence.")
        # Example:
        # total_leagues_in_db = self.db_session.query(League).count()
        # total_matches_in_db = self.db_session.query(DBMatch).count()
        # total_standings_in_db = self.db_session.query(DBStanding).count()
        # self.logger.info(f"DB Summary: {total_leagues_in_db} leagues, {total_matches_in_db} matches, {total_standings_in_db} standings.")
        pass


    # 4. SEPARATE TASK METHODS for better parallelization
    async def fetch_historical_matches_task(self, league_api_id: int, league_name: str, semaphore: asyncio.Semaphore):
        """Separate task for historical matches, saves to DB."""
        async with semaphore:
            try:
                # Add recency check logic here if needed (e.g., don't refetch if data for league/season is very recent)
                # This would involve querying the DB for the latest match date for this league/season.
                # For simplicity, this example will always try to fetch and rely on unique constraints for inserts.
                
                self.logger.info(f"Starting historical matches task for {league_name} (API ID: {league_api_id})")
                start_time = time.time()
                
                count = await self.fetch_historical_matches(league_api_id, league_name) # Returns count of new matches added
                if count > 0:
                    self.logger.info(f"Historical matches task completed for {league_name} (API ID: {league_api_id}), added {count} new matches in {time.time() - start_time:.2f}s")
                else:
                    self.logger.info(f"No new historical matches added for {league_name} (API ID: {league_api_id}) in {time.time() - start_time:.2f}s")
                    
            except Exception as e:
                self.logger.error(f"Error in historical matches task for {league_name} (API ID: {league_api_id}): {e}", exc_info=True)


    async def fetch_standings_task(self, league_api_id: int, league_name: str, semaphore: asyncio.Semaphore):
        """Separate task for standings, saves to DB."""
        async with semaphore:
            try:
                # Add recency check for standings (e.g., don't refetch if fetched_at is recent)
                # db_league = self.db_session.query(League).filter_by(api_league_id=league_api_id).first()
                # if db_league:
                #    latest_standing = self.db_session.query(DBStanding.fetched_at).filter_by(league_id=db_league.id).order_by(DBStanding.fetched_at.desc()).first()
                #    if latest_standing and (datetime.utcnow() - latest_standing[0] < timedelta(hours=6)):
                #        self.logger.info(f"Recent standings data exists for {league_name}, skipping DB update.")
                #        return

                self.logger.info(f"Starting standings task for {league_name} (API ID: {league_api_id})")
                start_time = time.time()
                
                count = await self.fetch_standings(league_api_id, league_name) # Returns count of standings processed
                if count > 0:
                    self.logger.info(f"Standings task completed for {league_name} (API ID: {league_api_id}), processed {count} standings in {time.time() - start_time:.2f}s")
                else:
                    self.logger.info(f"No standings processed or updated for {league_name} (API ID: {league_api_id}) in {time.time() - start_time:.2f}s")
                    
            except Exception as e:
                self.logger.error(f"Error in standings task for {league_name} (API ID: {league_api_id}): {e}", exc_info=True)

    async def fetch_all_fixtures_task(self, all_leagues_from_api: Dict[int, str], semaphore: asyncio.Semaphore):
        """
        Fetches upcoming fixtures for all configured leagues.
        The current API for fixtures seems to return all leagues in one go, or can be filtered by league_id.
        This task will iterate configured leagues and call fetch_upcoming_fixtures_for_league.
        """
        async with semaphore: # This semaphore might be too broad if each call below is quick.
                              # Consider if semaphore should be inside the loop for finer control.
            self.logger.info("Starting task to fetch fixtures for all configured leagues.")
            fixture_tasks = []
            for league_api_id in self.config.processing.league_ids:
                league_name = all_leagues_from_api.get(league_api_id, f"LeagueAPI-{league_api_id}")
                # Create a sub-task for each league's fixture fetch to run them concurrently if semaphore allows
                # This reuses the semaphore passed in, which might not be what we want if this task itself is one of many.
                # For now, let's run them sequentially within this task to avoid over-complicating semaphore logic.
                # A better approach for Prefect would be to make fetch_upcoming_fixtures_for_league a task itself.
                try:
                    self.logger.info(f"Fetching fixtures for {league_name} (API ID: {league_api_id}) as part of all_fixtures_task")
                    start_time_league = time.time()
                    count = await self.fetch_upcoming_fixtures_for_league(league_api_id, league_name)
                    if count > 0:
                         self.logger.info(f"Fixtures sub-task for {league_name} added {count} new fixtures in {time.time() - start_time_league:.2f}s")
                    else:
                        self.logger.info(f"No new fixtures added for {league_name} in {time.time() - start_time_league:.2f}s")
                except Exception as e:
                    self.logger.error(f"Error fetching fixtures for {league_name} (API ID: {league_api_id}) in all_fixtures_task: {e}", exc_info=True)

            self.logger.info("Completed fetching fixtures for all configured leagues.")


    async def run(self):
        """Main execution method with parallel processing."""
        self.logger.info("Starting enhanced data fetch process using database.")
        
        try:
            self.cache.clear_expired()
            
            # max_concurrent controls parallel API calls via semaphore
            max_concurrent = self.config.processing.max_concurrent_requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            self.logger.info("Fetching all available leagues from API and populating/updating DB...")
            # fetch_leagues now also stores them in the DB.
            # all_leagues_api_map: Dict[api_league_id, league_name]
            all_leagues_api_map = await self.fetch_leagues()
            self.logger.info(f"Fetched/updated {len(all_leagues_api_map)} leagues in DB based on API.")
            
            all_tasks = []
            
            # Iterate through configured league_ids to schedule tasks
            for league_api_id_to_process in self.config.processing.league_ids:
                league_name = all_leagues_api_map.get(league_api_id_to_process)
                if not league_name:
                    # This case should ideally be handled by ensuring _get_or_create_league runs first
                    # or by fetching details for unknown league_ids if necessary.
                    # For now, we'll log and potentially create a placeholder name.
                    self.logger.warning(f"League API ID {league_api_id_to_process} from config not found in initial API league scan. Attempting to proceed.")
                    league_name = f"LeagueAPI-{league_api_id_to_process}"
                    # Ensure this league is in the DB if it's in config
                    self._get_or_create_league(api_league_id=league_api_id_to_process, league_name=league_name)


                self.logger.info(f"Scheduling data fetch tasks for {league_name} (API ID: {league_api_id_to_process})")
                
                all_tasks.extend([
                    self.fetch_historical_matches_task(league_api_id_to_process, league_name, semaphore),
                    self.fetch_standings_task(league_api_id_to_process, league_name, semaphore),
                    # fetch_fixtures_task is now part of fetch_all_fixtures_task logic
                ])
            
            # Add the task for fetching all fixtures
            # This task iterates internally, consider if it should be broken down further
            # if individual league fixture calls are made.
            # If API supports bulk fixture fetch well, this is fine.
            # The current fetch_upcoming_fixtures_for_league is per league, so fetch_all_fixtures_task iterates.
            all_tasks.append(self.fetch_all_fixtures_task(all_leagues_api_map, semaphore))
            
            self.logger.info(f"Executing {len(all_tasks)} main tasks/sub-flows in parallel with max {max_concurrent} concurrent API operations")
            
            start_time = time.time()
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful_tasks = 0
            failed_tasks = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Detailed logging for the specific task that failed would be inside the task method itself.
                    self.logger.error(f"A main task group (e.g., historical for one league, or all_fixtures) encountered an error: {result}", exc_info=True)
                    failed_tasks += 1
                else:
                    successful_tasks += 1
            
            self.logger.info(f"Data fetch process completed in {total_time:.2f}s")
            self.logger.info(f"Task group results: {successful_tasks} successful, {failed_tasks} failed.")
            
            await self.generate_summary_report() # Needs update for DB
            
        except Exception as e:
            self.logger.error(f"Critical error in main run loop: {e}", exc_info=True)
            raise
        finally:
            self.db_session.close() # Ensure DB session is closed


# Main execution
from prefect import task, flow, get_run_logger

@task(name="Run Data Fetcher")
async def run_data_fetcher_task(db_session_override: Optional[Session] = None):
    """Prefect task to run the DataFetcher."""
    logger = get_run_logger()
    if not app_config:
        logger.error("DataFetcher: Global app_config not loaded. Cannot proceed.")
        raise ValueError("Global app_config not loaded.")

    # Determine DB session: use override if provided (for testing), else create new.
    # The flow should ideally manage the session's lifecycle.
    # If this task is run standalone or if db_session_override is None, it manages its own session.
    session_managed_locally = False
    if db_session_override:
        db_sess = db_session_override
    else:
        db_sess = SessionLocal()
        session_managed_locally = True
        logger.info("DataFetcher task created its own DB session.")

    fetcher = None # Ensure fetcher is defined for finally block
    try:
        fetcher_app_config = app_config.data_fetcher
        fetcher = EnhancedDataFetcher(fetcher_config=fetcher_app_config, db_session=db_sess)
        await fetcher.run()
        logger.info("DataFetcher task completed successfully.")
    except FileNotFoundError as e: # This might be redundant if app_config load fails earlier
        logger.error(f"DataFetcher: Configuration file not found during task execution (should have been caught at import): {e}")
        raise
    except ValidationError as e:
        logging.error(f"DataFetcher: Configuration validation error: {e}")
        raise
    except Exception as e:
        logging.error(f"DataFetcher: Unexpected error during execution: {e}", exc_info=True)
        raise
    finally:
        # Ensure fetcher is defined (it would be if try block started)
        # The session passed to EnhancedDataFetcher (db_sess) should be closed here
        # ONLY if it was created locally within this task.
        if session_managed_locally:
            db_sess.close()
            logger.info("DataFetcher task closed its locally managed DB session.")


if __name__ == "__main__":
    # Example of how to run the task directly (for testing, not part of a flow yet)
    # This will be part of a larger Prefect flow defined in pipeline.py

    # To test this task directly, you'd need to set up a DB session:
    # async def test_task():
    #     db_session_instance = SessionLocal()
    #     try:
    #         await run_data_fetcher_task(db_session_override=db_session_instance) # Hypothetical override for testing
    #     finally:
    #         db_session_instance.close()
    # asyncio.run(test_task())
    pass
