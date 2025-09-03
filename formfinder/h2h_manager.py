#!/usr/bin/env python3
"""
H2H Statistics Manager

Manages head-to-head statistics computation and caching for the FormFinder system.
This component is part of the Enhanced Data Collection Layer as specified in the PRD.
Integrates with the H2H collection service and fallback system for optimal data management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from .clients.api_client import SoccerDataAPIClient


class H2HManager:
    """Manages H2H statistics computation and caching with fallback integration."""
    
    def __init__(self, db_session: Session, api_client: SoccerDataAPIClient, 
                 fallback_system=None, collection_service=None):
        self.db_session = db_session
        self.api_client = api_client
        self.fallback_system = fallback_system
        self.collection_service = collection_service
        self.cache_ttl = timedelta(hours=24)
        self.logger = logging.getLogger(__name__)
    
    async def get_or_compute_h2h(self, team1_id: int, team2_id: int, 
                                competition_id: int, use_fallback: bool = True) -> Dict[str, Any]:
        """Get H2H stats from cache or compute via API with fallback support."""
        try:
            # Check if we have cached H2H data
            cached_h2h = self._get_cached_h2h(team1_id, team2_id, competition_id)
            if cached_h2h and self._is_cache_valid(cached_h2h['updated_at']):
                self.logger.debug(f"Using cached H2H for teams {team1_id} vs {team2_id}")
                return cached_h2h
            
            # Use fallback system if available and enabled
            if use_fallback and self.fallback_system:
                self.logger.info(f"Using fallback system for H2H: {team1_id} vs {team2_id}")
                h2h_data = await self.fallback_system.get_h2h_with_fallback(
                    team1_id, team2_id, competition_id
                )
                if h2h_data and h2h_data.get('source') != 'default':
                    return h2h_data
            
            # Compute fresh H2H data via API
            self.logger.info(f"Computing fresh H2H for teams {team1_id} vs {team2_id}")
            h2h_data = await self._compute_h2h_via_api(team1_id, team2_id, competition_id)
            
            # Cache the computed data
            self._cache_h2h_data(team1_id, team2_id, competition_id, h2h_data)
            
            return h2h_data
            
        except Exception as e:
            self.logger.error(f"Error getting H2H for teams {team1_id} vs {team2_id}: {e}")
            # Return default values if computation fails
            return self._get_default_h2h()
    
    async def batch_compute_h2h(self, team_pairs: List[Tuple[int, int]], 
                               competition_id: int) -> Dict[str, int]:
        """Batch compute H2H statistics for multiple team pairs."""
        results = {
            'computed': 0,
            'cached': 0,
            'failed': 0
        }
        
        for team1_id, team2_id in team_pairs:
            try:
                h2h_data = await self.get_or_compute_h2h(team1_id, team2_id, competition_id)
                if h2h_data.get('source') == 'cache':
                    results['cached'] += 1
                else:
                    results['computed'] += 1
            except Exception as e:
                self.logger.error(f"Failed to compute H2H for {team1_id} vs {team2_id}: {e}")
                results['failed'] += 1
        
        self.logger.info(f"Batch H2H computation complete: {results}")
        return results
    
    def _get_cached_h2h(self, team1_id: int, team2_id: int, competition_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached H2H data from database."""
        try:
            query = text("""
                SELECT team1_id, team2_id, competition_id, 
                       overall_games_played, overall_team1_wins, overall_team2_wins, overall_draws,
                       overall_team1_scored, overall_team2_scored,
                       team1_games_played_at_home, team1_wins_at_home, team1_losses_at_home, team1_draws_at_home,
                       team1_scored_at_home, team1_conceded_at_home,
                       team2_games_played_at_home, team2_wins_at_home, team2_losses_at_home, team2_draws_at_home,
                       team2_scored_at_home, team2_conceded_at_home,
                       avg_total_goals, last_fetched_at
                FROM h2h_cache 
                WHERE ((team1_id = :team1_id AND team2_id = :team2_id) OR 
                       (team1_id = :team2_id AND team2_id = :team1_id))
                AND (competition_id = :competition_id OR (:competition_id IS NULL AND competition_id IS NULL))
                ORDER BY last_fetched_at DESC
                LIMIT 1
            """)
            
            result = self.db_session.execute(query, {
                'team1_id': team1_id,
                'team2_id': team2_id,
                'competition_id': competition_id
            }).fetchone()
            
            if result:
                return {
                    'team1_id': result.team1_id,
                    'team2_id': result.team2_id,
                    'competition_id': result.competition_id,
                    'overall_games_played': result.overall_games_played,
                    'overall_team1_wins': result.overall_team1_wins,
                    'overall_team2_wins': result.overall_team2_wins,
                    'overall_draws': result.overall_draws,
                    'overall_team1_scored': result.overall_team1_scored,
                    'overall_team2_scored': result.overall_team2_scored,
                    'team1_games_played_at_home': result.team1_games_played_at_home,
                    'team1_wins_at_home': result.team1_wins_at_home,
                    'team1_losses_at_home': result.team1_losses_at_home,
                    'team1_draws_at_home': result.team1_draws_at_home,
                    'team1_scored_at_home': result.team1_scored_at_home,
                    'team1_conceded_at_home': result.team1_conceded_at_home,
                    'team2_games_played_at_home': result.team2_games_played_at_home,
                    'team2_wins_at_home': result.team2_wins_at_home,
                    'team2_losses_at_home': result.team2_losses_at_home,
                    'team2_draws_at_home': result.team2_draws_at_home,
                    'team2_scored_at_home': result.team2_scored_at_home,
                    'team2_conceded_at_home': result.team2_conceded_at_home,
                    'avg_total_goals': float(result.avg_total_goals) if result.avg_total_goals else 0.0,
                    'updated_at': result.last_fetched_at,
                    'source': 'cache'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached H2H: {e}")
            return None
    
    def _is_cache_valid(self, updated_at: datetime) -> bool:
        """Check if cached data is still valid."""
        return datetime.now() - updated_at < self.cache_ttl
    
    async def _compute_h2h_via_api(self, team1_id: int, team2_id: int, competition_id: int) -> Dict[str, Any]:
        """Compute H2H statistics via API call using the new API format."""
        try:
            # Get H2H data from API (now returns flattened dict directly)
            h2h_data = self.api_client.get_h2h_stats(team1_id, team2_id, competition_id)
            
            if not h2h_data:
                return self._get_default_h2h()
            
            # The API client now returns flattened data that matches our cache structure
            # Just add metadata and return
            h2h_data.update({
                'updated_at': datetime.now(),
                'source': 'api'
            })
            
            return h2h_data
            
        except Exception as e:
            self.logger.error(f"Error computing H2H via API: {e}")
            return self._get_default_h2h()
    
    def _cache_h2h_data(self, team1_id: int, team2_id: int, competition_id: int, h2h_data: Dict[str, Any]):
        """Cache H2H data in database."""
        try:
            # The API client already handles caching via UPSERT, so this method is now simplified
            # We just log the caching action since the actual caching is done in the API client
            self.logger.debug(f"H2H data cached for teams {team1_id} vs {team2_id} via API client")
            
        except Exception as e:
            self.logger.error(f"Error in H2H cache logging: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN last_fetched_at > :recent_threshold THEN 1 END) as recent_entries,
                    AVG(overall_games_played) as avg_games_played,
                    MIN(last_fetched_at) as oldest_entry,
                    MAX(last_fetched_at) as newest_entry
                FROM h2h_cache
            """)
            
            recent_threshold = datetime.now() - self.cache_ttl
            result = self.db_session.execute(query, {
                'recent_threshold': recent_threshold
            }).fetchone()
            
            if result:
                return {
                    'total_entries': result.total_entries,
                    'recent_entries': result.recent_entries,
                    'cache_hit_potential': (result.recent_entries / max(result.total_entries, 1)) * 100,
                    'avg_games_played': float(result.avg_games_played) if result.avg_games_played else 0.0,
                    'oldest_entry': result.oldest_entry,
                    'newest_entry': result.newest_entry,
                    'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600
                }
            
            return {'total_entries': 0, 'recent_entries': 0, 'cache_hit_potential': 0.0}
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def queue_h2h_collection(self, team1_id: int, team2_id: int, competition_id: int, 
                           priority: str = 'medium') -> bool:
        """Queue H2H data collection via the collection service."""
        if not self.collection_service:
            self.logger.warning("Collection service not available for queuing")
            return False
        
        try:
            success = self.collection_service.queue_h2h_request(
                team1_id, team2_id, competition_id, priority
            )
            if success:
                self.logger.info(f"Queued H2H collection for {team1_id} vs {team2_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error queuing H2H collection: {e}")
            return False
    
    def invalidate_cache(self, team1_id: int = None, team2_id: int = None, 
                        competition_id: int = None) -> int:
        """Invalidate cache entries based on criteria."""
        try:
            conditions = []
            params = {}
            
            if team1_id is not None:
                conditions.append("(team1_id = :team1_id OR team2_id = :team1_id)")
                params['team1_id'] = team1_id
            
            if team2_id is not None:
                conditions.append("(team1_id = :team2_id OR team2_id = :team2_id)")
                params['team2_id'] = team2_id
            
            if competition_id is not None:
                conditions.append("competition_id = :competition_id")
                params['competition_id'] = competition_id
            
            if not conditions:
                # Invalidate all old entries
                conditions.append("last_fetched_at < :old_threshold")
                params['old_threshold'] = datetime.now() - self.cache_ttl
            
            where_clause = " AND ".join(conditions)
            query = text(f"DELETE FROM h2h_cache WHERE {where_clause}")
            
            result = self.db_session.execute(query, params)
            self.db_session.commit()
            
            deleted_count = result.rowcount
            self.logger.info(f"Invalidated {deleted_count} cache entries")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
            self.db_session.rollback()
            return 0
    

    
    def _get_default_h2h(self) -> Dict[str, Any]:
        """Return default H2H values when computation fails."""
        return {
            'team1_id': 0,
            'team2_id': 0,
            'competition_id': 0,
            'overall_games_played': 0,
            'overall_team1_wins': 0,
            'overall_team2_wins': 0,
            'overall_draws': 0,
            'overall_team1_scored': 0,
            'overall_team2_scored': 0,
            'team1_games_played_at_home': 0,
            'team1_wins_at_home': 0,
            'team1_losses_at_home': 0,
            'team1_draws_at_home': 0,
            'team1_scored_at_home': 0,
            'team1_conceded_at_home': 0,
            'team2_games_played_at_home': 0,
            'team2_wins_at_home': 0,
            'team2_losses_at_home': 0,
            'team2_draws_at_home': 0,
            'team2_scored_at_home': 0,
            'team2_conceded_at_home': 0,
            'avg_total_goals': 2.5,  # League average fallback
            'updated_at': datetime.now(),
            'source': 'default'
        }