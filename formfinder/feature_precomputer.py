"""Feature Pre-computation Engine

This module implements the core feature pre-computation functionality
to eliminate API dependency during model training.

Author: FormFinder2 Team
Created: 2025-01-01
Purpose: Pre-compute and cache all features for training pipeline
"""


import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_


from .database import get_db_session
from .config import FeatureComputationConfig
from .exceptions import FeatureComputationError
from .db_h2h_computer import DatabaseH2HComputer
from .enhanced_feature_computer import EnhancedFeatureComputer


class FeaturePrecomputer:
    """Pre-computes and caches all features for training."""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.db_h2h_computer = DatabaseH2HComputer(db_session)
        self.enhanced_computer = EnhancedFeatureComputer(db_session)
        self.logger = logging.getLogger(__name__)
        self.config = FeatureComputationConfig()
        
        # Performance tracking
        self.stats = {
            'total_fixtures': 0,
            'successful_computations': 0,
            'failed_computations': 0,
            'api_calls_used': 0,
            'cache_hits': 0,
            'start_time': None,
            'end_time': None
        }
    
    def get_cached_features(self, fixture_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve cached features for given fixture IDs.
        
        Args:
            fixture_ids: List of fixture IDs to retrieve features for
            
        Returns:
            List of feature dictionaries for fixtures with cached features
        """
        if not fixture_ids:
            return []
        
        # Query for cached features
        placeholders = ','.join([':id' + str(i) for i in range(len(fixture_ids))])
        query = text(f"""
            SELECT fixture_id, features_json
            FROM pre_computed_features
            WHERE fixture_id IN ({placeholders})
                AND features_computed_at > NOW() - INTERVAL '24 hours'
        """)
        
        params = {f'id{i}': fixture_id for i, fixture_id in enumerate(fixture_ids)}
        
        try:
            results = self.db_session.execute(query, params).fetchall()
            
            cached_features = []
            for row in results:
                try:
                    features = json.loads(row.features_json)
                    features['fixture_id'] = row.fixture_id
                    cached_features.append(features)
                except (json.JSONDecodeError, AttributeError) as e:
                    self.logger.warning(f"Failed to parse cached features for fixture {row.fixture_id}: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(cached_features)} cached feature sets from {len(fixture_ids)} requested")
            return cached_features
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached features: {e}")
            return []
    
    def compute_all_features(self, fixture_ids: List[int], 
                                 force_refresh: bool = False) -> Dict[str, int]:
        """Compute all features for given fixtures.
        
        Args:
            fixture_ids: List of fixture IDs to process
            force_refresh: Whether to force refresh of cached features
            
        Returns:
            Dictionary with computation statistics
        """
        self.stats['start_time'] = time.time()
        self.stats['total_fixtures'] = len(fixture_ids)
        
        self.logger.info(f"Starting feature computation for {len(fixture_ids)} fixtures")
        
        # Process fixtures in batches
        batch_size = self.config.feature_batch_size
        
        batch_results = []
        for i in range(0, len(fixture_ids), batch_size):
            batch = fixture_ids[i:i + batch_size]
            try:
                result = self._process_fixture_batch(batch, force_refresh)
                batch_results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                batch_results.append(e)
        
        # Aggregate results
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing failed: {result}")
                self.stats['failed_computations'] += 1
            else:
                self.stats['successful_computations'] += result.get('successful', 0)
                self.stats['failed_computations'] += result.get('failed', 0)
                # No API calls used anymore
                self.stats['cache_hits'] += result.get('cache_hits', 0)
        
        self.stats['end_time'] = time.time()
        
        # Log final statistics
        self._log_computation_stats()
        
        # Update data quality metrics
        self._update_data_quality_metrics()
        
        return self.stats
    
    def _process_fixture_batch(self, fixture_ids: List[int], 
                                   force_refresh: bool) -> Dict[str, int]:
        """Process a batch of fixtures."""
        batch_stats = {'successful': 0, 'failed': 0, 'cache_hits': 0}
        
        for fixture_id in fixture_ids:
            try:
                result = self._compute_fixture_features(fixture_id, force_refresh)
                if result['success']:
                    batch_stats['successful'] += 1
                else:
                    batch_stats['failed'] += 1
                
                batch_stats['cache_hits'] += result.get('cache_hits', 0)
                
            except Exception as e:
                self.logger.error(f"Failed to compute features for fixture {fixture_id}: {e}")
                batch_stats['failed'] += 1
                
                # Log the failure
                self._log_computation_failure(fixture_id, 'all', str(e))
        
        return batch_stats
    
    def _compute_fixture_features(self, fixture_id: int, 
                                      force_refresh: bool) -> Dict[str, Any]:
        """Compute all features for a single fixture."""
        start_time = time.time()
        result = {
            'success': False,
            'api_calls': 0,
            'cache_hits': 0,
            'features_computed': 0
        }
        
        try:
            # Get fixture details
            fixture = self._get_fixture_details(fixture_id)
            if not fixture:
                raise FeatureComputationError(f"Fixture {fixture_id} not found")
            
            # Check if features already exist and are fresh
            if not force_refresh and self._features_exist_and_fresh(fixture_id):
                self.logger.debug(f"Features for fixture {fixture_id} are already fresh")
                result['cache_hits'] += 1
                result['success'] = True
                return result
            
            # Compute all 87 features using enhanced computer
            all_features = self.enhanced_computer.compute_all_features(fixture_id)
            
            # Aggregate results
            total_api_calls = 0  # No API calls needed anymore
            features_computed = len(all_features)
            
            # Create result structure for compatibility
            form_result = {'success': True, 'features_computed': features_computed}
            h2h_result = {'success': True, 'features_computed': 0}
            preview_result = {'success': True, 'features_computed': 0}
            
            # Calculate data quality score
            quality_score = self._calculate_quality_score(
                form_result, h2h_result, preview_result
            )
            
            # Store the computed features
            self._store_computed_features(
                fixture_id, fixture, all_features, quality_score
            )
            
            result.update({
                'success': True,
                'api_calls': total_api_calls,
                'features_computed': features_computed,
                'quality_score': quality_score
            })
            
            # Log successful computation
            computation_time = int((time.time() - start_time) * 1000)
            self._log_computation_success(
                fixture_id, 'all', computation_time, total_api_calls, 
                features_computed, quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Feature computation failed for fixture {fixture_id}: {e}")
            computation_time = int((time.time() - start_time) * 1000)
            self._log_computation_failure(fixture_id, 'all', str(e), computation_time)
            raise
        
        return result
    
    def compute_form_features(self, fixture_id: int) -> Dict[str, Any]:
        """Compute rolling form features for both teams."""
        start_time = time.time()
        
        try:
            fixture = self._get_fixture_details(fixture_id)
            if not fixture:
                raise FeatureComputationError(f"Fixture {fixture_id} not found")
            
            home_team_id = fixture['home_team_id']
            away_team_id = fixture['away_team_id']
            match_date = fixture['match_date']
            league_id = fixture['league_id']
            
            # Get form features for both teams
            home_form = self._get_team_form_features(
                home_team_id, match_date, league_id, venue='home'
            )
            away_form = self._get_team_form_features(
                away_team_id, match_date, league_id, venue='away'
            )
            
            result = {
                'success': True,
                'home_form': home_form,
                'away_form': away_form,
                'api_calls': 0,  # Form features use database only
                'features_computed': len(home_form) + len(away_form),
                'computation_time_ms': int((time.time() - start_time) * 1000)
            }
            
            self.logger.debug(f"Form features computed for fixture {fixture_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Form feature computation failed for fixture {fixture_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'api_calls': 0,
                'features_computed': 0,
                'computation_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def compute_h2h_features(self, fixture_id: int) -> Dict[str, Any]:
        """Compute and cache H2H statistics from database."""
        start_time = time.time()
        
        try:
            fixture = self._get_fixture_details(fixture_id)
            if not fixture:
                raise FeatureComputationError(f"Fixture {fixture_id} not found")
            
            home_team_id = fixture['home_team_id']
            away_team_id = fixture['away_team_id']
            league_id = fixture['league_id']
            
            # Check cache first
            cached_h2h = self.db_h2h_computer.get_cached_h2h_stats(home_team_id, away_team_id, league_id)
            
            if cached_h2h:
                self.logger.debug(f"Using cached H2H for teams {home_team_id} vs {away_team_id}")
                h2h_features = self._extract_h2h_features(cached_h2h)
            else:
                # Compute from database
                self.logger.debug(f"Computing H2H from database for teams {home_team_id} vs {away_team_id}")
                h2h_data = self.db_h2h_computer.compute_h2h_stats(
                    home_team_id, away_team_id, league_id
                )
                
                # Cache the result
                self.db_h2h_computer.cache_h2h_stats(home_team_id, away_team_id, league_id, h2h_data)
                h2h_features = self._extract_h2h_features(h2h_data)
            
            result = {
                'success': True,
                'h2h_features': h2h_features,
                'features_computed': len(h2h_features),
                'computation_time_ms': int((time.time() - start_time) * 1000)
            }
            
            self.logger.debug(f"H2H features computed for fixture {fixture_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"H2H feature computation failed for fixture {fixture_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'api_calls': 0,
                'features_computed': 0,
                'computation_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def compute_preview_features(self, fixture_id: int) -> Dict[str, Any]:
        """Compute preview and weather features from database."""
        start_time = time.time()
        
        try:
            fixture = self._get_fixture_details(fixture_id)
            if not fixture:
                raise FeatureComputationError(f"Fixture {fixture_id} not found")
            
            # Get weather data from database
            weather_features = self._get_weather_features_from_db(fixture_id)
            
            # Combine preview and weather features
            preview_features = {
                'excitement_rating': 0.0,  # Default for now
                **weather_features
            }
            
            result = {
                'success': True,
                'preview_features': preview_features,
                'features_computed': len(preview_features),
                'computation_time_ms': int((time.time() - start_time) * 1000)
            }
            
            self.logger.debug(f"Preview features computed for fixture {fixture_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Preview feature computation failed for fixture {fixture_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'features_computed': 0,
                'computation_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def _get_fixture_details(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Get fixture details from database."""
        query = text("""
            SELECT id, home_team_id, away_team_id, match_date, league_id, status
            FROM fixtures 
            WHERE id = :fixture_id
        """)
        
        result = self.db_session.execute(query, {'fixture_id': fixture_id}).fetchone()
        if result:
            return {
                'id': result[0],
                'home_team_id': result[1],
                'away_team_id': result[2],
                'match_date': result[3],
                'league_id': result[4],
                'status': result[5]
            }
        return None
    
    def _features_exist_and_fresh(self, fixture_id: int) -> bool:
        """Check if features exist and are fresh."""
        query = text("""
            SELECT features_computed_at, data_quality_score
            FROM pre_computed_features 
            WHERE fixture_id = :fixture_id
        """)
        
        result = self.db_session.execute(query, {'fixture_id': fixture_id}).fetchone()
        if not result:
            return False
        
        computed_at, quality_score = result
        
        # Check if features are fresh (computed within last 24 hours)
        if computed_at < datetime.now() - timedelta(hours=24):
            return False
        
        # Check if quality score is acceptable
        if quality_score < self.config.MIN_FEATURE_COMPLETENESS:
            return False
        
        return True
    
    def _get_team_form_features(self, team_id: int, match_date: datetime, 
                              league_id: int, venue: str = 'all') -> Dict[str, Any]:
        """Get team form features from recent matches."""
        # Get last 5 matches for the team before the given date
        venue_filter = ""
        if venue == 'home':
            venue_filter = "AND f.home_team_id = :team_id"
        elif venue == 'away':
            venue_filter = "AND f.away_team_id = :team_id"
        else:
            venue_filter = "AND (f.home_team_id = :team_id OR f.away_team_id = :team_id)"
        
        query = text(f"""
            SELECT f.home_team_id, f.away_team_id, f.home_score, f.away_score, f.match_date
            FROM fixtures f
            WHERE f.league_id = :league_id
                AND f.match_date < :match_date
                AND f.status = 'finished'
                AND f.home_score IS NOT NULL
                AND f.away_score IS NOT NULL
                {venue_filter}
            ORDER BY f.match_date DESC
            LIMIT :limit
        """)
        
        results = self.db_session.execute(query, {
            'team_id': team_id,
            'league_id': league_id,
            'match_date': match_date,
            'limit': self.config.form_lookback_games
        }).fetchall()
        
        if not results:
            return self._get_default_form_features()
        
        # Calculate form statistics
        wins = draws = losses = 0
        goals_for = goals_against = 0
        form_sequence = []
        
        for match in results:
            home_id, away_id, home_score, away_score, match_date = match
            
            if home_id == team_id:
                # Team played at home
                goals_for += home_score
                goals_against += away_score
                if home_score > away_score:
                    wins += 1
                    form_sequence.append('W')
                elif home_score == away_score:
                    draws += 1
                    form_sequence.append('D')
                else:
                    losses += 1
                    form_sequence.append('L')
            else:
                # Team played away
                goals_for += away_score
                goals_against += home_score
                if away_score > home_score:
                    wins += 1
                    form_sequence.append('W')
                elif away_score == home_score:
                    draws += 1
                    form_sequence.append('D')
                else:
                    losses += 1
                    form_sequence.append('L')
        
        games_played = len(results)
        avg_goals_scored = goals_for / games_played if games_played > 0 else 0.0
        avg_goals_conceded = goals_against / games_played if games_played > 0 else 0.0
        
        return {
            'avg_goals_scored': round(avg_goals_scored, 2),
            'avg_goals_conceded': round(avg_goals_conceded, 2),
            'wins_last_5': wins,
            'draws_last_5': draws,
            'losses_last_5': losses,
            'goals_for_last_5': goals_for,
            'goals_against_last_5': goals_against,
            'form_last_5_games': json.dumps(form_sequence)
        }
    
    def _get_default_form_features(self) -> Dict[str, Any]:
        """Get default form features when no data is available."""
        return {
            'avg_goals_scored': 0.0,
            'avg_goals_conceded': 0.0,
            'wins_last_5': 0,
            'draws_last_5': 0,
            'losses_last_5': 0,
            'goals_for_last_5': 0,
            'goals_against_last_5': 0,
            'form_last_5_games': json.dumps([])
        }
    
    def _get_cached_h2h(self, team1_id: int, team2_id: int, 
                       league_id: int) -> Optional[Dict[str, Any]]:
        """Get cached H2H data."""
        query = text("""
            SELECT * FROM h2h_cache_enhanced
            WHERE team1_id = :team1_id AND team2_id = :team2_id 
                AND competition_id = :league_id
        """)
        
        result = self.db_session.execute(query, {
            'team1_id': team1_id,
            'team2_id': team2_id,
            'league_id': league_id
        }).fetchone()
        
        if result:
            return dict(result._mapping)
        return None
    
    def _is_h2h_cache_fresh(self, cached_data: Dict[str, Any]) -> bool:
        """Check if H2H cache is still fresh."""
        if 'cache_expires_at' not in cached_data or not cached_data['cache_expires_at']:
            return False
        
        return cached_data['cache_expires_at'] > datetime.now()
    
    def _cache_h2h_data(self, team1_id: int, team2_id: int, 
                       league_id: int, h2h_data: Dict[str, Any]) -> None:
        """Cache H2H data in enhanced cache table."""
        expires_at = datetime.now() + timedelta(hours=self.config.H2H_CACHE_TTL_HOURS)
        
        # Extract and process H2H data
        processed_data = self._process_h2h_data(h2h_data)
        
        # Insert or update cache
        query = text("""
            INSERT INTO h2h_cache_enhanced (
                team1_id, team2_id, competition_id, total_games, team1_wins, team2_wins, draws,
                total_goals_scored, avg_total_goals, avg_team1_goals, avg_team2_goals,
                team1_home_games, team1_home_wins, team1_home_losses, team1_home_draws,
                team1_home_goals_for, team1_home_goals_against,
                team2_home_games, team2_home_wins, team2_home_losses, team2_home_draws,
                team2_home_goals_for, team2_home_goals_against,
                cache_expires_at, last_updated, data_source, api_calls_used
            ) VALUES (
                :team1_id, :team2_id, :competition_id, :total_games, :team1_wins, :team2_wins, :draws,
                :total_goals_scored, :avg_total_goals, :avg_team1_goals, :avg_team2_goals,
                :team1_home_games, :team1_home_wins, :team1_home_losses, :team1_home_draws,
                :team1_home_goals_for, :team1_home_goals_against,
                :team2_home_games, :team2_home_wins, :team2_home_losses, :team2_home_draws,
                :team2_home_goals_for, :team2_home_goals_against,
                :cache_expires_at, :last_updated, :data_source, :api_calls_used
            )
            ON CONFLICT (team1_id, team2_id, COALESCE(competition_id, 0))
            DO UPDATE SET
                total_games = EXCLUDED.total_games,
                team1_wins = EXCLUDED.team1_wins,
                team2_wins = EXCLUDED.team2_wins,
                draws = EXCLUDED.draws,
                total_goals_scored = EXCLUDED.total_goals_scored,
                avg_total_goals = EXCLUDED.avg_total_goals,
                avg_team1_goals = EXCLUDED.avg_team1_goals,
                avg_team2_goals = EXCLUDED.avg_team2_goals,
                team1_home_games = EXCLUDED.team1_home_games,
                team1_home_wins = EXCLUDED.team1_home_wins,
                team1_home_losses = EXCLUDED.team1_home_losses,
                team1_home_draws = EXCLUDED.team1_home_draws,
                team1_home_goals_for = EXCLUDED.team1_home_goals_for,
                team1_home_goals_against = EXCLUDED.team1_home_goals_against,
                team2_home_games = EXCLUDED.team2_home_games,
                team2_home_wins = EXCLUDED.team2_home_wins,
                team2_home_losses = EXCLUDED.team2_home_losses,
                team2_home_draws = EXCLUDED.team2_home_draws,
                team2_home_goals_for = EXCLUDED.team2_home_goals_for,
                team2_home_goals_against = EXCLUDED.team2_home_goals_against,
                cache_expires_at = EXCLUDED.cache_expires_at,
                last_updated = EXCLUDED.last_updated,
                api_calls_used = h2h_cache_enhanced.api_calls_used + 1
        """)
        
        # Calculate derived values
        total_games = processed_data.get('overall_games_played', 0)
        team1_scored = processed_data.get('overall_team1_scored', 0)
        team2_scored = processed_data.get('overall_team2_scored', 0)
        total_goals_scored = team1_scored + team2_scored
        avg_team1_goals = team1_scored / total_games if total_games > 0 else 0.0
        avg_team2_goals = team2_scored / total_games if total_games > 0 else 0.0
        
        self.db_session.execute(query, {
            'team1_id': team1_id,
            'team2_id': team2_id,
            'competition_id': league_id,
            'total_games': total_games,
            'team1_wins': processed_data.get('overall_team1_wins', 0),
            'team2_wins': processed_data.get('overall_team2_wins', 0),
            'draws': processed_data.get('overall_draws', 0),
            'total_goals_scored': total_goals_scored,
            'avg_total_goals': processed_data.get('avg_total_goals', 0.0),
            'avg_team1_goals': avg_team1_goals,
            'avg_team2_goals': avg_team2_goals,
            'team1_home_games': processed_data.get('team1_games_played_at_home', 0),
            'team1_home_wins': processed_data.get('team1_wins_at_home', 0),
            'team1_home_losses': processed_data.get('team1_losses_at_home', 0),
            'team1_home_draws': processed_data.get('team1_draws_at_home', 0),
            'team1_home_goals_for': processed_data.get('team1_scored_at_home', 0),
            'team1_home_goals_against': processed_data.get('team1_conceded_at_home', 0),
            'team2_home_games': processed_data.get('team2_games_played_at_home', 0),
            'team2_home_wins': processed_data.get('team2_wins_at_home', 0),
            'team2_home_losses': processed_data.get('team2_losses_at_home', 0),
            'team2_home_draws': processed_data.get('team2_draws_at_home', 0),
            'team2_home_goals_for': processed_data.get('team2_scored_at_home', 0),
            'team2_home_goals_against': processed_data.get('team2_conceded_at_home', 0),
            'cache_expires_at': expires_at,
            'last_updated': datetime.now(),
            'data_source': 'api',
            'api_calls_used': 1
        })
        
        self.db_session.commit()
    
    def _process_h2h_data(self, h2h_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw H2H data into standardized format."""
        # Process the new API response format with all fields
        return {
            'overall_games_played': h2h_data.get('overall_games_played', 0),
            'overall_team1_wins': h2h_data.get('overall_team1_wins', 0),
            'overall_team2_wins': h2h_data.get('overall_team2_wins', 0),
            'overall_draws': h2h_data.get('overall_draws', 0),
            'overall_team1_scored': h2h_data.get('overall_team1_scored', 0),
            'overall_team2_scored': h2h_data.get('overall_team2_scored', 0),
            'avg_total_goals': h2h_data.get('avg_total_goals', 0.0),
            'team1_games_played_at_home': h2h_data.get('team1_games_played_at_home', 0),
            'team1_wins_at_home': h2h_data.get('team1_wins_at_home', 0),
            'team1_losses_at_home': h2h_data.get('team1_losses_at_home', 0),
            'team1_draws_at_home': h2h_data.get('team1_draws_at_home', 0),
            'team1_scored_at_home': h2h_data.get('team1_scored_at_home', 0),
            'team1_conceded_at_home': h2h_data.get('team1_conceded_at_home', 0),
            'team2_games_played_at_home': h2h_data.get('team2_games_played_at_home', 0),
            'team2_wins_at_home': h2h_data.get('team2_wins_at_home', 0),
            'team2_losses_at_home': h2h_data.get('team2_losses_at_home', 0),
            'team2_draws_at_home': h2h_data.get('team2_draws_at_home', 0),
            'team2_scored_at_home': h2h_data.get('team2_scored_at_home', 0),
            'team2_conceded_at_home': h2h_data.get('team2_conceded_at_home', 0)
        }
    
    def _extract_h2h_features(self, h2h_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract H2H features for training."""
        return {
            # Core H2H features expected by EnhancedGoalPredictor
            'h2h_total_matches': h2h_data.get('overall_games_played', 0),
            'h2h_home_wins': h2h_data.get('overall_team1_wins', 0),
            'h2h_away_wins': h2h_data.get('overall_team2_wins', 0),
            'h2h_draws': h2h_data.get('overall_draws', 0),
            'h2h_avg_goals': h2h_data.get('avg_total_goals', 0.0),
            'h2h_avg_home_goals': h2h_data.get('overall_team1_scored', 0) / max(h2h_data.get('overall_games_played', 1), 1),
            'h2h_avg_away_goals': h2h_data.get('overall_team2_scored', 0) / max(h2h_data.get('overall_games_played', 1), 1),
            # Legacy feature names for backward compatibility
            'h2h_overall_games': h2h_data.get('overall_games_played', 0),
            'h2h_avg_total_goals': h2h_data.get('avg_total_goals', 0.0),
            'h2h_overall_home_goals': h2h_data.get('overall_team1_scored', 0) / max(h2h_data.get('overall_games_played', 1), 1),
            'h2h_overall_away_goals': h2h_data.get('overall_team2_scored', 0) / max(h2h_data.get('overall_games_played', 1), 1),
            'h2h_team1_wins': h2h_data.get('overall_team1_wins', 0),
            'h2h_team2_wins': h2h_data.get('overall_team2_wins', 0),
            'h2h_home_advantage': self._calculate_home_advantage(h2h_data),
            # Enhanced features from new API
            'h2h_team1_home_win_rate': self._calculate_home_win_rate(h2h_data, 'team1'),
            'h2h_team2_home_win_rate': self._calculate_home_win_rate(h2h_data, 'team2'),
            'h2h_team1_home_goals_avg': h2h_data.get('team1_scored_at_home', 0) / max(h2h_data.get('team1_games_played_at_home', 1), 1),
            'h2h_team2_home_goals_avg': h2h_data.get('team2_scored_at_home', 0) / max(h2h_data.get('team2_games_played_at_home', 1), 1)
        }
    
    def _calculate_home_advantage(self, h2h_data: Dict[str, Any]) -> float:
        """Calculate home advantage from H2H data."""
        total_games = h2h_data.get('overall_games_played', 0)
        if total_games == 0:
            return 0.0
        
        team1_wins = h2h_data.get('overall_team1_wins', 0)
        return round((team1_wins / total_games) - 0.5, 3)  # Advantage over 50%
    
    def _calculate_home_win_rate(self, h2h_data: Dict[str, Any], team: str) -> float:
        """Calculate home win rate for a specific team from H2H data."""
        if team == 'team1':
            home_games = h2h_data.get('team1_games_played_at_home', 0)
            home_wins = h2h_data.get('team1_wins_at_home', 0)
        else:  # team2
            home_games = h2h_data.get('team2_games_played_at_home', 0)
            home_wins = h2h_data.get('team2_wins_at_home', 0)
        
        if home_games == 0:
            return 0.0
        
        return round(home_wins / home_games, 3)
    
    def _extract_preview_features(self, preview_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract preview features from API data."""
        return {
            'excitement_rating': preview_data.get('excitement_rating', 0.0)
        }
    
    def _extract_weather_features(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract weather features from API data."""
        return {
            'weather_temp_c': weather_data.get('temperature_c', 21.0),
            'weather_temp_f': weather_data.get('temperature_f', 69.8),
            'weather_humidity': weather_data.get('humidity', 50.0),
            'weather_wind_speed': weather_data.get('wind_speed', 5.0),
            'weather_precipitation': weather_data.get('precipitation', 0.0),
            'weather_condition': weather_data.get('condition', 'Clear')
        }
    
    def _get_weather_features_from_db(self, fixture_id: int) -> Dict[str, Any]:
        """Get weather features from database WeatherData model. 
        
        Args:
            fixture_id: ID of the fixture to get weather data for
            
        Returns:
            Dictionary containing weather features with defaults if no data found
        """
        try:
            # Query weather data from database
            query = text("""
                SELECT 
                    temperature_2m,
                    relative_humidity_2m,
                    wind_speed_10m,
                    precipitation,
                    cloud_cover,
                    visibility,
                    weather_code
                FROM weather_data 
                WHERE fixture_id = :fixture_id
                ORDER BY weather_datetime DESC
                LIMIT 1
            """)
            
            result = self.db_session.execute(query, {'fixture_id': fixture_id}).fetchone()
            
            if result:
                temp_c = result[0] if result[0] is not None else 21.0
                temp_f = (temp_c * 9/5) + 32 if temp_c is not None else 69.8
                
                return {
                    'weather_temp_c': round(temp_c, 1),
                    'weather_temp_f': round(temp_f, 1),
                    'weather_humidity': result[1] if result[1] is not None else 50.0,
                    'weather_wind_speed': result[2] if result[2] is not None else 5.0,
                    'weather_precipitation': result[3] if result[3] is not None else 0.0,
                    'weather_condition': result[6] if result[6] is not None else 'Clear'
                }
            else:
                # Return default values if no weather data found
                self.logger.debug(f"No weather data found for fixture {fixture_id}, using defaults")
                return {
                    'weather_temp_c': 21.0,
                    'weather_temp_f': 69.8,
                    'weather_humidity': 50.0,
                    'weather_wind_speed': 5.0,
                    'weather_precipitation': 0.0,
                    'weather_condition': 'Clear'
                }
                
        except Exception as e:
            self.logger.warning(f"Error fetching weather data for fixture {fixture_id}: {e}")
            # Return default values on error
            return {
                'weather_temp_c': 21.0,
                'weather_temp_f': 69.8,
                'weather_humidity': 50.0,
                'weather_wind_speed': 5.0,
                'weather_precipitation': 0.0,
                'weather_condition': 'Clear'
            }
    
    def _calculate_quality_score(self, form_result: Dict[str, Any], 
                               h2h_result: Dict[str, Any], 
                               preview_result: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        # Form features quality (always available from DB)
        if form_result.get('success', False):
            scores.append(1.0)
        else:
            scores.append(0.5)  # Partial score for fallback values
        
        # H2H features quality
        if h2h_result.get('success', False):
            h2h_games = h2h_result.get('h2h_features', {}).get('h2h_overall_games', 0)
            if h2h_games >= 5:
                scores.append(1.0)
            elif h2h_games >= 1:
                scores.append(0.8)
            else:
                scores.append(0.3)
        else:
            scores.append(0.1)
        
        # Preview features quality (optional)
        if preview_result.get('success', False):
            scores.append(0.8)  # Preview is nice to have but not critical
        else:
            scores.append(0.5)  # Default values are acceptable
        
        return round(sum(scores) / len(scores), 2)
    
    def _store_computed_features(self, fixture_id: int, fixture: Dict[str, Any],
                               all_features: Dict[str, Any], quality_score: float) -> None:
        """Store all 87 computed features in the database."""
        # Get target variables if match is finished
        target_data = self._get_target_variables(fixture_id, fixture)
        
        # Build dynamic SQL query for all 87 features
        feature_columns = []
        feature_values = []
        update_clauses = []
        
        # Add all features from enhanced computer
        for feature_name, feature_value in all_features.items():
            feature_columns.append(feature_name)
            feature_values.append(f':{feature_name}')
            update_clauses.append(f'{feature_name} = EXCLUDED.{feature_name}')
        
        # Base columns
        base_columns = ['fixture_id', 'home_team_id', 'away_team_id', 'match_date', 'league_id']
        base_values = [':fixture_id', ':home_team_id', ':away_team_id', ':match_date', ':league_id']
        
        # Target and metadata columns
        meta_columns = ['total_goals', 'over_2_5', 'home_score', 'away_score', 'match_result',
                       'features_computed_at', 'data_quality_score', 'computation_source']
        meta_values = [':total_goals', ':over_2_5', ':home_score', ':away_score', ':match_result',
                      ':features_computed_at', ':data_quality_score', ':computation_source']
        meta_updates = [f'{col} = EXCLUDED.{col}' for col in meta_columns]
        
        all_columns = base_columns + feature_columns + meta_columns
        all_values = base_values + feature_values + meta_values
        all_updates = update_clauses + meta_updates
        
        query = text(f"""
            INSERT INTO pre_computed_features (
                {', '.join(all_columns)}
            ) VALUES (
                {', '.join(all_values)}
            )
            ON CONFLICT (fixture_id) DO UPDATE SET
                {', '.join(all_updates)}
        """)
        
        # Prepare parameters for the query
        params = {
            'fixture_id': fixture_id,
            'home_team_id': fixture['home_team_id'] if isinstance(fixture, dict) else fixture.home_team_id,
            'away_team_id': fixture['away_team_id'] if isinstance(fixture, dict) else fixture.away_team_id,
            'match_date': fixture['match_date'] if isinstance(fixture, dict) else fixture.match_date,
            'league_id': fixture['league_id'] if isinstance(fixture, dict) else fixture.league_id,
            
            # Target variables
            'total_goals': target_data.get('total_goals'),
            'over_2_5': target_data.get('over_2_5'),
            'home_score': target_data.get('home_score'),
            'away_score': target_data.get('away_score'),
            'match_result': target_data.get('match_result'),
            
            # Metadata
            'features_computed_at': datetime.now(),
            'data_quality_score': quality_score,
            'computation_source': 'database'
        }
        
        # Add all features from enhanced computer
        params.update(all_features)
        
        self.db_session.execute(query, params)
        
        self.db_session.commit()
    
    def _get_target_variables(self, fixture_id: int, fixture: Dict[str, Any]) -> Dict[str, Any]:
        """Get target variables for finished matches."""
        if fixture['status'] != 'FINISHED':
            return {}
        
        # Get scores from fixtures table
        query = text("""
            SELECT home_score, away_score
            FROM fixtures
            WHERE id = :fixture_id AND home_score IS NOT NULL AND away_score IS NOT NULL
        """)
        
        result = self.db_session.execute(query, {'fixture_id': fixture_id}).fetchone()
        if not result:
            return {}
        
        home_score, away_score = result
        total_goals = home_score + away_score
        over_2_5 = total_goals > 2.5
        
        if home_score > away_score:
            match_result = 'H'
        elif away_score > home_score:
            match_result = 'A'
        else:
            match_result = 'D'
        
        return {
            'total_goals': total_goals,
            'over_2_5': over_2_5,
            'home_score': home_score,
            'away_score': away_score,
            'match_result': match_result
        }
    
    def _log_computation_success(self, fixture_id: int, computation_type: str,
                               computation_time_ms: int, api_calls: int,
                               features_computed: int, quality_score: float) -> None:
        """Log successful computation."""
        query = text("""
            INSERT INTO feature_computation_log (
                fixture_id, computation_type, status, computation_time_ms,
                api_calls_used, features_computed, data_quality_score
            ) VALUES (
                :fixture_id, :computation_type, 'success', :computation_time_ms,
                :api_calls_used, :features_computed, :data_quality_score
            )
        """)
        
        self.db_session.execute(query, {
            'fixture_id': fixture_id,
            'computation_type': computation_type,
            'computation_time_ms': computation_time_ms,
            'api_calls_used': api_calls,
            'features_computed': features_computed,
            'data_quality_score': quality_score
        })
        
        self.db_session.commit()
    
    def _log_computation_failure(self, fixture_id: int, computation_type: str,
                               error_message: str, computation_time_ms: int = 0) -> None:
        """Log failed computation."""
        query = text("""
            INSERT INTO feature_computation_log (
                fixture_id, computation_type, status, error_message, computation_time_ms
            ) VALUES (
                :fixture_id, :computation_type, 'failed', :error_message, :computation_time_ms
            )
        """)
        
        self.db_session.execute(query, {
            'fixture_id': fixture_id,
            'computation_type': computation_type,
            'error_message': error_message,
            'computation_time_ms': computation_time_ms
        })
        
        self.db_session.commit()
    
    def _log_computation_stats(self) -> None:
        """Log final computation statistics."""
        duration = self.stats['end_time'] - self.stats['start_time']
        success_rate = (self.stats['successful_computations'] / 
                       max(self.stats['total_fixtures'], 1)) * 100
        
        self.logger.info(
            f"Feature computation completed: "
            f"{self.stats['successful_computations']}/{self.stats['total_fixtures']} successful "
            f"({success_rate:.1f}%), {self.stats['api_calls_used']} API calls, "
            f"{self.stats['cache_hits']} cache hits, {duration:.1f}s total"
        )
    
    def _update_data_quality_metrics(self) -> None:
        """Update daily data quality metrics."""
        today = datetime.now().date()
        
        # Calculate feature completeness rate (ensure it's between 0.0 and 1.0)
        completeness_rate = min(1.0, max(0.0, 
            self.stats['successful_computations'] / max(self.stats['total_fixtures'], 1)
        ))
        
        # Calculate cache hit rate (ensure it's between 0.0 and 1.0)
        total_requests = self.stats['api_calls_used'] + self.stats['cache_hits']
        cache_hit_rate = min(1.0, max(0.0, 
            self.stats['cache_hits'] / max(total_requests, 1)
        ))
        
        # Update metrics
        metrics = [
            ('feature_completeness', round(completeness_rate, 4), self.stats['total_fixtures'], 
             self.stats['successful_computations'], self.stats['failed_computations']),
            ('cache_hit_rate', round(cache_hit_rate, 4), total_requests, 
             self.stats['cache_hits'], self.stats['api_calls_used'])
        ]
        
        for metric_type, value, total, successful, failed in metrics:
            query = text("""
                INSERT INTO data_quality_metrics (
                    metric_date, metric_type, metric_value, total_records,
                    successful_records, failed_records
                ) VALUES (
                    :metric_date, :metric_type, :metric_value, :total_records,
                    :successful_records, :failed_records
                )
                ON CONFLICT (metric_date, metric_type) DO UPDATE SET
                    metric_value = EXCLUDED.metric_value,
                    total_records = EXCLUDED.total_records,
                    successful_records = EXCLUDED.successful_records,
                    failed_records = EXCLUDED.failed_records,
                    created_at = CURRENT_TIMESTAMP
            """)
            
            self.db_session.execute(query, {
                'metric_date': today,
                'metric_type': metric_type,
                'metric_value': value,
                'total_records': total,
                'successful_records': successful,
                'failed_records': failed
            })
        
        self.db_session.commit()