"""Enhanced Feature Computer

This module implements computation logic for all 87 expected features including
Markov chains, team positions, league averages, strength metrics, and advanced analytics.

Author: FormFinder2 Team
Created: 2025-01-27
Purpose: Compute all missing features for complete model training
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import get_db_session
from .exceptions import FeatureComputationError
from .markov_feature_generator import MarkovFeatureGenerator

class EnhancedFeatureComputer:
    """Computes all 87 expected features for model training."""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.markov_generator = MarkovFeatureGenerator(db_session)
        
    def compute_all_features(self, fixture_id: int) -> Dict[str, Any]:
        """Compute all 87 features for a fixture.
        
        Args:
            fixture_id: Fixture ID to compute features for
            
        Returns:
            Dictionary containing all computed features
        """
        try:
            # Get fixture details
            fixture = self._get_fixture_details(fixture_id)
            if not fixture:
                raise FeatureComputationError(f"Fixture {fixture_id} not found")
            
            features = {}
            
            # Compute basic form features
            form_features = self._compute_form_features(fixture)
            features.update(form_features)
            
            # Compute H2H features
            h2h_features = self._compute_h2h_features(fixture)
            features.update(h2h_features)
            
            # Compute team strength features
            strength_features = self._compute_strength_features(fixture)
            features.update(strength_features)
            
            # Compute position features
            position_features = self._compute_position_features(fixture)
            features.update(position_features)
            
            # Compute league features
            league_features = self._compute_league_features(fixture)
            features.update(league_features)
            
            # Compute Markov features
            markov_features = self._compute_markov_features(fixture)
            features.update(markov_features)
            
            # Compute sentiment and xG features
            advanced_features = self._compute_advanced_features(fixture)
            features.update(advanced_features)
            
            # Compute weather features
            weather_features = self._compute_weather_features(fixture)
            features.update(weather_features)
            
            self.logger.info(f"Computed {len(features)} features for fixture {fixture_id}")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature computation failed for fixture {fixture_id}: {e}")
            raise
    
    def _get_fixture_details(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Get fixture details from database."""
        query = text("""
            SELECT id, home_team_id, away_team_id, match_date, league_id, status,
                   home_score, away_score
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
                'status': result[5],
                'home_score': result[6],
                'away_score': result[7]
            }
        return None
    
    def _compute_form_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Compute team form features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_id = fixture[f'{team_type}_team_id']
            
            # Get recent matches
            recent_matches = self._get_recent_matches(
                team_id, fixture['match_date'], fixture['league_id'], limit=10
            )
            
            if not recent_matches:
                # Default values for teams with no recent matches
                features.update({
                    f'{team_type}_avg_goals_for': 1.5,
                    f'{team_type}_avg_goals_against': 1.5,
                    f'{team_type}_form_diff': 0.0,
                    f'{team_type}_team_form_score': 0.5
                })
                continue
            
            # Calculate averages
            goals_for = []
            goals_against = []
            form_points = []
            
            for match in recent_matches:
                if match['home_team_id'] == team_id:
                    goals_for.append(match['home_score'])
                    goals_against.append(match['away_score'])
                    if match['home_score'] > match['away_score']:
                        form_points.append(3)
                    elif match['home_score'] == match['away_score']:
                        form_points.append(1)
                    else:
                        form_points.append(0)
                else:
                    goals_for.append(match['away_score'])
                    goals_against.append(match['home_score'])
                    if match['away_score'] > match['home_score']:
                        form_points.append(3)
                    elif match['away_score'] == match['home_score']:
                        form_points.append(1)
                    else:
                        form_points.append(0)
            
            avg_goals_for = np.mean(goals_for) if goals_for else 1.5
            avg_goals_against = np.mean(goals_against) if goals_against else 1.5
            form_score = np.mean(form_points) / 3.0 if form_points else 0.5
            
            # Form difference (recent 5 vs previous 5)
            recent_5_points = np.mean(form_points[:5]) / 3.0 if len(form_points) >= 5 else form_score
            previous_5_points = np.mean(form_points[5:10]) / 3.0 if len(form_points) >= 10 else form_score
            form_diff = recent_5_points - previous_5_points
            
            features.update({
                f'{team_type}_avg_goals_for': avg_goals_for,
                f'{team_type}_avg_goals_against': avg_goals_against,
                f'{team_type}_form_diff': form_diff,
                f'{team_type}_team_form_score': form_score
            })
        
        return features
    
    def _compute_h2h_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Compute head-to-head features."""
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        
        # Get H2H matches
        h2h_matches = self._get_h2h_matches(home_team_id, away_team_id, fixture['match_date'])
        
        if not h2h_matches:
            return {
                'h2h_total_goals': 2.5,
                'h2h_competitiveness': 0.5,
                'h2h_total_matches': 0,
                'h2h_avg_goals': 2.5,
                'h2h_home_wins': 0,
                'h2h_away_wins': 0
            }
        
        total_goals = []
        home_wins = 0
        away_wins = 0
        draws = 0
        
        for match in h2h_matches:
            total_goals.append(match['home_score'] + match['away_score'])
            
            if match['home_team_id'] == home_team_id:
                if match['home_score'] > match['away_score']:
                    home_wins += 1
                elif match['home_score'] < match['away_score']:
                    away_wins += 1
                else:
                    draws += 1
            else:
                if match['away_score'] > match['home_score']:
                    home_wins += 1
                elif match['away_score'] < match['home_score']:
                    away_wins += 1
                else:
                    draws += 1
        
        avg_total_goals = np.mean(total_goals) if total_goals else 2.5
        total_matches = len(h2h_matches)
        
        # Competitiveness (how close the matches are)
        competitiveness = draws / total_matches if total_matches > 0 else 0.5
        if total_matches > 0:
            win_distribution = [home_wins, away_wins, draws]
            competitiveness = 1.0 - (max(win_distribution) / total_matches)
        
        return {
            'h2h_total_goals': avg_total_goals,
            'h2h_competitiveness': competitiveness,
            'h2h_total_matches': total_matches,
            'h2h_avg_goals': avg_total_goals,
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins
        }
    
    def _compute_strength_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Compute team strength features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_id = fixture[f'{team_type}_team_id']
            
            # Get team's recent performance
            recent_matches = self._get_recent_matches(
                team_id, fixture['match_date'], fixture['league_id'], limit=10
            )
            
            if not recent_matches:
                features.update({
                    f'{team_type}_attack_strength': 1.0,
                    f'{team_type}_defense_strength': 1.0
                })
                continue
            
            # Calculate league averages for context
            league_avg_goals = self._get_league_average_goals(fixture['league_id'])
            
            goals_scored = []
            goals_conceded = []
            
            for match in recent_matches:
                if match['home_team_id'] == team_id:
                    goals_scored.append(match['home_score'])
                    goals_conceded.append(match['away_score'])
                else:
                    goals_scored.append(match['away_score'])
                    goals_conceded.append(match['home_score'])
            
            avg_goals_scored = np.mean(goals_scored) if goals_scored else league_avg_goals
            avg_goals_conceded = np.mean(goals_conceded) if goals_conceded else league_avg_goals
            
            # Strength relative to league average (convert decimal to float)
            league_avg_goals_float = float(league_avg_goals) if league_avg_goals is not None else 1.0
            attack_strength = avg_goals_scored / league_avg_goals_float if league_avg_goals_float > 0 else 1.0
            defense_strength = league_avg_goals_float / avg_goals_conceded if avg_goals_conceded > 0 else 1.0
            
            features.update({
                f'{team_type}_attack_strength': attack_strength,
                f'{team_type}_defense_strength': defense_strength
            })
        
        return features
    
    def _compute_position_features(self, fixture: Dict[str, Any]) -> Dict[str, int]:
        """Compute team position features."""
        features = {}
        
        for team_type in ['home', 'away']:
            team_id = fixture[f'{team_type}_team_id']
            
            # Get current league position
            position = self._get_team_position_simple(team_id, fixture['league_id'], fixture['match_date'])
            features[f'{team_type}_team_position'] = position
        
        return features
    
    def _compute_league_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Compute league-level features."""
        league_avg_goals = self._get_league_average_goals(fixture['league_id'])
        
        return {
            'league_avg_goals': league_avg_goals
        }
    
    def _compute_markov_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Compute Markov chain features."""
        try:
            # Generate Markov features for both teams
            home_markov = self.markov_generator.generate_team_features(
                fixture['home_team_id'],
                fixture['league_id'],
                fixture['match_date'],
                'home'
            )
            
            away_markov = self.markov_generator.generate_team_features(
                fixture['away_team_id'],
                fixture['league_id'],
                fixture['match_date'],
                'away'
            )
            
            # Combine and rename features
            markov_features = {}
            
            # Home team Markov features
            for key, value in home_markov.items():
                if key not in ['feature_date', 'home_away_context', 'lookback_window']:
                    # Convert trend_direction string to numeric value
                    if key == 'trend_direction':
                        value = self._convert_trend_direction_to_numeric(value)
                    markov_features[f'markov_home_{key}'] = value
            
            # Away team Markov features
            for key, value in away_markov.items():
                if key not in ['feature_date', 'home_away_context', 'lookback_window']:
                    # Convert trend_direction string to numeric value
                    if key == 'trend_direction':
                        value = self._convert_trend_direction_to_numeric(value)
                    markov_features[f'markov_away_{key}'] = value
            
            # Compute differential features
            if 'markov_home_momentum_score' in markov_features and 'markov_away_momentum_score' in markov_features:
                markov_features['markov_momentum_diff'] = (
                    markov_features['markov_home_momentum_score'] - 
                    markov_features['markov_away_momentum_score']
                )
            
            if 'markov_home_performance_volatility' in markov_features and 'markov_away_performance_volatility' in markov_features:
                markov_features['markov_volatility_diff'] = (
                    markov_features['markov_home_performance_volatility'] - 
                    markov_features['markov_away_performance_volatility']
                )
            
            if 'markov_home_transition_entropy' in markov_features and 'markov_away_transition_entropy' in markov_features:
                markov_features['markov_entropy_diff'] = (
                    markov_features['markov_home_transition_entropy'] - 
                    markov_features['markov_away_transition_entropy']
                )
            
            return markov_features
            
        except Exception as e:
            self.logger.warning(f"Markov feature computation failed: {e}")
            # Return default Markov features
            return self._get_default_markov_features()
    
    def _convert_trend_direction_to_numeric(self, trend_direction: str) -> float:
        """Convert trend direction string to numeric value.
        
        Args:
            trend_direction: String value ('improving', 'stable', 'declining')
            
        Returns:
            Numeric representation: 1.0 for improving, 0.0 for stable, -1.0 for declining
        """
        trend_mapping = {
            'improving': 1.0,
            'stable': 0.0,
            'declining': -1.0
        }
        return trend_mapping.get(trend_direction, 0.0)
    
    def _compute_advanced_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Compute advanced features like home advantage, sentiment, xG."""
        features = {}
        
        # Home advantage calculation
        home_advantage = self._calculate_home_advantage(fixture['league_id'])
        defensive_home_advantage = self._calculate_defensive_home_advantage(fixture['league_id'])
        
        features.update({
            'home_advantage': home_advantage,
            'defensive_home_advantage': defensive_home_advantage
        })
        
        return features
    
    def _compute_weather_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Compute weather features with improved fallback logic."""
        # First try to get weather data for the specific fixture
        weather_query = text("""
            SELECT temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, weather_code
            FROM weather_data
            WHERE fixture_id = :fixture_id
            ORDER BY weather_datetime DESC
            LIMIT 1
        """)
        
        result = self.db_session.execute(weather_query, {'fixture_id': fixture['id']}).fetchone()
        
        if result:
            temp_c = result[0] or 21.0
            return {
                'weather_temp_c': temp_c,
                'weather_temp_f': (temp_c * 9/5) + 32,
                'weather_humidity': result[1] or 50.0,
                'weather_wind_speed': result[2] or 5.0,
                'weather_precipitation': result[3] or 0.0,
                'weather_condition': self._get_weather_condition_from_code(result[4]) if result[4] else 'Clear'
            }
        
        # If no weather data for this fixture, try to get recent weather data from the same location/league
        fallback_query = text("""
            SELECT temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, weather_code
            FROM weather_data wd
            JOIN fixtures f ON wd.fixture_id = f.id
            WHERE f.league_id = :league_id
                AND wd.weather_datetime <= :match_date
            ORDER BY wd.weather_datetime DESC
            LIMIT 1
        """)
        
        fallback_result = self.db_session.execute(fallback_query, {
            'league_id': fixture['league_id'],
            'match_date': fixture['match_date']
        }).fetchone()
        
        if fallback_result:
            temp_c = fallback_result[0] or 21.0
            return {
                'weather_temp_c': temp_c,
                'weather_temp_f': (temp_c * 9/5) + 32,
                'weather_humidity': fallback_result[1] or 50.0,
                'weather_wind_speed': fallback_result[2] or 5.0,
                'weather_precipitation': fallback_result[3] or 0.0,
                'weather_condition': self._get_weather_condition_from_code(fallback_result[4]) if fallback_result[4] else 'Clear'
            }
        
        # If still no data, return seasonal defaults based on match date
        return self._get_seasonal_weather_defaults(fixture['match_date'])
    
    def _get_weather_condition_from_code(self, weather_code: int) -> str:
        """Convert weather code to condition string."""
        weather_codes = {
            0: 'Clear',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Fog',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            61: 'Slight rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            71: 'Slight snow',
            73: 'Moderate snow',
            75: 'Heavy snow',
            95: 'Thunderstorm',
            96: 'Thunderstorm with hail',
            99: 'Thunderstorm with heavy hail'
        }
        return weather_codes.get(weather_code, 'Clear')
    
    def _get_seasonal_weather_defaults(self, match_date: datetime) -> Dict[str, float]:
        """Get seasonal weather defaults based on match date."""
        month = match_date.month
        
        # European seasonal averages (most leagues are European)
        if month in [12, 1, 2]:  # Winter
            return {
                'weather_temp_c': 5.0,
                'weather_temp_f': 41.0,
                'weather_humidity': 75.0,
                'weather_wind_speed': 8.0,
                'weather_precipitation': 2.0,
                'weather_condition': 'Overcast'
            }
        elif month in [3, 4, 5]:  # Spring
            return {
                'weather_temp_c': 15.0,
                'weather_temp_f': 59.0,
                'weather_humidity': 65.0,
                'weather_wind_speed': 6.0,
                'weather_precipitation': 1.0,
                'weather_condition': 'Partly cloudy'
            }
        elif month in [6, 7, 8]:  # Summer
            return {
                'weather_temp_c': 25.0,
                'weather_temp_f': 77.0,
                'weather_humidity': 55.0,
                'weather_wind_speed': 4.0,
                'weather_precipitation': 0.5,
                'weather_condition': 'Clear'
            }
        else:  # Autumn (9, 10, 11)
            return {
                'weather_temp_c': 12.0,
                'weather_temp_f': 53.6,
                'weather_humidity': 70.0,
                'weather_wind_speed': 7.0,
                'weather_precipitation': 1.5,
                'weather_condition': 'Partly cloudy'
            }
    
    def _get_recent_matches(self, team_id: int, match_date: datetime, 
                           league_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent matches for a team."""
        query = text("""
            SELECT home_team_id, away_team_id, home_score, away_score, match_date
            FROM fixtures
            WHERE (home_team_id = :team_id OR away_team_id = :team_id)
                AND league_id = :league_id
                AND match_date < :match_date
                AND status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
            ORDER BY match_date DESC
            LIMIT :limit
        """)
        
        results = self.db_session.execute(query, {
            'team_id': team_id,
            'league_id': league_id,
            'match_date': match_date,
            'limit': limit
        }).fetchall()
        
        return [{
            'home_team_id': row[0],
            'away_team_id': row[1],
            'home_score': row[2],
            'away_score': row[3],
            'match_date': row[4]
        } for row in results]
    
    def _get_h2h_matches(self, home_team_id: int, away_team_id: int, 
                        match_date: datetime, limit: int = 10) -> List[Dict[str, Any]]:
        """Get head-to-head matches between two teams."""
        query = text("""
            SELECT home_team_id, away_team_id, home_score, away_score, match_date
            FROM fixtures
            WHERE ((home_team_id = :home_team_id AND away_team_id = :away_team_id)
                   OR (home_team_id = :away_team_id AND away_team_id = :home_team_id))
                AND match_date < :match_date
                AND status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
            ORDER BY match_date DESC
            LIMIT :limit
        """)
        
        results = self.db_session.execute(query, {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'match_date': match_date,
            'limit': limit
        }).fetchall()
        
        return [{
            'home_team_id': row[0],
            'away_team_id': row[1],
            'home_score': row[2],
            'away_score': row[3],
            'match_date': row[4]
        } for row in results]
    
    def _get_league_average_goals(self, league_id: int) -> float:
        """Get league average goals per game."""
        query = text("""
            SELECT AVG(home_score + away_score) as avg_goals
            FROM fixtures
            WHERE league_id = :league_id
                AND status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
                AND match_date >= :start_date
        """)
        
        start_date = datetime.now() - timedelta(days=365)  # Last year
        result = self.db_session.execute(query, {
            'league_id': league_id,
            'start_date': start_date
        }).fetchone()
        
        return result[0] if result and result[0] else 2.5
    
    def _get_team_position(self, team_id: int, league_id: int, match_date: datetime) -> tuple[int, float]:
        """Get team's current league position with confidence score.
        
        Returns:
            tuple: (position, confidence_score)
                - position: Team's league position
                - confidence_score: 1.0 for recent data, 0.8 for older data, 0.5 for median, 0.0 for default
        """
        try:
            # Check if this league has any standings data at all (cache this check)
            if not hasattr(self, '_leagues_with_standings'):
                self._leagues_with_standings = set()
                self._leagues_without_standings = set()
                
                # Get all leagues that have standings data
                leagues_query = text("""
                    SELECT DISTINCT league_id FROM standings
                """)
                leagues_result = self.db_session.execute(leagues_query).fetchall()
                self._leagues_with_standings = {row[0] for row in leagues_result}
            
            # If this league has no standings data at all, use intelligent fallback without warning
            if league_id not in self._leagues_with_standings:
                 # Get league info
                 league_info_query = text("""
                     SELECT name FROM leagues WHERE id = :league_id LIMIT 1
                 """)
                 league_info = self.db_session.execute(league_info_query, {'league_id': league_id}).fetchone()
                 league_name = league_info[0] if league_info else f"League {league_id}"
                 
                 # Check if it's likely a cup competition based on name
                 cup_keywords = ['cup', 'champions', 'europa', 'conference', 'trophy', 'championship']
                 is_likely_cup = any(keyword in league_name.lower() for keyword in cup_keywords)
                 
                 if is_likely_cup:
                     # For cup competitions, use a neutral middle position
                     return 8, 0.3  # Slightly better than default for cups
                 else:
                     # For leagues without standings, only log warning once per league
                     if league_id not in getattr(self, '_logged_missing_leagues', set()):
                         if not hasattr(self, '_logged_missing_leagues'):
                             self._logged_missing_leagues = set()
                         self._logged_missing_leagues.add(league_id)
                         self.logger.warning(f"No standings data available for {league_name} (ID: {league_id}). Consider fetching standings for this league.")
                     return 10, 0.0
            
            # First, get the season for this league_id
            season_query = text("""
                SELECT season
                FROM leagues
                WHERE id = :league_id
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            season_result = self.db_session.execute(season_query, {
                'league_id': league_id
            }).fetchone()
            
            if not season_result:
                return 10, 0.0
            
            season = season_result[0]
            
            # Now get the most recent position data for this season
            query = text("""
                SELECT position, updated_at
                FROM standings
                WHERE team_id = :team_id
                    AND league_id = :league_id
                    AND season = :season
                ORDER BY updated_at DESC
                LIMIT 1
            """)
            
            result = self.db_session.execute(query, {
                'team_id': team_id,
                'league_id': league_id,
                'season': season
            }).fetchone()
            
            if result:
                position, updated_at = result
                # Check if data is recent (within 30 days of match date)
                if updated_at and match_date:
                    days_diff = abs((match_date - updated_at).days)
                    if days_diff <= 30:
                        return position, 1.0  # High confidence for recent data
                    else:
                        return position, 0.8  # Lower confidence for older data
                else:
                    return position, 0.8  # Medium confidence when dates unavailable
            
            # If no position data found for this team, try league-specific median for this season
            median_query = text("""
                SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY position) as median_position
                FROM standings
                WHERE league_id = :league_id
                    AND season = :season
                    AND position IS NOT NULL
            """)
            
            median_result = self.db_session.execute(median_query, {
                'league_id': league_id,
                'season': season
            }).fetchone()
            
            if median_result and median_result[0]:
                median_position = int(round(median_result[0]))
                return median_position, 0.5  # Medium confidence for median
            
            # Final fallback - only log warning for specific team if we haven't already
            team_key = f"{team_id}_{league_id}"
            if not hasattr(self, '_logged_missing_teams'):
                self._logged_missing_teams = set()
            
            if team_key not in self._logged_missing_teams:
                self._logged_missing_teams.add(team_key)
                self.logger.warning(f"No position data available for team {team_id} in league {league_id}, using default")
            
            return 10, 0.0  # No confidence for default
            
        except Exception as e:
            self.logger.debug(f"Error getting team position for team {team_id}: {e}")
            return 10, 0.0
    
    def _get_team_position_simple(self, team_id: int, league_id: int, match_date: datetime) -> int:
        """Get team's current league position (simple version for backward compatibility)."""
        position, _ = self._get_team_position(team_id, league_id, match_date)
        return position
    
    def _calculate_home_advantage(self, league_id: int) -> float:
        """Calculate home advantage for the league."""
        query = text("""
            SELECT 
                AVG(CASE WHEN home_score > away_score THEN 1.0 ELSE 0.0 END) as home_win_rate,
                AVG(home_score - away_score) as avg_goal_diff
            FROM fixtures
            WHERE league_id = :league_id
                AND status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
                AND match_date >= :start_date
        """)
        
        start_date = datetime.now() - timedelta(days=365)
        result = self.db_session.execute(query, {
            'league_id': league_id,
            'start_date': start_date
        }).fetchone()
        
        if result and result[0]:
            return result[1] or 0.0  # Average goal difference
        return 0.3  # Default home advantage
    
    def _calculate_defensive_home_advantage(self, league_id: int) -> float:
        """Calculate defensive home advantage for the league."""
        query = text("""
            SELECT AVG(away_score - home_score) as avg_defensive_diff
            FROM fixtures
            WHERE league_id = :league_id
                AND status = 'finished'
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
                AND match_date >= :start_date
        """)
        
        start_date = datetime.now() - timedelta(days=365)
        result = self.db_session.execute(query, {
            'league_id': league_id,
            'start_date': start_date
        }).fetchone()
        
        return -(result[0] or 0.0) if result else 0.2  # Defensive advantage
    
    def _get_default_markov_features(self) -> Dict[str, Any]:
        """Get default Markov features when computation fails."""
        return {
            'markov_home_current_state': 'average',  # String state
            'markov_home_state_duration': 3.0,
            'markov_home_state_confidence': 0.6,
            'markov_home_momentum_score': 0.0,
            'markov_home_trend_direction': 0.0,  # Numeric (stable)
            'markov_home_state_stability': 0.5,
            'markov_home_transition_entropy': 1.5,
            'markov_home_performance_volatility': 0.3,
            'markov_home_expected_next_state': 'average',  # String state
            'markov_home_next_state_probability': 0.4,
            'markov_home_mean_return_time': 5.0,
            'markov_home_steady_state_probability': 0.2,
            'markov_home_absorption_probability': 0.0,
            'markov_away_current_state': 'average',  # String state
            'markov_away_state_duration': 3.0,
            'markov_away_state_confidence': 0.6,
            'markov_away_momentum_score': 0.0,
            'markov_away_trend_direction': 0.0,  # Numeric (stable)
            'markov_away_state_stability': 0.5,
            'markov_away_transition_entropy': 1.5,
            'markov_away_performance_volatility': 0.3,
            'markov_away_expected_next_state': 'average',  # String state
            'markov_away_next_state_probability': 0.4,
            'markov_away_mean_return_time': 5.0,
            'markov_away_steady_state_probability': 0.2,
            'markov_away_absorption_probability': 0.0,
            'markov_momentum_diff': 0.0,
            'markov_volatility_diff': 0.0,
            'markov_entropy_diff': 0.0
        }