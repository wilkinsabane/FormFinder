#!/usr/bin/env python3
"""
Unified script to populate the pre_computed_features table.

This script combines the best features from both existing populate_precomputed_features.py scripts:
1. Team statistics extraction from historical fixtures (from root script)
2. Enhanced predictor features integration (from scripts/ version)
3. Proper weather data retrieval using WeatherFetcher
4. Configuration loading and error handling

The goal is to create complete and accurate data in the pre_computed_features table
without null/zero values for critical features.
"""

import asyncio
import json
import logging
import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.database import DatabaseManager, PreComputedFeatures, Fixture, get_db_session
from formfinder.weather_fetcher import WeatherFetcher
from enhanced_predictor import EnhancedGoalPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('populate_precomputed_features_unified.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
console = Console()


class UnifiedFeaturePopulator:
    """Unified populator that combines team stats, enhanced features, and weather data."""
    
    def __init__(self):
        """Initialize the unified populator with all required components."""
        # Load configuration first
        load_config()
        self.config = get_config()
        
        # Initialize database components
        self.db_manager = DatabaseManager()
        self.engine = self.db_manager.engine
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize enhanced predictor
        try:
            self.enhanced_predictor = EnhancedGoalPredictor()
            logger.info("Enhanced predictor initialized successfully")
        except Exception as e:
            logger.warning(f"Enhanced predictor initialization failed: {e}")
            self.enhanced_predictor = None
        
        # Initialize weather fetcher
        self.weather_fetcher = None
        
    async def initialize_weather_fetcher(self, session: Session):
        """Initialize weather fetcher with database session."""
        try:
            self.weather_fetcher = WeatherFetcher(session)
            logger.info("Weather fetcher initialized successfully")
        except Exception as e:
            logger.warning(f"Weather fetcher initialization failed: {e}")
            self.weather_fetcher = None
    
    def get_fixtures_to_process(self, session: Session, limit: Optional[int] = None, force_sentiment: bool = False) -> List[Dict[str, Any]]:
        """Get finished fixtures that need feature computation.
        
        Args:
            session: Database session
            limit: Optional limit on number of fixtures to process
            force_sentiment: If True, include fixtures that need sentiment data regeneration
            
        Returns:
            List of fixture dictionaries with required information
        """
        if force_sentiment:
            # Get fixtures that have features but need sentiment data regeneration
            query = """
            SELECT 
                f.id as fixture_id,
                f.home_team_id,
                f.away_team_id,
                f.match_date,
                f.league_id,
                f.home_score,
                f.away_score,
                f.stadium_city,
                (f.home_score + f.away_score) as total_goals,
                CASE WHEN (f.home_score + f.away_score) > 2.5 THEN TRUE ELSE FALSE END as over_2_5,
                CASE 
                    WHEN f.home_score > f.away_score THEN 'H'
                    WHEN f.away_score > f.home_score THEN 'A'
                    ELSE 'D'
                END as match_result
            FROM fixtures f
            INNER JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.home_score IS NOT NULL 
                AND f.away_score IS NOT NULL
                AND f.status = 'finished'
                AND (pcf.home_team_sentiment = 0.0 OR pcf.away_team_sentiment = 0.0 OR pcf.home_team_sentiment IS NULL OR pcf.away_team_sentiment IS NULL)
                AND f.match_date >= '2020-09-01'
            ORDER BY f.match_date DESC
            """
        else:
            # Original query for fixtures without any features
            query = """
            SELECT 
                f.id as fixture_id,
                f.home_team_id,
                f.away_team_id,
                f.match_date,
                f.league_id,
                f.home_score,
                f.away_score,
                f.stadium_city,
                (f.home_score + f.away_score) as total_goals,
                CASE WHEN (f.home_score + f.away_score) > 2.5 THEN TRUE ELSE FALSE END as over_2_5,
                CASE 
                    WHEN f.home_score > f.away_score THEN 'H'
                    WHEN f.away_score > f.home_score THEN 'A'
                    ELSE 'D'
                END as match_result
            FROM fixtures f
            LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
            WHERE f.home_score IS NOT NULL 
                AND f.away_score IS NOT NULL
                AND f.status = 'finished'
                AND pcf.fixture_id IS NULL
                AND f.match_date >= '2020-09-01'  -- Use available data
            ORDER BY f.match_date DESC
            """
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = session.execute(text(query))
        fixtures = [{
            'fixture_id': row.fixture_id,
            'home_team_id': row.home_team_id,
            'away_team_id': row.away_team_id,
            'match_date': row.match_date,
            'league_id': row.league_id,
            'home_score': row.home_score,
            'away_score': row.away_score,
            'stadium_city': row.stadium_city,
            'total_goals': row.total_goals,
            'over_2_5': row.over_2_5,
            'match_result': row.match_result
        } for row in result]
        
        logger.info(f"Found {len(fixtures)} fixtures to process")
        return fixtures
    
    def get_team_stats(self, team_id: int, match_date: datetime, session: Session) -> Dict[str, Any]:
        """Get comprehensive team statistics from historical fixtures.
        
        Args:
            team_id: ID of the team
            match_date: Date of the match (to exclude future matches)
            session: Database session
            
        Returns:
            Dictionary containing team statistics
        """
        lookback_date = match_date - timedelta(days=730)  # Look back 2 year(s)
        
        query = """
        SELECT 
            CASE WHEN home_team_id = :team_id THEN 'home' ELSE 'away' END as venue,
            CASE WHEN home_team_id = :team_id THEN home_score ELSE away_score END as goals_for,
            CASE WHEN home_team_id = :team_id THEN away_score ELSE home_score END as goals_against,
            match_date,
            CASE 
                WHEN (home_team_id = :team_id AND home_score > away_score) OR 
                     (away_team_id = :team_id AND away_score > home_score) THEN 'W'
                WHEN home_score = away_score THEN 'D'
                ELSE 'L'
            END as result
        FROM fixtures
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
            AND match_date < :match_date
            AND match_date >= :lookback_date
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
        ORDER BY match_date DESC
        LIMIT 20
        """
        
        try:
            matches_df = pd.read_sql_query(text(query), session.bind, params={
                'team_id': team_id,
                'match_date': match_date,
                'lookback_date': lookback_date
            })
            
            if matches_df.empty:
                # Return None when no historical data is available
                return None
            
            # Calculate statistics
            last_5 = matches_df.head(5)
            home_matches = matches_df[matches_df['venue'] == 'home']
            away_matches = matches_df[matches_df['venue'] == 'away']
            
            # Create form sequence for last 5 games
            form_sequence = last_5['result'].tolist()
            
            return {
                'avg_goals_for': float(matches_df['goals_for'].mean()),
                'avg_goals_against': float(matches_df['goals_against'].mean()),
                'avg_goals_for_home': float(home_matches['goals_for'].mean()) if not home_matches.empty else 1.5,
                'avg_goals_against_home': float(home_matches['goals_against'].mean()) if not home_matches.empty else 1.1,
                'avg_goals_for_away': float(away_matches['goals_for'].mean()) if not away_matches.empty else 1.1,
                'avg_goals_against_away': float(away_matches['goals_against'].mean()) if not away_matches.empty else 1.5,
                'wins_last_5': int(len(last_5[last_5['result'] == 'W'])),
                'draws_last_5': int(len(last_5[last_5['result'] == 'D'])),
                'losses_last_5': int(len(last_5[last_5['result'] == 'L'])),
                'goals_for_last_5': int(last_5['goals_for'].sum()),
                'goals_against_last_5': int(last_5['goals_against'].sum()),
                'form_last_5_games': json.dumps(form_sequence)
            }
        
        except Exception as e:
            logger.warning(f"Error getting stats for team {team_id}: {e}")
            # Return None when error occurs
            return None
    
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

    async def get_weather_features(self, fixture_info: Dict[str, Any], session: Session) -> Dict[str, Any]:
        """Get weather features for a fixture.
        
        Args:
            fixture_info: Fixture information dictionary
            session: Database session
            
        Returns:
            Dictionary containing weather features
        """
        # Try to get weather data from database first
        weather_features = self._get_weather_features_from_db(fixture_info['fixture_id'], session)
        
        # If no weather data found and we have weather fetcher, try to fetch it
        if (weather_features['weather_temp_c'] == 21.0 and  # Default value indicates no data
            self.weather_fetcher and 
            fixture_info.get('stadium_city')):
            
            try:
                # Create a minimal fixture object for weather fetching
                class MinimalFixture:
                    def __init__(self, fixture_info):
                        self.id = fixture_info['fixture_id']
                        self.stadium_city = fixture_info['stadium_city']
                        self.match_date = fixture_info['match_date']
                
                minimal_fixture = MinimalFixture(fixture_info)
                success = await self.weather_fetcher.fetch_weather_for_fixture(minimal_fixture)
                
                if success:
                    # Try to get weather data again after fetching
                    weather_features = self._get_weather_features_from_db(fixture_info['fixture_id'], session)
                    logger.info(f"Successfully fetched weather data for fixture {fixture_info['fixture_id']}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch weather for fixture {fixture_info['fixture_id']}: {e}")
        
        return weather_features
    
    def _get_weather_features_from_db(self, fixture_id: int, session: Session) -> Dict[str, Any]:
        """Get weather features from database WeatherData model.
        
        Args:
            fixture_id: ID of the fixture to get weather data for
            session: Database session
            
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
            
            result = session.execute(query, {'fixture_id': fixture_id}).fetchone()
            
            if result:
                temp_c = result[0] if result[0] is not None else 21.0
                temp_f = (temp_c * 9/5) + 32 if temp_c is not None else 69.8
                weather_code = result[6] if result[6] is not None else 0
                
                return {
                    'weather_temp_c': round(temp_c, 1),
                    'weather_temp_f': round(temp_f, 1),
                    'weather_humidity': result[1] if result[1] is not None else 50.0,
                    'weather_wind_speed': result[2] if result[2] is not None else 5.0,
                    'weather_precipitation': result[3] if result[3] is not None else 0.0,
                    'weather_condition': self._get_weather_condition_from_code(weather_code)
                }
            else:
                # Return default values if no weather data found
                return {
                    'weather_temp_c': 21.0,
                    'weather_temp_f': 69.8,
                    'weather_humidity': 50.0,
                    'weather_wind_speed': 5.0,
                    'weather_precipitation': 0.0,
                    'weather_condition': 'Clear'
                }
        
        except Exception as e:
            logger.warning(f"Error getting weather data for fixture {fixture_id}: {e}")
            return {
                'weather_temp_c': 21.0,
                'weather_temp_f': 69.8,
                'weather_humidity': 50.0,
                'weather_wind_speed': 5.0,
                'weather_precipitation': 0.0,
                'weather_condition': 'Clear'
            }
    
    def extract_enhanced_features(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Extract features using enhanced predictor if available.
        
        Args:
            fixture_id: ID of the fixture to extract features for
            
        Returns:
            Feature dictionary or None if extraction fails
        """
        if not self.enhanced_predictor:
            return None
        
        try:
            features = self.enhanced_predictor.extract_enhanced_features(fixture_id)
            if features is not None and len(features) > 0:
                logger.debug(f"Extracted {len(features)} enhanced features for fixture {fixture_id}")
                return features
            else:
                logger.warning(f"No enhanced features extracted for fixture {fixture_id}")
                return None
        except Exception as e:
            logger.warning(f"Failed to extract enhanced features for fixture {fixture_id}: {e}")
            return None
    
    async def generate_unified_features(self, fixture_info: Dict[str, Any], session: Session) -> Dict[str, Any]:
        """Generate unified features combining team stats, enhanced features, and weather data.
        
        Args:
            fixture_info: Fixture information dictionary
            session: Database session
            
        Returns:
            Dictionary containing all unified features
        """
        # Get team statistics from historical data
        home_stats = self.get_team_stats(fixture_info['home_team_id'], fixture_info['match_date'], session)
        away_stats = self.get_team_stats(fixture_info['away_team_id'], fixture_info['match_date'], session)
        
        # Get weather features
        weather_features = await self.get_weather_features(fixture_info, session)
        
        # Try to get enhanced features
        enhanced_features = self.extract_enhanced_features(fixture_info['fixture_id'])
        logger.debug(f"Extracted enhanced features for fixture {fixture_info['fixture_id']}: {type(enhanced_features)} - {enhanced_features is not None}")
        
        # Helper functions for safe conversion
        def safe_float(value, default=0.0):
            """Safely convert value to float, handling NaN and None."""
            if value is None:
                return default
            try:
                val = float(value)
                return default if np.isnan(val) or np.isinf(val) else val
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            """Safely convert value to int, handling NaN and None."""
            if value is None:
                return default
            try:
                val = float(value)
                return default if np.isnan(val) or np.isinf(val) else int(val)
            except (ValueError, TypeError):
                return default
        
        def trend_direction_to_numeric(trend_direction):
            """Convert trend direction string to numeric value."""
            trend_map = {
                'improving': 1.0,
                'stable': 0.0,
                'declining': -1.0
            }
            return trend_map.get(trend_direction, 0.0)
        
        # Build unified feature dictionary
        unified_features = {
            # Fixture metadata
            'fixture_id': fixture_info['fixture_id'],
            'league_id': fixture_info['league_id'],
            'home_team_id': fixture_info['home_team_id'],
            'away_team_id': fixture_info['away_team_id'],
            'match_date': fixture_info['match_date'],
            
            # Actual match results (for completed fixtures)
            'total_goals': fixture_info['total_goals'],
            'over_2_5': fixture_info['over_2_5'],
            'match_result': fixture_info['match_result'],
            'home_score': fixture_info['home_score'],
            'away_score': fixture_info['away_score'],
            
            # Home team features from historical data
            'home_avg_goals_scored': safe_float(home_stats.get('avg_goals_for') if home_stats else None),
            'home_avg_goals_conceded': safe_float(home_stats.get('avg_goals_against') if home_stats else None),
            'home_avg_goals_scored_home': safe_float(home_stats.get('avg_goals_for_home') if home_stats else None),
            'home_avg_goals_conceded_home': safe_float(home_stats.get('avg_goals_against_home') if home_stats else None),
            'home_wins_last_5': safe_int(home_stats.get('wins_last_5') if home_stats else None),
            'home_draws_last_5': safe_int(home_stats.get('draws_last_5') if home_stats else None),
            'home_losses_last_5': safe_int(home_stats.get('losses_last_5') if home_stats else None),
            'home_goals_for_last_5': safe_int(home_stats.get('goals_for_last_5') if home_stats else None),
            'home_goals_against_last_5': safe_int(home_stats.get('goals_against_last_5') if home_stats else None),
            'home_form_last_5_games': home_stats.get('form_last_5_games', '[]') if home_stats else '[]',
            
            # Away team features from historical data
            'away_avg_goals_scored': safe_float(away_stats.get('avg_goals_for') if away_stats else None),
            'away_avg_goals_conceded': safe_float(away_stats.get('avg_goals_against') if away_stats else None),
            'away_avg_goals_scored_away': safe_float(away_stats.get('avg_goals_for_away') if away_stats else None),
            'away_avg_goals_conceded_away': safe_float(away_stats.get('avg_goals_against_away') if away_stats else None),
            'away_wins_last_5': safe_int(away_stats.get('wins_last_5') if away_stats else None),
            'away_draws_last_5': safe_int(away_stats.get('draws_last_5') if away_stats else None),
            'away_losses_last_5': safe_int(away_stats.get('losses_last_5') if away_stats else None),
            'away_goals_for_last_5': safe_int(away_stats.get('goals_for_last_5') if away_stats else None),
            'away_goals_against_last_5': safe_int(away_stats.get('goals_against_last_5') if away_stats else None),
            'away_form_last_5_games': away_stats.get('form_last_5_games', '[]') if away_stats else '[]',
            
            # Weather features
            **weather_features,
            
            # Default excitement rating
            'excitement_rating': safe_float(np.random.uniform(3.0, 8.0)),
            
            # Metadata
            'features_computed_at': datetime.now(),
            'data_quality_score': 0.8,
            'computation_source': 'unified'
        }
        
        # Merge enhanced features if available
        if enhanced_features is not None and isinstance(enhanced_features, dict):
            logger.debug(f"Enhanced features available for fixture {fixture_info['fixture_id']}: {type(enhanced_features)}")
            # xG features
            unified_features.update({
                'home_xg': safe_float(enhanced_features.get('home_xg', 0.0)),
                'away_xg': safe_float(enhanced_features.get('away_xg', 0.0)),
                
                # Team strength features (attack/defense breakdown)
                'home_attack_strength': safe_float(enhanced_features.get('home_attack_strength', 0.0)),
                'home_defense_strength': safe_float(enhanced_features.get('home_defense_strength', 0.0)),
                'away_attack_strength': safe_float(enhanced_features.get('away_attack_strength', 0.0)),
                'away_defense_strength': safe_float(enhanced_features.get('away_defense_strength', 0.0)),
                
                # Team positions
                'home_team_position': safe_int(enhanced_features.get('home_team_position', 10)),
                'away_team_position': safe_int(enhanced_features.get('away_team_position', 10)),
                
                # Position confidence features
                'home_position_confidence': safe_float(enhanced_features.get('home_position_confidence', 0.0)),
                'away_position_confidence': safe_float(enhanced_features.get('away_position_confidence', 0.0)),
                
                # Home advantage features
                'home_advantage': safe_float(enhanced_features.get('home_advantage', 0.0)),
                'defensive_home_advantage': safe_float(enhanced_features.get('defensive_home_advantage', 0.0)),
                
                # Head-to-head features
                'h2h_total_goals': safe_int(enhanced_features.get('h2h_total_goals', 0)),
                'h2h_competitiveness': safe_float(enhanced_features.get('h2h_competitiveness', 0.0)),
                'h2h_total_matches': safe_int(enhanced_features.get('h2h_total_matches', 0)),
                'h2h_avg_goals': safe_float(enhanced_features.get('h2h_avg_goals', 0.0)),
                'h2h_home_wins': safe_int(enhanced_features.get('h2h_home_wins', 0)),
                'h2h_away_wins': safe_int(enhanced_features.get('h2h_away_wins', 0)),
                
                # Team form features
                'home_form_diff': safe_float(enhanced_features.get('home_form_diff', 0.0)),
                'away_form_diff': safe_float(enhanced_features.get('away_form_diff', 0.0)),
                'home_team_form_score': safe_float(enhanced_features.get('home_team_form_score', 0.0)),
                'away_team_form_score': safe_float(enhanced_features.get('away_team_form_score', 0.0)),
                
                # Average goals features
                'home_avg_goals_for': safe_float(enhanced_features.get('home_avg_goals_for', 0.0)),
                'home_avg_goals_against': safe_float(enhanced_features.get('home_avg_goals_against', 0.0)),
                'away_avg_goals_for': safe_float(enhanced_features.get('away_avg_goals_for', 0.0)),
                'away_avg_goals_against': safe_float(enhanced_features.get('away_avg_goals_against', 0.0)),
                
                # League average goals
                'league_avg_goals': safe_float(enhanced_features.get('league_avg_goals', 2.5)),
                
                # Team strength and momentum features
                'home_team_strength': safe_float(enhanced_features.get('home_team_strength', 0.0)),
                'away_team_strength': safe_float(enhanced_features.get('away_team_strength', 0.0)),
                'home_team_momentum': safe_float(enhanced_features.get('home_team_momentum', 0.0)),
                'away_team_momentum': safe_float(enhanced_features.get('away_team_momentum', 0.0)),
                
                # Sentiment features
                'home_team_sentiment': safe_float(enhanced_features.get('home_team_sentiment', 0.0)),
                'away_team_sentiment': safe_float(enhanced_features.get('away_team_sentiment', 0.0)),
                
                # Markov features - Home team
                'markov_home_current_state': self._validate_team_state(enhanced_features.get('markov_home_current_state', 'average')),
                'markov_home_state_duration': safe_int(enhanced_features.get('markov_home_state_duration', 0)),
                'markov_home_state_confidence': safe_float(enhanced_features.get('markov_home_state_confidence', 0.0)),
                'markov_home_momentum_score': safe_float(enhanced_features.get('markov_home_momentum_score', 0.0)),
                'markov_home_trend_direction': trend_direction_to_numeric(enhanced_features.get('markov_home_trend_direction', 'stable')),
                'markov_home_state_stability': safe_float(enhanced_features.get('markov_home_state_stability', 0.0)),
                'markov_home_transition_entropy': safe_float(enhanced_features.get('markov_home_transition_entropy', 0.0)),
                'markov_home_performance_volatility': safe_float(enhanced_features.get('markov_home_performance_volatility', 0.0)),
                'markov_home_expected_next_state': self._validate_team_state(enhanced_features.get('markov_home_expected_next_state', 'average')),
                'markov_home_next_state_probability': safe_float(enhanced_features.get('markov_home_next_state_probability', 0.0)),
                'markov_home_mean_return_time': safe_float(enhanced_features.get('markov_home_mean_return_time', 0.0)),
                
                # Markov features - Away team
                'markov_away_current_state': self._validate_team_state(enhanced_features.get('markov_away_current_state', 'average')),
                'markov_away_state_duration': safe_int(enhanced_features.get('markov_away_state_duration', 0)),
                'markov_away_state_confidence': safe_float(enhanced_features.get('markov_away_state_confidence', 0.0)),
                'markov_away_momentum_score': safe_float(enhanced_features.get('markov_away_momentum_score', 0.0)),
                'markov_away_trend_direction': trend_direction_to_numeric(enhanced_features.get('markov_away_trend_direction', 'stable')),
                'markov_away_performance_volatility': safe_float(enhanced_features.get('markov_away_performance_volatility', 0.0)),
                'markov_away_expected_next_state': self._validate_team_state(enhanced_features.get('markov_away_expected_next_state', 'average')),
                'markov_away_next_state_probability': safe_float(enhanced_features.get('markov_away_next_state_probability', 0.0)),
                'markov_away_mean_return_time': safe_float(enhanced_features.get('markov_away_mean_return_time', 0.0)),
                'markov_away_steady_state_probability': safe_float(enhanced_features.get('markov_away_steady_state_probability', 0.0)),
                'markov_away_absorption_probability': safe_float(enhanced_features.get('markov_away_absorption_probability', 0.0)),
                
                # Markov differential features
                'markov_momentum_diff': safe_float(enhanced_features.get('markov_momentum_diff', 0.0)),
                'markov_volatility_diff': safe_float(enhanced_features.get('markov_volatility_diff', 0.0)),
                'markov_entropy_diff': safe_float(enhanced_features.get('markov_entropy_diff', 0.0)),
                
                # Legacy Markov features for backward compatibility
                'home_team_markov_momentum': safe_float(enhanced_features.get('markov_home_momentum_score', 0.0)),
                'away_team_markov_momentum': safe_float(enhanced_features.get('markov_away_momentum_score', 0.0)),
                'home_team_state_stability': safe_float(enhanced_features.get('markov_home_state_stability', 0.0)),
                'away_team_state_stability': safe_float(enhanced_features.get('markov_away_state_stability', 0.0)),
                'home_team_transition_entropy': safe_float(enhanced_features.get('markov_home_transition_entropy', 0.0)),
                'away_team_transition_entropy': safe_float(enhanced_features.get('markov_away_transition_entropy', 0.0)),
                'home_team_performance_volatility': safe_float(enhanced_features.get('markov_home_performance_volatility', 0.0)),
                'away_team_performance_volatility': safe_float(enhanced_features.get('markov_away_performance_volatility', 0.0)),
                'home_team_current_state': self._validate_team_state(enhanced_features.get('markov_home_current_state', 'average')),
                'away_team_current_state': self._validate_team_state(enhanced_features.get('markov_away_current_state', 'average')),
                'home_team_state_duration': safe_int(enhanced_features.get('markov_home_state_duration', 0)),
                'away_team_state_duration': safe_int(enhanced_features.get('markov_away_state_duration', 0)),
                'home_team_expected_next_state': self._validate_team_state(enhanced_features.get('markov_home_expected_next_state', 'average')),
                'away_team_expected_next_state': self._validate_team_state(enhanced_features.get('markov_away_expected_next_state', 'average')),
                'home_team_state_confidence': safe_float(enhanced_features.get('markov_home_state_confidence', 0.0)),
                'away_team_state_confidence': safe_float(enhanced_features.get('markov_away_state_confidence', 0.0)),
                'markov_match_prediction_confidence': safe_float(enhanced_features.get('markov_match_prediction_confidence', 0.0)),
                'markov_outcome_probabilities': str(enhanced_features.get('markov_outcome_probabilities', '{}') if enhanced_features else '{}'),
                
                # Update data quality score if enhanced features are available
                'data_quality_score': 0.9,
                'computation_source': 'unified_enhanced'
            })
        
        return unified_features
    
    def _validate_team_state(self, state: str) -> str:
        """Validate team state value.
        
        Args:
            state: Team state string
            
        Returns:
            Valid team state string
        """
        valid_states = ['excellent', 'good', 'average', 'poor', 'terrible']
        return state if state in valid_states else 'average'
    
    def save_features_to_db(self, features: Dict[str, Any], session: Session, force_sentiment: bool = False) -> bool:
        """Save unified features to the pre_computed_features table.
        
        Args:
            features: Dictionary of feature values
            session: Database session
            force_sentiment: If True, update existing records instead of inserting
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if force_sentiment:
                # Update existing record with new sentiment data
                update_query = text("""
                UPDATE pre_computed_features 
                SET home_team_sentiment = :home_team_sentiment,
                    away_team_sentiment = :away_team_sentiment,
                    features_computed_at = :features_computed_at,
                    computation_source = :computation_source
                WHERE fixture_id = :fixture_id
                """)
                
                session.execute(update_query, {
                    'home_team_sentiment': features['home_team_sentiment'],
                    'away_team_sentiment': features['away_team_sentiment'],
                    'features_computed_at': features['features_computed_at'],
                    'computation_source': features['computation_source'],
                    'fixture_id': features['fixture_id']
                })
            else:
                # Convert to DataFrame for easy insertion
                df = pd.DataFrame([features])
                df.to_sql('pre_computed_features', session.bind, if_exists='append', index=False)
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save features for fixture {features['fixture_id']}: {e}")
            session.rollback()
            return False
    
    async def populate_features(self, limit: Optional[int] = None, force_sentiment: bool = False):
        """Main method to populate pre-computed features.
        
        Args:
            limit: Optional limit on number of fixtures to process
            force_sentiment: If True, regenerate sentiment data for existing fixtures
        """
        if force_sentiment:
            console.print("[bold blue]=== Regenerating Sentiment Data ===[/bold blue]")
        else:
            console.print("[bold blue]=== Unified Pre-computed Features Population ===[/bold blue]")
        
        with get_db_session() as session:
            # Initialize weather fetcher
            await self.initialize_weather_fetcher(session)
            
            # Get fixtures to process
            fixtures = self.get_fixtures_to_process(session, limit, force_sentiment)
            
            if not fixtures:
                if force_sentiment:
                    console.print("✅ All fixtures already have sentiment data")
                else:
                    console.print("✅ All fixtures already have pre-computed features")
                return
            
            # Process fixtures with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing fixtures...", total=len(fixtures))
                
                success_count = 0
                error_count = 0
                
                for fixture_info in fixtures:
                    try:
                        # Generate unified features
                        features = await self.generate_unified_features(fixture_info, session)
                        
                        # Check if features were generated successfully
                        if features is None:
                            logger.error(f"Failed to generate features for fixture {fixture_info['fixture_id']}")
                            error_count += 1
                        else:
                            # Save to database
                            if self.save_features_to_db(features, session, force_sentiment):
                                success_count += 1
                                logger.debug(f"Successfully processed fixture {fixture_info['fixture_id']}")
                            else:
                                error_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing fixture {fixture_info['fixture_id']}: {e}")
                        error_count += 1
                    
                    progress.update(task, advance=1)
            
            console.print(f"✅ Successfully processed {success_count} fixtures")
            if error_count > 0:
                console.print(f"⚠️ {error_count} fixtures had errors")
            
            # Verify results
            verify_query = "SELECT COUNT(*) as count FROM pre_computed_features WHERE total_goals IS NOT NULL"
            result = pd.read_sql_query(text(verify_query), session.bind)
            console.print(f"📈 Total pre-computed features now: {result.iloc[0]['count']}")


async def main():
    """Main entry point for the unified feature populator."""
    parser = argparse.ArgumentParser(description='Populate pre-computed features for fixtures')
    parser.add_argument('--limit', type=int, help='Limit number of fixtures to process')
    parser.add_argument('--force-sentiment', action='store_true', 
                       help='Force regeneration of sentiment data for existing fixtures')
    
    args = parser.parse_args()
    
    try:
        populator = UnifiedFeaturePopulator()
        await populator.populate_features(limit=args.limit, force_sentiment=args.force_sentiment)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        console.print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())