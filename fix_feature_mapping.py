#!/usr/bin/env python3
"""
Fix feature mapping issues in pre_computed_features table.

This script addresses two main issues:
1. Weather data showing identical values (defaults) due to missing weather data in database
2. H2H features not being properly populated due to column name mismatches

The script will:
- Check current weather data availability
- Fetch missing weather data for recent fixtures
- Update the feature computation to ensure proper H2H mapping
- Validate the fixes by checking data variation
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.database import get_db_connection
from formfinder.weather_fetcher import WeatherFetcher
from populate_precomputed_features_unified import UnifiedFeaturePopulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureMappingFixer:
    """Fix feature mapping issues in pre_computed_features table."""
    
    def __init__(self):
        """Initialize the feature mapping fixer."""
        self.engine = get_db_connection()
        self.Session = sessionmaker(bind=self.engine)
        self.weather_fetcher = WeatherFetcher()
        
    async def check_weather_data_availability(self) -> Dict[str, Any]:
        """Check weather data availability in the database."""
        logger.info("Checking weather data availability...")
        
        with self.Session() as session:
            # Check total fixtures vs weather data coverage
            query = text("""
                SELECT 
                    COUNT(DISTINCT f.id) as total_fixtures,
                    COUNT(DISTINCT w.fixture_id) as fixtures_with_weather,
                    COUNT(DISTINCT CASE WHEN f.match_date >= :recent_date THEN f.id END) as recent_fixtures,
                    COUNT(DISTINCT CASE WHEN f.match_date >= :recent_date THEN w.fixture_id END) as recent_with_weather
                FROM fixtures f
                LEFT JOIN weather_data w ON f.id = w.fixture_id
                WHERE f.home_score IS NOT NULL
            """)
            
            recent_date = datetime.now() - timedelta(days=30)
            result = session.execute(query, {'recent_date': recent_date}).fetchone()
            
            return {
                'total_fixtures': result[0],
                'fixtures_with_weather': result[1],
                'recent_fixtures': result[2],
                'recent_with_weather': result[3],
                'weather_coverage_pct': (result[1] / result[0] * 100) if result[0] > 0 else 0,
                'recent_weather_coverage_pct': (result[3] / result[2] * 100) if result[2] > 0 else 0
            }
    
    async def fetch_missing_weather_data(self, limit: int = 20) -> Dict[str, Any]:
        """Fetch missing weather data for recent fixtures."""
        logger.info(f"Fetching missing weather data for up to {limit} fixtures...")
        
        with self.Session() as session:
            # Get recent fixtures without weather data
            query = text("""
                SELECT f.id, f.stadium_city, f.match_date
                FROM fixtures f
                LEFT JOIN weather_data w ON f.id = w.fixture_id
                WHERE w.fixture_id IS NULL
                    AND f.home_score IS NOT NULL
                    AND f.match_date >= :recent_date
                    AND f.stadium_city IS NOT NULL
                ORDER BY f.match_date DESC
                LIMIT :limit
            """)
            
            recent_date = datetime.now() - timedelta(days=60)
            results = session.execute(query, {
                'recent_date': recent_date,
                'limit': limit
            }).fetchall()
            
            logger.info(f"Found {len(results)} fixtures without weather data")
            
            success_count = 0
            error_count = 0
            
            for fixture_id, stadium_city, match_date in results:
                try:
                    # Create a minimal fixture object
                    class MinimalFixture:
                        def __init__(self, fixture_id, stadium_city, match_date):
                            self.id = fixture_id
                            self.stadium_city = stadium_city
                            self.match_date = match_date
                    
                    fixture = MinimalFixture(fixture_id, stadium_city, match_date)
                    success = await self.weather_fetcher.fetch_weather_for_fixture(fixture)
                    
                    if success:
                        success_count += 1
                        logger.info(f"✓ Fetched weather for fixture {fixture_id} ({stadium_city})")
                    else:
                        error_count += 1
                        logger.warning(f"✗ Failed to fetch weather for fixture {fixture_id} ({stadium_city})")
                        
                    # Small delay to avoid overwhelming the weather API
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error fetching weather for fixture {fixture_id}: {e}")
            
            return {
                'total_attempted': len(results),
                'success_count': success_count,
                'error_count': error_count
            }
    
    async def recompute_features_for_recent_fixtures(self, limit: int = 10) -> Dict[str, Any]:
        """Recompute features for recent fixtures to fix H2H and weather mapping."""
        logger.info(f"Recomputing features for up to {limit} recent fixtures...")
        
        with self.Session() as session:
            # Get recent fixtures that need feature recomputation
            query = text("""
                SELECT f.id, f.home_team_id, f.away_team_id, f.match_date, f.league_id,
                       f.home_score, f.away_score, f.stadium_city
                FROM fixtures f
                WHERE f.home_score IS NOT NULL
                    AND f.match_date >= :recent_date
                ORDER BY f.match_date DESC
                LIMIT :limit
            """)
            
            recent_date = datetime.now() - timedelta(days=30)
            results = session.execute(query, {
                'recent_date': recent_date,
                'limit': limit
            }).fetchall()
            
            logger.info(f"Found {len(results)} recent fixtures to recompute")
            
            # Initialize the unified feature populator
            populator = UnifiedFeaturePopulator()
            
            success_count = 0
            error_count = 0
            
            for row in results:
                try:
                    fixture_info = {
                        'fixture_id': row[0],
                        'home_team_id': row[1],
                        'away_team_id': row[2],
                        'match_date': row[3],
                        'league_id': row[4],
                        'home_score': row[5],
                        'away_score': row[6],
                        'total_goals': (row[5] or 0) + (row[6] or 0),
                        'over_2_5': 1 if ((row[5] or 0) + (row[6] or 0)) > 2.5 else 0,
                        'match_result': 'H' if (row[5] or 0) > (row[6] or 0) else ('A' if (row[6] or 0) > (row[5] or 0) else 'D'),
                        'stadium_city': row[7]
                    }
                    
                    # Generate unified features
                    features = await populator.generate_unified_features(fixture_info, session)
                    
                    if features:
                        # Update computation source to indicate this is a fixed version
                        features['computation_source'] = 'unified_fixed'
                        features['features_computed_at'] = datetime.now()
                        
                        # Save features to database
                        await populator.save_precomputed_features([features], session)
                        
                        success_count += 1
                        logger.info(f"✓ Recomputed features for fixture {row[0]}")
                    else:
                        error_count += 1
                        logger.warning(f"✗ Failed to generate features for fixture {row[0]}")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error recomputing features for fixture {row[0]}: {e}")
            
            return {
                'total_attempted': len(results),
                'success_count': success_count,
                'error_count': error_count
            }
    
    async def validate_fixes(self) -> Dict[str, Any]:
        """Validate that the fixes have improved data variation."""
        logger.info("Validating fixes...")
        
        with self.Session() as session:
            # Check data variation in key columns
            query = text("""
                SELECT 
                    COUNT(DISTINCT h2h_total_matches) as h2h_total_matches_unique,
                    COUNT(DISTINCT h2h_home_wins) as h2h_home_wins_unique,
                    COUNT(DISTINCT h2h_away_wins) as h2h_away_wins_unique,
                    COUNT(DISTINCT weather_temp_c) as weather_temp_unique,
                    COUNT(DISTINCT weather_humidity) as weather_humidity_unique,
                    COUNT(DISTINCT weather_condition) as weather_condition_unique,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN h2h_total_matches IS NOT NULL AND h2h_total_matches > 0 THEN 1 END) as h2h_populated,
                    COUNT(CASE WHEN weather_temp_c != 21.0 THEN 1 END) as weather_non_default,
                    COUNT(CASE WHEN computation_source = 'unified_fixed' THEN 1 END) as fixed_records
                FROM pre_computed_features
                WHERE created_at >= :recent_date
            """)
            
            recent_date = datetime.now() - timedelta(days=7)
            result = session.execute(query, {'recent_date': recent_date}).fetchone()
            
            return {
                'total_records': result[9],
                'fixed_records': result[9],
                'h2h_variation': {
                    'total_matches_unique': result[0],
                    'home_wins_unique': result[1],
                    'away_wins_unique': result[2],
                    'populated_count': result[7]
                },
                'weather_variation': {
                    'temp_unique': result[3],
                    'humidity_unique': result[4],
                    'condition_unique': result[5],
                    'non_default_count': result[8]
                }
            }

async def main():
    """Main function to fix feature mapping issues."""
    logger.info("Starting feature mapping fix process...")
    
    fixer = FeatureMappingFixer()
    
    try:
        # Step 1: Check current weather data availability
        logger.info("=== Step 1: Checking weather data availability ===")
        weather_status = await fixer.check_weather_data_availability()
        logger.info(f"Weather data coverage: {weather_status['weather_coverage_pct']:.1f}% overall, {weather_status['recent_weather_coverage_pct']:.1f}% recent")
        
        # Step 2: Fetch missing weather data
        logger.info("=== Step 2: Fetching missing weather data ===")
        weather_fetch_result = await fixer.fetch_missing_weather_data(limit=20)
        logger.info(f"Weather fetch result: {weather_fetch_result['success_count']}/{weather_fetch_result['total_attempted']} successful")
        
        # Step 3: Recompute features for recent fixtures
        logger.info("=== Step 3: Recomputing features for recent fixtures ===")
        recompute_result = await fixer.recompute_features_for_recent_fixtures(limit=15)
        logger.info(f"Feature recomputation: {recompute_result['success_count']}/{recompute_result['total_attempted']} successful")
        
        # Step 4: Validate fixes
        logger.info("=== Step 4: Validating fixes ===")
        validation_result = await fixer.validate_fixes()
        logger.info(f"Validation results:")
        logger.info(f"  - Total records: {validation_result['total_records']}")
        logger.info(f"  - Fixed records: {validation_result['fixed_records']}")
        logger.info(f"  - H2H variation: {validation_result['h2h_variation']}")
        logger.info(f"  - Weather variation: {validation_result['weather_variation']}")
        
        logger.info("Feature mapping fix process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during feature mapping fix: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())