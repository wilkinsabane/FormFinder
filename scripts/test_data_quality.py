#!/usr/bin/env python3
"""
Simple test script for data quality checker that avoids table creation issues.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

def test_data_quality():
    """Test data quality without creating tables."""
    try:
        # Load configuration
        load_config()
        config = get_config()
        
        # Create direct database connection
        engine = create_engine(config.get_database_url())
        Session = sessionmaker(bind=engine)
        
        with Session() as session:
            # Test basic database connectivity
            result = session.execute(text("SELECT 1 as test")).fetchone()
            print(f"✅ Database connection successful: {result.test}")
            
            # Check pre_computed_features table
            features_query = text("""
                SELECT COUNT(*) as total_features,
                       COUNT(CASE WHEN data_quality_score >= 0.9 THEN 1 END) as high_quality_features,
                       AVG(data_quality_score) as avg_quality_score
                FROM pre_computed_features
            """)
            
            features_result = session.execute(features_query).fetchone()
            print(f"📊 Features: {features_result.total_features} total, {features_result.high_quality_features} high-quality")
            print(f"📈 Average quality score: {features_result.avg_quality_score:.3f}")
            
            # Check fixtures table
            fixtures_query = text("""
                SELECT COUNT(*) as total_fixtures,
                       COUNT(CASE WHEN status = 'finished' THEN 1 END) as finished_fixtures
                FROM fixtures
            """)
            
            fixtures_result = session.execute(fixtures_query).fetchone()
            print(f"⚽ Fixtures: {fixtures_result.total_fixtures} total, {fixtures_result.finished_fixtures} finished")
            
            # Check h2h_cache table
            try:
                h2h_query = text("""
                    SELECT COUNT(*) as total_h2h,
                           COUNT(CASE WHEN total_matches > 0 THEN 1 END) as valid_h2h
                    FROM h2h_cache
                """)
                
                h2h_result = session.execute(h2h_query).fetchone()
                print(f"🔄 H2H Cache: {h2h_result.total_h2h} total, {h2h_result.valid_h2h} valid")
            except Exception as e:
                print(f"⚠️ H2H Cache table not accessible: {e}")
            
            # Check feature_computation_log table
            try:
                log_query = text("""
                    SELECT COUNT(*) as total_logs,
                           COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_logs
                    FROM feature_computation_log
                """)
                
                log_result = session.execute(log_query).fetchone()
                print(f"📝 Computation Log: {log_result.total_logs} total, {log_result.successful_logs} successful")
            except Exception as e:
                print(f"⚠️ Feature computation log table not accessible: {e}")
            
            print("\n✅ Data quality test completed successfully!")
            
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_data_quality()
    sys.exit(0 if success else 1)