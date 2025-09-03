#!/usr/bin/env python3
"""Check training data size and calibrator issues."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
import pandas as pd
import joblib
from pathlib import Path

def check_training_data():
    """Check the size of available training data."""
    print("=== Training Data Analysis ===")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Check total completed fixtures
    query = """
    SELECT 
        COUNT(*) as total_fixtures,
        MIN(match_date) as earliest_date,
        MAX(match_date) as latest_date
    FROM fixtures 
    WHERE home_score IS NOT NULL 
        AND away_score IS NOT NULL
    """
    
    result = pd.read_sql_query(text(query), engine)
    print(f"Total completed fixtures: {result.iloc[0]['total_fixtures']}")
    print(f"Date range: {result.iloc[0]['earliest_date']} to {result.iloc[0]['latest_date']}")
    
    # Check fixtures by league
    league_query = """
    SELECT 
        l.league_pk as league_id,
        l.name as league_name,
        COUNT(f.id) as fixture_count
    FROM leagues l
    JOIN fixtures f ON l.league_pk = f.league_id
    WHERE f.home_score IS NOT NULL
        AND f.away_score IS NOT NULL
    GROUP BY l.league_pk, l.name
    ORDER BY COUNT(f.id) DESC
    LIMIT 10
    """
    
    league_result = pd.read_sql_query(text(league_query), engine)
    print("\nTop 10 leagues by fixture count:")
    for _, row in league_result.iterrows():
        print(f"  {row['league_name']} (ID: {row['league_id']}): {row['fixture_count']} fixtures")
    
    # Check recent fixtures (last 365 days)
    recent_query = """
    SELECT COUNT(*) as recent_fixtures
    FROM fixtures 
    WHERE home_score IS NOT NULL 
        AND away_score IS NOT NULL
        AND match_date >= CURRENT_DATE - INTERVAL '365 days'
    """
    
    recent_result = pd.read_sql_query(text(recent_query), engine)
    print(f"\nFixtures in last 365 days: {recent_result.iloc[0]['recent_fixtures']}")
    
    engine.dispose()

def check_calibrators():
    """Check calibrator file and contents."""
    print("\n=== Calibrator Analysis ===")
    
    calibrator_path = Path("models/calibrators.joblib")
    
    if not calibrator_path.exists():
        print("❌ Calibrator file does not exist")
        return
    
    file_size = calibrator_path.stat().st_size
    print(f"Calibrator file size: {file_size} bytes")
    
    try:
        calibrators = joblib.load(calibrator_path)
        print(f"Calibrators loaded successfully: {type(calibrators)}")
        print(f"Number of calibrators: {len(calibrators) if hasattr(calibrators, '__len__') else 'N/A'}")
        
        if isinstance(calibrators, dict):
            print("Calibrator contents:")
            for league_id, calibrator_data in calibrators.items():
                print(f"  League {league_id}: {type(calibrator_data)}")
                if isinstance(calibrator_data, dict):
                    print(f"    Keys: {list(calibrator_data.keys())}")
        else:
            print(f"Calibrators type: {type(calibrators)}")
            print(f"Calibrators content: {calibrators}")
            
    except Exception as e:
        print(f"❌ Error loading calibrators: {e}")

def check_predictions_table():
    """Check if predictions table has data for calibration."""
    print("\n=== Predictions Table Analysis ===")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    try:
        # Check if predictions table exists and has data
        pred_query = """
        SELECT COUNT(*) as total_predictions
        FROM predictions p
        JOIN fixtures f ON p.fixture_id = f.id
        WHERE f.home_score IS NOT NULL
            AND f.away_score IS NOT NULL
        """
        
        pred_result = pd.read_sql_query(text(pred_query), engine)
        print(f"Total predictions with actual results: {pred_result.iloc[0]['total_predictions']}")
        
        # Check recent predictions
        recent_pred_query = """
        SELECT COUNT(*) as recent_predictions
        FROM predictions p
        JOIN fixtures f ON p.fixture_id = f.id
        WHERE f.home_score IS NOT NULL
            AND f.away_score IS NOT NULL
            AND f.match_date >= CURRENT_DATE - INTERVAL '90 days'
        """
        
        recent_pred_result = pd.read_sql_query(text(recent_pred_query), engine)
        print(f"Recent predictions (90 days): {recent_pred_result.iloc[0]['recent_predictions']}")
        
        # Check predictions by league
        league_pred_query = """
        SELECT 
            f.league_id,
            COUNT(*) as prediction_count
        FROM predictions p
        JOIN fixtures f ON p.fixture_id = f.id
        WHERE f.home_score IS NOT NULL
            AND f.away_score IS NOT NULL
            AND f.match_date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY f.league_id
        HAVING COUNT(*) >= 20
        ORDER BY COUNT(*) DESC
        """
        
        league_pred_result = pd.read_sql_query(text(league_pred_query), engine)
        print(f"\nLeagues with sufficient predictions for calibration (≥20):")
        for _, row in league_pred_result.iterrows():
            print(f"  League {row['league_id']}: {row['prediction_count']} predictions")
            
    except Exception as e:
        print(f"❌ Error checking predictions table: {e}")
    
    engine.dispose()

if __name__ == "__main__":
    check_training_data()
    check_calibrators()
    check_predictions_table()