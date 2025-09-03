#!/usr/bin/env python3
"""
Simple data validation for pre_computed_features table.
"""

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
import pandas as pd

def simple_validation():
    """Simple validation of pre_computed_features table."""
    print("=== Simple Data Validation ===")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Basic row count
    basic_query = """
    SELECT 
        COUNT(*) as total_rows,
        COUNT(CASE WHEN total_goals IS NOT NULL THEN 1 END) as rows_with_goals,
        COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as rows_with_scores,
        COUNT(CASE WHEN home_avg_goals_scored IS NOT NULL THEN 1 END) as rows_with_home_avg,
        COUNT(CASE WHEN home_avg_goals_scored = 0 THEN 1 END) as zero_home_avg,
        COUNT(CASE WHEN away_avg_goals_scored = 0 THEN 1 END) as zero_away_avg,
        MIN(match_date) as earliest_date,
        MAX(match_date) as latest_date
    FROM pre_computed_features
    """
    
    try:
        result = pd.read_sql_query(text(basic_query), engine)
        row = result.iloc[0]
        
        print(f"Total rows: {row['total_rows']}")
        print(f"Rows with total_goals: {row['rows_with_goals']}")
        print(f"Rows with scores: {row['rows_with_scores']}")
        print(f"Rows with home avg goals: {row['rows_with_home_avg']}")
        print(f"Zero home avg goals: {row['zero_home_avg']}")
        print(f"Zero away avg goals: {row['zero_away_avg']}")
        print(f"Date range: {row['earliest_date']} to {row['latest_date']}")
        
        if row['rows_with_home_avg'] > 0:
            zero_pct_home = (row['zero_home_avg'] / row['rows_with_home_avg']) * 100
            zero_pct_away = (row['zero_away_avg'] / row['rows_with_home_avg']) * 100
            print(f"\nZero values analysis:")
            print(f"Home avg goals zero rate: {zero_pct_home:.1f}%")
            print(f"Away avg goals zero rate: {zero_pct_away:.1f}%")
            
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Sample data with actual values
    sample_query = """
    SELECT 
        fixture_id,
        match_date,
        total_goals,
        home_score,
        away_score,
        home_avg_goals_scored,
        away_avg_goals_scored,
        home_wins_last_5,
        away_wins_last_5,
        data_quality_score
    FROM pre_computed_features 
    WHERE home_avg_goals_scored IS NOT NULL
    ORDER BY id DESC
    LIMIT 10
    """
    
    try:
        result = pd.read_sql_query(text(sample_query), engine)
        print(f"\nSample data ({len(result)} records):")
        for idx, row in result.iterrows():
            print(f"Fixture {row['fixture_id']}: Score {row['home_score']}-{row['away_score']}, "
                  f"Avg Goals H:{row['home_avg_goals_scored']:.2f} A:{row['away_avg_goals_scored']:.2f}, "
                  f"Wins H:{row['home_wins_last_5']} A:{row['away_wins_last_5']}, "
                  f"Quality:{row['data_quality_score']:.2f}")
    except Exception as e:
        print(f"Error in sample query: {e}")
    
    # Form data analysis
    form_query = """
    SELECT 
        home_form_last_5_games,
        away_form_last_5_games,
        fixture_id
    FROM pre_computed_features 
    WHERE home_form_last_5_games IS NOT NULL 
    AND home_form_last_5_games != '[]'
    LIMIT 5
    """
    
    try:
        result = pd.read_sql_query(text(form_query), engine)
        print(f"\nForm data samples ({len(result)} records):")
        for idx, row in result.iterrows():
            print(f"Fixture {row['fixture_id']}:")
            print(f"  Home: {row['home_form_last_5_games']}")
            print(f"  Away: {row['away_form_last_5_games']}")
    except Exception as e:
        print(f"Error in form query: {e}")
    
    print("\n=== Validation Complete ===")

if __name__ == "__main__":
    simple_validation()