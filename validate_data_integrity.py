#!/usr/bin/env python3
"""
Validate data integrity in pre_computed_features table.
Focus on numerical fields, form data structure, and identifying patterns.
"""

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
import pandas as pd
import json

def validate_data_integrity():
    """Comprehensive validation of pre_computed_features table data integrity."""
    print("=== Data Integrity Validation ===\n")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # 1. Basic table statistics
    print("1. BASIC TABLE STATISTICS")
    print("-" * 40)
    
    basic_stats_query = """
    SELECT 
        COUNT(*) as total_rows,
        COUNT(CASE WHEN total_goals IS NOT NULL THEN 1 END) as rows_with_total_goals,
        COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as rows_with_home_score,
        COUNT(CASE WHEN away_score IS NOT NULL THEN 1 END) as rows_with_away_score,
        COUNT(CASE WHEN data_quality_score IS NOT NULL THEN 1 END) as rows_with_quality_score,
        MIN(match_date) as earliest_date,
        MAX(match_date) as latest_date
    FROM pre_computed_features
    """
    
    try:
        result = pd.read_sql_query(text(basic_stats_query), engine)
        row = result.iloc[0]
        print(f"Total rows: {row['total_rows']}")
        print(f"Rows with total_goals: {row['rows_with_total_goals']}")
        print(f"Rows with home_score: {row['rows_with_home_score']}")
        print(f"Rows with away_score: {row['rows_with_away_score']}")
        print(f"Rows with quality_score: {row['rows_with_quality_score']}")
        print(f"Date range: {row['earliest_date']} to {row['latest_date']}")
    except Exception as e:
        print(f"Error in basic stats: {e}")
        return
    
    # 2. Numerical fields analysis - focus on zero values
    print("\n2. NUMERICAL FIELDS ANALYSIS")
    print("-" * 40)
    
    numerical_analysis_query = """
    SELECT 
        COUNT(CASE WHEN home_avg_goals_scored = 0 THEN 1 END) as home_avg_goals_scored_zeros,
        COUNT(CASE WHEN home_avg_goals_scored IS NOT NULL THEN 1 END) as home_avg_goals_scored_total,
        COUNT(CASE WHEN away_avg_goals_scored = 0 THEN 1 END) as away_avg_goals_scored_zeros,
        COUNT(CASE WHEN away_avg_goals_scored IS NOT NULL THEN 1 END) as away_avg_goals_scored_total,
        COUNT(CASE WHEN home_wins_last_5 = 0 THEN 1 END) as home_wins_last_5_zeros,
        COUNT(CASE WHEN home_wins_last_5 IS NOT NULL THEN 1 END) as home_wins_last_5_total,
        COUNT(CASE WHEN away_wins_last_5 = 0 THEN 1 END) as away_wins_last_5_zeros,
        COUNT(CASE WHEN away_wins_last_5 IS NOT NULL THEN 1 END) as away_wins_last_5_total,
        AVG(CASE WHEN home_avg_goals_scored IS NOT NULL THEN home_avg_goals_scored END) as avg_home_goals_scored,
        AVG(CASE WHEN away_avg_goals_scored IS NOT NULL THEN away_avg_goals_scored END) as avg_away_goals_scored,
        AVG(CASE WHEN data_quality_score IS NOT NULL THEN data_quality_score END) as avg_quality_score
    FROM pre_computed_features
    """
    
    try:
        result = pd.read_sql_query(text(numerical_analysis_query), engine)
        row = result.iloc[0]
        
        print("Home Goals Scored Analysis:")
        if row['home_avg_goals_scored_total'] > 0:
            zero_pct = (row['home_avg_goals_scored_zeros'] / row['home_avg_goals_scored_total']) * 100
            print(f"  Zero values: {row['home_avg_goals_scored_zeros']}/{row['home_avg_goals_scored_total']} ({zero_pct:.1f}%)")
            print(f"  Average value: {row['avg_home_goals_scored']:.3f}")
        else:
            print("  No data available")
            
        print("Away Goals Scored Analysis:")
        if row['away_avg_goals_scored_total'] > 0:
            zero_pct = (row['away_avg_goals_scored_zeros'] / row['away_avg_goals_scored_total']) * 100
            print(f"  Zero values: {row['away_avg_goals_scored_zeros']}/{row['away_avg_goals_scored_total']} ({zero_pct:.1f}%)")
            print(f"  Average value: {row['avg_away_goals_scored']:.3f}")
        else:
            print("  No data available")
            
        print("Home Wins Last 5 Analysis:")
        if row['home_wins_last_5_total'] > 0:
            zero_pct = (row['home_wins_last_5_zeros'] / row['home_wins_last_5_total']) * 100
            print(f"  Zero values: {row['home_wins_last_5_zeros']}/{row['home_wins_last_5_total']} ({zero_pct:.1f}%)")
        else:
            print("  No data available")
            
        print(f"\nOverall Quality Score: {row['avg_quality_score']:.3f}" if row['avg_quality_score'] else "No quality scores available")
        
    except Exception as e:
        print(f"Error in numerical analysis: {e}")
    
    # 3. Form data structure analysis
    print("\n3. FORM DATA STRUCTURE ANALYSIS")
    print("-" * 40)
    
    form_analysis_query = """
    SELECT 
        home_form_last_5_games,
        away_form_last_5_games,
        fixture_id
    FROM pre_computed_features 
    WHERE home_form_last_5_games IS NOT NULL 
    LIMIT 10
    """
    
    try:
        result = pd.read_sql_query(text(form_analysis_query), engine)
        if len(result) > 0:
            print("Sample form data structures:")
            for idx, row in result.iterrows():
                print(f"\nFixture {row['fixture_id']}:")
                print(f"  Home form: {row['home_form_last_5_games']}")
                print(f"  Away form: {row['away_form_last_5_games']}")
                
                # Try to parse as JSON
                try:
                    home_form = json.loads(row['home_form_last_5_games']) if row['home_form_last_5_games'] else None
                    away_form = json.loads(row['away_form_last_5_games']) if row['away_form_last_5_games'] else None
                    print(f"  Home form parsed: {type(home_form)} with {len(home_form) if home_form else 0} items")
                    print(f"  Away form parsed: {type(away_form)} with {len(away_form) if away_form else 0} items")
                except json.JSONDecodeError as e:
                    print(f"  JSON parsing error: {e}")
        else:
            print("No form data found")
    except Exception as e:
        print(f"Error in form analysis: {e}")
    
    # 4. Markov and sentiment features analysis
    print("\n4. MARKOV AND SENTIMENT FEATURES ANALYSIS")
    print("-" * 40)
    
    markov_sentiment_query = """
    SELECT 
        COUNT(CASE WHEN markov_home_current_state IS NOT NULL THEN 1 END) as markov_home_state_count,
        COUNT(CASE WHEN markov_away_current_state IS NOT NULL THEN 1 END) as markov_away_state_count,
        COUNT(CASE WHEN markov_match_prediction_confidence IS NOT NULL THEN 1 END) as markov_confidence_count,
        COUNT(CASE WHEN home_team_sentiment IS NOT NULL THEN 1 END) as home_sentiment_count,
        COUNT(CASE WHEN away_team_sentiment IS NOT NULL THEN 1 END) as away_sentiment_count,
        COUNT(*) as total_rows
    FROM pre_computed_features
    """
    
    try:
        result = pd.read_sql_query(text(markov_sentiment_query), engine)
        row = result.iloc[0]
        total = row['total_rows']
        
        if total > 0:
            print(f"Markov home state coverage: {row['markov_home_state_count']}/{total} ({(row['markov_home_state_count']/total)*100:.1f}%)")
            print(f"Markov away state coverage: {row['markov_away_state_count']}/{total} ({(row['markov_away_state_count']/total)*100:.1f}%)")
            print(f"Markov confidence coverage: {row['markov_confidence_count']}/{total} ({(row['markov_confidence_count']/total)*100:.1f}%)")
            print(f"Home sentiment coverage: {row['home_sentiment_count']}/{total} ({(row['home_sentiment_count']/total)*100:.1f}%)")
            print(f"Away sentiment coverage: {row['away_sentiment_count']}/{total} ({(row['away_sentiment_count']/total)*100:.1f}%)")
        else:
            print("No data available for analysis")
    except Exception as e:
        print(f"Error in Markov/sentiment analysis: {e}")
    
    # 5. Sample data inspection
    print("\n5. SAMPLE DATA INSPECTION")
    print("-" * 40)
    
    sample_query = """
    SELECT 
        fixture_id,
        match_date,
        home_team_id,
        away_team_id,
        total_goals,
        home_score,
        away_score,
        home_avg_goals_scored,
        away_avg_goals_scored,
        data_quality_score
    FROM pre_computed_features 
    ORDER BY id DESC
    LIMIT 5
    """
    
    try:
        result = pd.read_sql_query(text(sample_query), engine)
        if len(result) > 0:
            print("Latest 5 records:")
            for idx, row in result.iterrows():
                print(f"\nFixture {row['fixture_id']} ({row['match_date']})")
                print(f"  Teams: {row['home_team_id']} vs {row['away_team_id']}")
                print(f"  Score: {row['home_score']}-{row['away_score']} (Total: {row['total_goals']})")
                print(f"  Avg Goals: Home {row['home_avg_goals_scored']}, Away {row['away_avg_goals_scored']}")
                print(f"  Quality: {row['data_quality_score']}")
        else:
            print("No sample data available")
    except Exception as e:
        print(f"Error in sample inspection: {e}")
    
    print("\n=== Validation Complete ===")

if __name__ == "__main__":
    validate_data_integrity()