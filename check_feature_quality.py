#!/usr/bin/env python3
"""
Check the quality of features in pre_computed_features table.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
import pandas as pd

def check_feature_quality():
    """Check the quality of computed features."""
    load_config()
    
    with get_db_session() as session:
        conn = session.get_bind().raw_connection()
        
        # Check overall statistics for basic features
        query1 = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(home_xg) as home_xg_count,
            COUNT(away_xg) as away_xg_count,
            COUNT(home_team_strength) as home_strength_count,
            COUNT(away_team_strength) as away_strength_count,
            COUNT(home_team_momentum) as home_momentum_count,
            COUNT(away_team_momentum) as away_momentum_count,
            COUNT(home_team_sentiment) as home_sentiment_count,
            COUNT(away_team_sentiment) as away_sentiment_count
        FROM pre_computed_features
        """
        
        df1 = pd.read_sql(query1, conn)
        print("=== Feature Completeness Statistics ===")
        print(df1.to_string(index=False))
        print()
        
        # Check null percentages
        query2 = """
        SELECT 
            ROUND(100.0 * (COUNT(*) - COUNT(home_xg)) / COUNT(*), 2) as home_xg_null_pct,
            ROUND(100.0 * (COUNT(*) - COUNT(away_xg)) / COUNT(*), 2) as away_xg_null_pct,
            ROUND(100.0 * (COUNT(*) - COUNT(home_team_strength)) / COUNT(*), 2) as home_strength_null_pct,
            ROUND(100.0 * (COUNT(*) - COUNT(away_team_strength)) / COUNT(*), 2) as away_strength_null_pct,
            ROUND(100.0 * (COUNT(*) - COUNT(home_team_momentum)) / COUNT(*), 2) as home_momentum_null_pct,
            ROUND(100.0 * (COUNT(*) - COUNT(away_team_momentum)) / COUNT(*), 2) as away_momentum_null_pct,
            ROUND(100.0 * (COUNT(*) - COUNT(home_team_sentiment)) / COUNT(*), 2) as home_sentiment_null_pct,
            ROUND(100.0 * (COUNT(*) - COUNT(away_team_sentiment)) / COUNT(*), 2) as away_sentiment_null_pct
        FROM pre_computed_features
        """
        
        df2 = pd.read_sql(query2, conn)
        print("=== Null Value Percentages ===")
        print(df2.to_string(index=False))
        print()
        
        # Sample records with null values
        query3 = """
        SELECT 
            fixture_id,
            home_xg,
            away_xg,
            home_team_strength,
            away_team_strength,
            home_team_momentum,
            away_team_momentum,
            home_team_sentiment,
            away_team_sentiment
        FROM pre_computed_features
        WHERE home_xg IS NULL OR away_xg IS NULL OR home_team_strength IS NULL
        ORDER BY fixture_id DESC
        LIMIT 10
        """
        
        df3 = pd.read_sql(query3, conn)
        print("=== Sample Records with Null Values ===")
        print(df3.to_string(index=False))
        print()
        
        # Sample records with complete values
        query4 = """
        SELECT 
            fixture_id,
            home_xg,
            away_xg,
            home_team_strength,
            away_team_strength,
            home_team_momentum,
            away_team_momentum,
            home_team_sentiment,
            away_team_sentiment
        FROM pre_computed_features
        WHERE home_xg IS NOT NULL AND away_xg IS NOT NULL AND home_team_strength IS NOT NULL
        ORDER BY fixture_id DESC
        LIMIT 10
        """
        
        df4 = pd.read_sql(query4, conn)
        print("=== Sample Records with Complete Values ===")
        print(df4.to_string(index=False))
        
        conn.close()

if __name__ == "__main__":
    check_feature_quality()