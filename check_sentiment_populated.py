#!/usr/bin/env python3
"""Check if sentiment values are now populated in the database."""

import os
import sys
from sqlalchemy import text

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and initialize config first
from formfinder.config import load_config, get_config
load_config()

# Now import database modules
from formfinder.database import get_db_manager

def check_sentiment_populated():
    """Check if sentiment values are populated in pre_computed_features."""
    print("Checking Sentiment Values in Database")
    print("=" * 40)
    
    try:
        db_manager = get_db_manager()
        
        with db_manager.get_session() as session:
            # Check for non-zero sentiment values
            query = text("""
                SELECT 
                    fixture_id,
                    home_team_sentiment,
                    away_team_sentiment,
                    features_computed_at
                FROM pre_computed_features 
                WHERE (home_team_sentiment != 0.0 OR away_team_sentiment != 0.0)
                ORDER BY features_computed_at DESC
                LIMIT 10
            """)
            
            results = session.execute(query).fetchall()
            
            if results:
                print(f"✅ Found {len(results)} fixtures with non-zero sentiment values:")
                print()
                for row in results:
                    print(f"Fixture {row.fixture_id}:")
                    print(f"  Home sentiment: {row.home_team_sentiment:.3f}")
                    print(f"  Away sentiment: {row.away_team_sentiment:.3f}")
                    print(f"  Computed at: {row.features_computed_at}")
                    print()
            else:
                print("❌ No fixtures found with non-zero sentiment values")
                
                # Check total count of recent features
                count_query = text("""
                    SELECT COUNT(*) as total_count
                    FROM pre_computed_features 
                    WHERE features_computed_at >= NOW() - INTERVAL '1 day'
                """)
                
                count_result = session.execute(count_query).fetchone()
                print(f"Total features computed in last 24 hours: {count_result.total_count if count_result else 0}")
                
                # Check a few recent entries
                recent_query = text("""
                    SELECT 
                        fixture_id,
                        home_team_sentiment,
                        away_team_sentiment,
                        features_computed_at
                    FROM pre_computed_features 
                    ORDER BY features_computed_at DESC
                    LIMIT 5
                """)
                
                recent_results = session.execute(recent_query).fetchall()
                
                if recent_results:
                    print("\nMost recent 5 entries:")
                    for row in recent_results:
                        home_sentiment = row.home_team_sentiment if row.home_team_sentiment is not None else 0.0
                        away_sentiment = row.away_team_sentiment if row.away_team_sentiment is not None else 0.0
                        print(f"Fixture {row.fixture_id}: home={home_sentiment:.3f}, away={away_sentiment:.3f}")
                        
    except Exception as e:
        print(f"❌ Error checking sentiment values: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_sentiment_populated()