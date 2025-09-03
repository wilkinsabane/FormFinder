#!/usr/bin/env python3
"""
Check for Markov columns in pre_computed_features table.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def main():
    # Load configuration first
    load_config()
    with get_db_session() as session:
        # Check for Markov columns
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'pre_computed_features' 
            AND column_name LIKE '%markov%' 
            ORDER BY column_name
        """))
        
        markov_cols = [row[0] for row in result.fetchall()]
        
        print("Markov columns in database:")
        if markov_cols:
            for col in markov_cols:
                print(f"  {col}")
        else:
            print("  No Markov columns found!")
        
        print(f"\nTotal Markov columns: {len(markov_cols)}")
        
        # Also check for sentiment columns
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'pre_computed_features' 
            AND column_name LIKE '%sentiment%' 
            ORDER BY column_name
        """))
        
        sentiment_cols = [row[0] for row in result.fetchall()]
        
        print("\nSentiment columns in database:")
        if sentiment_cols:
            for col in sentiment_cols:
                print(f"  {col}")
        else:
            print("  No sentiment columns found!")
        
        print(f"\nTotal sentiment columns: {len(sentiment_cols)}")

if __name__ == "__main__":
    main()