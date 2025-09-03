#!/usr/bin/env python3
"""
Check markov_outcome_probabilities values in pre_computed_features table.
"""

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def check_markov_outcomes():
    """Check distinct markov_outcome_probabilities values."""
    # Load configuration first
    load_config()
    with get_db_session() as session:
        try:
            # Check distinct values
            query = """
            SELECT DISTINCT markov_outcome_probabilities 
            FROM pre_computed_features 
            WHERE markov_outcome_probabilities IS NOT NULL 
            LIMIT 10
            """
            
            result = session.execute(text(query))
            rows = result.fetchall()
            
            print("Distinct markov_outcome_probabilities values:")
            if rows:
                for i, row in enumerate(rows):
                    print(f"{i+1}: {row[0]}")
            else:
                print("No markov_outcome_probabilities values found!")
                
            # Check total count
            count_query = """
            SELECT COUNT(*) as total,
                   COUNT(markov_outcome_probabilities) as with_markov
            FROM pre_computed_features
            """
            
            count_result = session.execute(text(count_query))
            count_row = count_result.fetchone()
            
            print(f"\nTotal records: {count_row[0]}")
            print(f"Records with markov_outcome_probabilities: {count_row[1]}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_markov_outcomes()