#!/usr/bin/env python3

import sys
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_latest_predictions():
    """Check the latest predictions in the database."""
    load_config()
    
    with get_db_session() as session:
        # Get latest predictions by updated_at
        result = session.execute(text("""
            SELECT fixture_id, predicted_total_goals, over_2_5_probability, 
                   updated_at, created_at
            FROM predictions 
            ORDER BY updated_at DESC 
            LIMIT 10
        """)).fetchall()
        
        print("Latest predictions by updated_at:")
        for r in result:
            print(f"Fixture {r[0]}: Goals={r[1]:.2f}, Over2.5={r[2]:.3f}, Updated={r[3]}, Created={r[4]}")
        
        # Check if we have any predictions from today
        today_result = session.execute(text("""
            SELECT COUNT(*) 
            FROM predictions 
            WHERE DATE(updated_at) = CURRENT_DATE
        """)).fetchone()
        
        print(f"\nPredictions updated today: {today_result[0]}")
        
        # Check specific fixture IDs from our recent run
        specific_result = session.execute(text("""
            SELECT fixture_id, predicted_total_goals, over_2_5_probability, updated_at
            FROM predictions 
            WHERE fixture_id IN (1001, 1002, 1003, 1004, 1005)
            ORDER BY fixture_id
        """)).fetchall()
        
        print("\nSpecific fixtures from recent run:")
        for r in specific_result:
            print(f"Fixture {r[0]}: Goals={r[1]:.2f}, Over2.5={r[2]:.3f}, Updated={r[3]}")

if __name__ == "__main__":
    check_latest_predictions()