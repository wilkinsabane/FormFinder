#!/usr/bin/env python3

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

# Load configuration
load_config()

def check_predictions_table():
    """Check if predictions table exists and show its structure."""
    with get_db_session() as session:
        # Check if predictions table exists
        result = session.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'predictions'"
        ))
        table_exists = bool(result.fetchone())
        print(f"Predictions table exists: {table_exists}")
        
        if table_exists:
            # Show table structure
            result = session.execute(text(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'predictions' ORDER BY ordinal_position"
            ))
            columns = result.fetchall()
            print("\nTable structure:")
            for col in columns:
                print(f"  {col[0]}: {col[1]}")
            
            # Check if there are any predictions
            result = session.execute(text("SELECT COUNT(*) FROM predictions"))
            count = result.fetchone()[0]
            print(f"\nNumber of predictions in table: {count}")
            
            if count > 0:
                # Show sample predictions
                result = session.execute(text(
                    "SELECT fixture_id, predicted_total_goals, over_2_5_probability, created_at FROM predictions ORDER BY created_at DESC LIMIT 5"
                ))
                predictions = result.fetchall()
                print("\nSample predictions (latest 5):")
                for pred in predictions:
                    print(f"  Fixture {pred[0]}: Goals={pred[1]:.2f}, Over2.5={pred[2]:.3f}, Created={pred[3]}")

if __name__ == "__main__":
    check_predictions_table()