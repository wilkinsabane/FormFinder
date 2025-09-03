import sqlite3
import os

def test_database_setup():
    """Test that the database migration was successful."""
    db_path = "data/formfinder.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if h2h_cache table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='h2h_cache'")
        h2h_exists = cursor.fetchone() is not None
        print(f"h2h_cache table exists: {h2h_exists}")
        
        # Check if predictions table has new columns
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        required_columns = ['predicted_total_goals', 'over_2_5_probability', 'under_2_5_probability']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"Missing columns in predictions table: {missing_columns}")
            return False
        else:
            print("All required columns exist in predictions table")
            return True
            
    except Exception as e:
        print(f"Error checking database: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    test_database_setup()