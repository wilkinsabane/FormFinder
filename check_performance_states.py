from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def check_performance_states():
    # Load configuration first
    load_config()
    with get_db_session() as session:
        try:
            # Check if table exists
            result = session.execute(text("SELECT COUNT(*) FROM team_performance_states"))
            count = result.scalar()
            print(f"team_performance_states table has {count} records")
            
            if count > 0:
                # Show sample data
                sample = session.execute(text("""
                    SELECT team_id, league_id, performance_state, state_date, home_away_context
                    FROM team_performance_states 
                    LIMIT 5
                """)).fetchall()
                print("\nSample data:")
                for row in sample:
                    print(f"  Team {row[0]}, League {row[1]}, State: {row[2]}, Date: {row[3]}, Context: {row[4]}")
        except Exception as e:
            print(f"Error checking team_performance_states: {e}")
            
            # Check if table exists at all
            try:
                tables = session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE '%performance%'
                """)).fetchall()
                print(f"\nTables with 'performance' in name: {[t[0] for t in tables]}")
            except Exception as e2:
                print(f"Error checking table existence: {e2}")

if __name__ == "__main__":
    check_performance_states()