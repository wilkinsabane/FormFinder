from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_table_schema():
    load_config()
    
    with get_db_session() as session:
        # Check if table exists
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'pre_computed_features'
        """))
        
        if result.fetchone():
            print("Table 'pre_computed_features' exists")
            
            # Get column names
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'pre_computed_features' 
                ORDER BY ordinal_position
            """))
            
            print("\nColumns in pre_computed_features:")
            for row in result:
                print(f"- {row[0]} ({row[1]})")
                
            # Check sample data
            result = session.execute(text("""
                SELECT * FROM pre_computed_features 
                LIMIT 3
            """))
            
            print("\nSample data (first 3 rows):")
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                print(f"Columns: {list(columns)}")
                for i, row in enumerate(rows):
                    print(f"Row {i+1}: {dict(zip(columns, row))}")
            else:
                print("No data found")
        else:
            print("Table 'pre_computed_features' does not exist")

if __name__ == "__main__":
    check_table_schema()