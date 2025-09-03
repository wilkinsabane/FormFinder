#!/usr/bin/env python3
"""
Validation script for the Football Data Dashboard
Tests database connection and basic dashboard functionality
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

def test_database_connection():
    """Test database connection and basic table structure."""
    print("🔍 Testing database connection...")
    
    # Try to get database URI
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        try:
            # Try to import from secrets if running in Streamlit context
            import streamlit as st
            if hasattr(st, 'secrets'):
                db_uri = st.secrets.get("DB_URI")
        except ImportError:
            pass
    
    if not db_uri:
        print("❌ DB_URI not found in environment variables or Streamlit secrets")
        print("   Please set DB_URI environment variable or configure .streamlit/secrets.toml")
        return False
    
    try:
        engine = create_engine(db_uri)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        
        # Test table existence
        required_tables = [
            'teams', 'standings', 'predictions', 
            'leagues', 'high_form_teams', 'fixtures', 'data_fetch_logs'
        ]
        
        for table in required_tables:
            try:
                result = pd.read_sql_query(
                    text(f"SELECT COUNT(*) as count FROM {table}"), 
                    engine
                )
                count = result.iloc[0]['count']
                print(f"✅ {table}: {count} records")
            except Exception as e:
                print(f"❌ {table}: {str(e)}")
        
        return True
        
    except SQLAlchemyError as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

def test_streamlit_import():
    """Test if Streamlit and required dependencies are installed."""
    print("\n🔍 Testing Streamlit dependencies...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit not installed. Run: pip install streamlit")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError:
        print("❌ Plotly not installed. Run: pip install plotly")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError:
        print("❌ Pandas not installed. Run: pip install pandas")
        return False
    
    return True

def check_configuration():
    """Check configuration files."""
    print("\n🔍 Checking configuration files...")
    
    # Check .streamlit directory
    if os.path.exists(".streamlit"):
        print("✅ .streamlit directory exists")
    else:
        print("❌ .streamlit directory not found")
    
    # Check config.toml
    if os.path.exists(".streamlit/config.toml"):
        print("✅ .streamlit/config.toml exists")
    else:
        print("❌ .streamlit/config.toml not found")
    
    # Check secrets.toml
    if os.path.exists(".streamlit/secrets.toml"):
        print("✅ .streamlit/secrets.toml exists")
    else:
        print("⚠️  .streamlit/secrets.toml not found (copy from .streamlit/secrets.toml.example)")
    
    # Check dashboard.py
    if os.path.exists("dashboard.py"):
        print("✅ dashboard.py exists")
    else:
        print("❌ dashboard.py not found")

def main():
    """Run all validation tests."""
    print("🚀 Football Data Dashboard Validation")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_streamlit_import()
    
    # Test configuration
    check_configuration()
    
    # Test database connection
    db_ok = test_database_connection()
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    
    if imports_ok and db_ok:
        print("✅ All tests passed! Dashboard is ready to run.")
        print("\nTo start the dashboard:")
        print("  Windows: run_dashboard.bat")
        print("  Unix: ./run_dashboard.sh")
        print("  Manual: streamlit run dashboard.py")
    else:
        print("❌ Some tests failed. Please check the messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()