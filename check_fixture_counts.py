#!/usr/bin/env python3

import sys
sys.path.append('.')

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def check_fixture_counts():
    """Check fixture counts for debugging prediction numbers."""
    load_config()
    
    with get_db_session() as session:
        # Load free leagues
        with open('free_leagues.txt', 'r') as f:
            leagues = [int(line.strip()) for line in f if line.strip().isdigit()]
        
        print(f"Free leagues count: {len(leagues)}")
        print(f"Free leagues: {leagues}")
        
        # Total upcoming fixtures
        result = session.execute(text("""
            SELECT COUNT(*) 
            FROM fixtures 
            WHERE status != 'finished' 
            AND match_date > NOW()
        """)).fetchone()
        print(f"Total upcoming fixtures: {result[0]}")
        
        # Upcoming fixtures in free leagues
        result2 = session.execute(text("""
            SELECT COUNT(*) 
            FROM fixtures 
            WHERE status != 'finished' 
            AND match_date > NOW() 
            AND league_id = ANY(:leagues)
        """), {"leagues": leagues}).fetchone()
        print(f"Upcoming fixtures in free leagues: {result2[0]}")
        
        # Check what tables exist
        result3 = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%feature%'
        """)).fetchall()
        print(f"Feature-related tables: {[r[0] for r in result3]}")
        
        # Check all tables
        result4 = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)).fetchall()
        print(f"All tables: {[r[0] for r in result4]}")
        
        # Check if there are any feature-related tables with data
        feature_tables = [r[0] for r in result3]
        if feature_tables:
            for table in feature_tables:
                try:
                    count_result = session.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    print(f"Records in {table}: {count_result[0]}")
                except Exception as e:
                    print(f"Error checking {table}: {e}")

if __name__ == "__main__":
    check_fixture_counts()