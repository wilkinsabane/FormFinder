#!/usr/bin/env python3

from formfinder.database import get_db_session
from sqlalchemy import text

def test_enhanced_regeneration():
    """Delete some enhanced features and regenerate them to test if momentum/sentiment work."""
    with get_db_session() as session:
        # Get some fixture IDs to delete
        result = session.execute(text("""
            SELECT fixture_id FROM precomputed_features 
            WHERE computation_source = 'unified_enhanced' 
            LIMIT 5
        """))
        fixture_ids = [row[0] for row in result.fetchall()]
        
        if not fixture_ids:
            print("No enhanced features found to delete")
            return
        
        # Delete these records
        for fixture_id in fixture_ids:
            session.execute(text("""
                DELETE FROM precomputed_features 
                WHERE fixture_id = :fixture_id AND computation_source = 'unified_enhanced'
            """), {"fixture_id": fixture_id})
        
        session.commit()
        print(f"Deleted {len(fixture_ids)} enhanced feature records: {fixture_ids}")
        
        # Check remaining count
        result = session.execute(text("""
            SELECT COUNT(*) FROM precomputed_features 
            WHERE computation_source = 'unified_enhanced'
        """))
        remaining = result.fetchone()[0]
        print(f"Remaining enhanced features: {remaining}")

if __name__ == "__main__":
    test_enhanced_regeneration()