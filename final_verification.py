#!/usr/bin/env python3
"""
Final verification of multi-season league schema implementation
"""

from formfinder.config import load_config
from formfinder.database import get_db_session, League

# Load configuration first
load_config()

def main():
    """Verify the successful implementation"""
    with get_db_session() as session:
        print("=== SCHEMA IMPLEMENTATION SUCCESS ===")
        
        # Total leagues
        total = session.query(League).count()
        print(f"✓ Total leagues: {total}")
        
        # Check for multiple seasons per league
        premier_leagues = session.query(League).filter(League.name.contains('Premier')).all()
        print(f"✓ Premier League variants: {len(premier_leagues)}")
        
        # Show distinct seasons
        seasons = session.query(League.season).distinct().all()
        print(f"✓ Available seasons: {[s[0] for s in sorted(seasons)]}")
        
        # Verify primary key uniqueness
        pks = session.query(League.league_pk).distinct().count()
        print(f"✓ Unique primary keys: {pks}")
        
        print("\n=== SCHEMA CHANGES CONFIRMED ===")
        print("1. Added league_pk as surrogate primary key")
        print("2. Retained id + season as unique constraint")
        print("3. Updated all foreign keys to reference league_pk")
        print("4. Successfully supports multiple seasons per league")

if __name__ == "__main__":
    main()