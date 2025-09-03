from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

def test_h2h_cache():
    """Test H2H cache functionality comprehensively."""
    load_config()
    engine = create_engine(get_config().get_database_url())
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        # 1. Check h2h_cache table structure
        print("=== H2H Cache Table Structure ===")
        result = session.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'h2h_cache' 
            ORDER BY ordinal_position
        """))
        
        columns = result.fetchall()
        if not columns:
            print("❌ h2h_cache table does not exist!")
            return False
        
        print("H2H cache table columns:")
        for col in columns:
            print(f"  {col.column_name}: {col.data_type} ({'nullable' if col.is_nullable == 'YES' else 'not null'})")
        
        # 2. Check current cache entries
        print("\n=== Current H2H Cache Status ===")
        result = session.execute(text("SELECT COUNT(*) as count FROM h2h_cache"))
        count = result.scalar()
        print(f"Total H2H cache entries: {count}")
        
        if count > 0:
            # Show sample entries
            result = session.execute(text("SELECT * FROM h2h_cache LIMIT 3"))
            print("\nSample H2H cache entries:")
            for row in result:
                print(f"  {dict(row._mapping)}")
        
        # 3. Test API client initialization
        print("\n=== Testing API Client ===")
        try:
            from formfinder.clients.api_client import SoccerDataAPIClient
            api_client = SoccerDataAPIClient(session)
            print("✓ API client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize API client: {e}")
            return False
        
        # 4. Test H2H data fetching (without actually calling API)
        print("\n=== Testing H2H Cache Logic ===")
        
        # Check if we have some teams to test with
        result = session.execute(text("""
            SELECT DISTINCT home_team_id, away_team_id 
            FROM fixtures 
            WHERE home_team_id IS NOT NULL AND away_team_id IS NOT NULL
            LIMIT 1
        """))
        
        test_match = result.fetchone()
        if test_match:
            team1_id = test_match.home_team_id
            team2_id = test_match.away_team_id
            print(f"Testing with teams: {team1_id} vs {team2_id}")
            
            # Check if this combination exists in cache
            result = session.execute(text("""
                SELECT * FROM h2h_cache 
                WHERE team1_id = :team1 AND team2_id = :team2
            """), {"team1": team1_id, "team2": team2_id})
            
            cached_entry = result.fetchone()
            if cached_entry:
                print("✓ Found cached H2H data for this team combination")
                print(f"  Cache entry: {dict(cached_entry._mapping)}")
            else:
                print("ℹ️ No cached H2H data for this team combination (expected for first run)")
        else:
            print("❌ No fixture data found to test with")
        
        # 5. Check API configuration
        print("\n=== API Configuration Check ===")
        config = get_config()
        api_token = os.getenv("SOCCERDATA_API_KEY")
        if api_token:
            print(f"✓ API token configured (length: {len(api_token)})")
        else:
            print("⚠️ No API token found in environment (SOCCERDATA_API_KEY)")
            print("  H2H data fetching will fail without a valid API token")
        
        print(f"API base URL: {config.api.base_url}")
        print(f"H2H TTL: {config.api.h2h_ttl_seconds} seconds")
        
        return True

def test_manual_h2h_insert():
    """Test manual insertion into h2h_cache to verify table works."""
    load_config()
    engine = create_engine(get_config().get_database_url())
    Session = sessionmaker(bind=engine)
    
    print("\n=== Testing Manual H2H Cache Insert ===")
    
    with Session() as session:
        try:
            # Insert a test entry
            session.execute(text("""
                INSERT INTO h2h_cache (
                    team1_id, team2_id, competition_id,
                    overall_games_played, overall_team1_wins, overall_team2_wins, overall_draws,
                    overall_team1_scored, overall_team2_scored,
                    team1_games_played_at_home, team1_wins_at_home, team1_losses_at_home, team1_draws_at_home,
                    team1_scored_at_home, team1_conceded_at_home,
                    team2_games_played_at_home, team2_wins_at_home, team2_losses_at_home, team2_draws_at_home,
                    team2_scored_at_home, team2_conceded_at_home,
                    avg_total_goals, last_fetched_at
                ) VALUES (
                    9999, 9998, 1,
                    5, 3, 1, 1,
                    8, 3,
                    3, 2, 0, 1,
                    5, 1,
                    2, 1, 1, 0,
                    3, 2,
                    2.2, NOW()
                )
            """))
            session.commit()
            print("✓ Successfully inserted test H2H cache entry")
            
            # Verify the insert
            result = session.execute(text("""
                SELECT COUNT(*) FROM h2h_cache WHERE team1_id = 9999 AND team2_id = 9998
            """))
            count = result.scalar()
            print(f"✓ Verified: {count} test entry found in cache")
            
            # Clean up test entry
            session.execute(text("""
                DELETE FROM h2h_cache WHERE team1_id = 9999 AND team2_id = 9998
            """))
            session.commit()
            print("✓ Test entry cleaned up")
            
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Failed to insert test H2H cache entry: {e}")
            return False

if __name__ == "__main__":
    print("Testing H2H Cache Functionality...\n")
    
    success1 = test_h2h_cache()
    success2 = test_manual_h2h_insert()
    
    print("\n=== Test Summary ===")
    if success1 and success2:
        print("✅ All H2H cache tests passed!")
        print("\nNext steps:")
        print("1. Ensure SOCCERDATA_API_KEY is set in environment")
        print("2. Run prediction script to test actual H2H data fetching")
        print("3. Check cache after prediction run")
    else:
        print("❌ Some H2H cache tests failed")
        print("Please check the errors above and fix the issues")