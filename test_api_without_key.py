from formfinder.config import load_config, get_config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from formfinder.exceptions import APIError

def test_api_without_key():
    """Test what happens when API client tries to fetch H2H data without API key."""
    print("=== Testing API Client Without Key ===")
    
    # Load config
    load_config()
    config = get_config()
    
    # Database connection
    engine = create_engine(config.get_database_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Initialize API client
        from formfinder.clients.api_client import SoccerDataAPIClient
        api_client = SoccerDataAPIClient(session)
        print(f"API token configured: {'YES' if config.api.auth_token else 'NO (empty)'}")
        print(f"API token length: {len(config.api.auth_token)}")
        
        # Try to fetch H2H data for sample teams
        print("\nTrying to fetch H2H data for teams 2689 vs 2693...")
        try:
            h2h_data = api_client.get_h2h_stats(2689, 2693)
            print(f"✓ H2H data fetched successfully: {h2h_data}")
        except APIError as e:
            print(f"❌ API Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            
    except Exception as e:
        print(f"❌ Failed to initialize or test API client: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    test_api_without_key()