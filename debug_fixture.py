#!/usr/bin/env python3
"""
Debug script to check fixture data
"""

from formfinder.enhanced_feature_computer import EnhancedFeatureComputer
from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text

def debug_fixture():
    """Debug fixture data"""
    print("Loading configuration...")
    load_config()
    
    print("Getting database session...")
    with get_db_session() as session:
        print("Initializing EnhancedFeatureComputer...")
        computer = EnhancedFeatureComputer(session)
        
        print("Finding test fixture...")
        result = session.execute(text("SELECT id FROM fixtures LIMIT 1")).fetchall()
        
        if result:
            fixture_id = result[0][0]
            print(f"Getting fixture details for fixture {fixture_id}...")
            
            fixture = computer._get_fixture_details(fixture_id)
            print(f"Fixture data: {fixture}")
            print(f"Fixture type: {type(fixture)}")
            
            if fixture:
                print(f"home_team_id: {fixture['home_team_id']} (type: {type(fixture['home_team_id'])})")
                print(f"away_team_id: {fixture['away_team_id']} (type: {type(fixture['away_team_id'])})")
                print(f"league_id: {fixture['league_id']} (type: {type(fixture['league_id'])})")
                print(f"match_date: {fixture['match_date']} (type: {type(fixture['match_date'])})")
                
                # Test markov feature computation
                print("\nTesting markov feature computation...")
                try:
                    print(f"About to call _compute_markov_features with fixture: {fixture}")
                    print(f"Home team ID: {fixture['home_team_id']}, Away team ID: {fixture['away_team_id']}, League ID: {fixture['league_id']}")

                    # Test the markov generator directly
                    print("\nTesting markov generator directly...")
                    try:
                        home_markov = computer.markov_generator.generate_team_features(
                            fixture['home_team_id'],
                            fixture['league_id'], 
                            fixture['match_date'],
                            'home'
                        )
                        print(f"Home markov features: {len(home_markov)} features")
                        print(f"Home context in features: {home_markov.get('home_away_context', 'NOT_FOUND')}")
                        print(f"Home markov feature keys: {list(home_markov.keys())}")
                        print("\nHome markov feature values and types:")
                        for key, value in home_markov.items():
                            print(f"  {key}: {value} (type: {type(value)})")
                    except Exception as e:
                        print(f"Error in home markov generation: {e}")
                        import traceback
                        traceback.print_exc()

                    try:
                        away_markov = computer.markov_generator.generate_team_features(
                            fixture['away_team_id'],
                            fixture['league_id'],
                            fixture['match_date'], 
                            'away'
                        )
                        print(f"Away markov features: {len(away_markov)} features")
                        print(f"Away context in features: {away_markov.get('home_away_context', 'NOT_FOUND')}")
                        print("\nAway markov feature values and types:")
                        for key, value in away_markov.items():
                            print(f"  {key}: {value} (type: {type(value)})")
                    except Exception as e:
                        print(f"Error in away markov generation: {e}")
                        import traceback
                        traceback.print_exc()

                    print("\nTesting full _compute_markov_features method...")
                    markov_features = computer._compute_markov_features(fixture)
                    print(f"Markov features computed successfully: {len(markov_features)} features")
                    print("\nFinal markov feature values and types:")
                    for key, value in markov_features.items():
                        print(f"  {key}: {value} (type: {type(value)})")
                        if isinstance(value, str) and key not in ['feature_date', 'home_away_context']:
                            print(f"    WARNING: String value found for {key}: {value}")
                except Exception as e:
                    print(f"Error in markov computation: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("No fixtures found in database")

if __name__ == "__main__":
    debug_fixture()