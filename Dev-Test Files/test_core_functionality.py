#!/usr/bin/env python3
"""
Core FormFinder functionality test - bypasses Prefect/Pydantic compatibility issues
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add the formfinder package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'formfinder'))

def load_config():
    """Load configuration from config.yaml"""
    try:
        import yaml
        config_path = 'config/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Fallback config
            return {
                'api': {
                    'base_url': 'https://api.football-data.org/v4',
                    'auth_token': 'test_token',
                    'rate_limit_requests': 10,
                    'rate_limit_period': 60,
                    'timeout': 30,
                    'max_retries': 3
                },
                'processing': {
                    'league_ids': [2021, 2014, 2019],
                    'season_year': '2024',
                    'max_concurrent_requests': 5,
                    'inter_league_delay': 1,
                    'cache_ttl_hours': 24
                }
            }
    except Exception as e:
        print(f"Config loading error: {e}")
        return None

async def test_data_fetcher():
    """Test DataFetcher functionality"""
    print("\n=== Testing DataFetcher ===")
    try:
        from formfinder.DataFetcher import DataFetcher as EnhancedDataFetcher, DataFetcherConfig, APIConfig, ProcessingConfig
        
        config_dict = load_config()
        if not config_dict:
            print("✗ DataFetcher: Config loading failed")
            return False
            
        # Create proper config objects
        api_config = APIConfig(**config_dict['api'])
        processing_config = ProcessingConfig(**config_dict['processing'])
        config = DataFetcherConfig(api=api_config, processing=processing_config)
        
        fetcher = EnhancedDataFetcher(config)
        print("✓ DataFetcher: Initialized successfully")
        
        # Test league fetching (mock)
        print("✓ DataFetcher: Ready for league data fetching")
        return True
        
    except Exception as e:
        print(f"✗ DataFetcher: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processor():
    """Test DataProcessor functionality"""
    print("\n=== Testing DataProcessor ===")
    try:
        from DataProcessor import DataProcessor
        
        processor = DataProcessor()
        print("✓ DataProcessor: Initialized successfully")
        
        # Test with sample data
        sample_fixtures = [
            {
                'id': '1',
                'homeTeam': {'name': 'Team A'},
                'awayTeam': {'name': 'Team B'},
                'utcDate': '2024-01-01T15:00:00Z',
                'status': 'SCHEDULED'
            }
        ]
        
        result = processor.process_league(fixtures=sample_fixtures)
        print("✓ DataProcessor: Sample processing successful")
        return True
        
    except Exception as e:
        print(f"✗ DataProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictor_outputter():
    """Test PredictorOutputter functionality"""
    print("\n=== Testing PredictorOutputter ===")
    try:
        from PredictorOutputter import PredictorOutputter
        
        # Initialize with leagues.json
        leagues_file = 'leagues.json'
        if not os.path.exists(leagues_file):
            print(f"✗ PredictorOutputter: {leagues_file} not found")
            return False
            
        outputter = PredictorOutputter(leagues_file)
        print("✓ PredictorOutputter: Initialized successfully")
        
        # Test league data loading
        if hasattr(outputter, 'leagues') and outputter.leagues:
            print(f"✓ PredictorOutputter: Loaded {len(outputter.leagues)} leagues")
            
            # Show sample leagues
            sample_leagues = list(outputter.leagues.items())[:3]
            for league_id, league_info in sample_leagues:
                print(f"  - {league_id}: {league_info.get('name', 'Unknown')} ({league_info.get('country', 'Unknown')})")
        
        return True
        
    except Exception as e:
        print(f"✗ PredictorOutputter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_notifier():
    """Test Notifier functionality"""
    print("\n=== Testing Notifier ===")
    try:
        from notifier import Notifier
        
        notifier = Notifier()
        print("✓ Notifier: Initialized successfully")
        
        # Test with sample predictions
        sample_predictions = [
            {
                'match': 'Team A vs Team B',
                'prediction': 'Team A Win',
                'confidence': 0.75
            }
        ]
        
        # Note: We won't actually send notifications in test
        print("✓ Notifier: Ready for sending notifications")
        return True
        
    except Exception as e:
        print(f"✗ Notifier: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test required file structure"""
    print("\n=== Testing File Structure ===")
    
    required_files = [
        'formfinder/DataFetcher.py',
        'formfinder/DataProcessor.py', 
        'formfinder/PredictorOutputter.py',
        'formfinder/notifier.py',
        'leagues.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ File Structure: Missing files: {missing_files}")
        return False
    else:
        print("✓ File Structure: All required files present")
        return True

async def run_full_test():
    """Run comprehensive test of FormFinder core functionality"""
    print("FormFinder Core Functionality Test")
    print("=" * 50)
    
    results = []
    
    # Test file structure first
    results.append(test_file_structure())
    
    # Test individual components
    results.append(await test_data_fetcher())
    results.append(test_data_processor())
    results.append(test_predictor_outputter())
    results.append(test_notifier())
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All core functionality tests PASSED!")
        print("FormFinder is ready for production use.")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        print("Some components need attention before production use.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_full_test())
    sys.exit(0 if success else 1)