#!/usr/bin/env python3
"""
FormFinder Main Application
Runs the complete football prediction pipeline without Prefect dependency
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add the formfinder package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'formfinder'))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/formfinder_main.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('FormFinder')

def load_config():
    """Load configuration from config.yaml"""
    try:
        import yaml
        config_path = 'config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except Exception as e:
        print(f"Config loading error: {e}")
        return None

async def fetch_data(config, logger):
    """Fetch data using DataFetcher"""
    logger.info("Starting data fetching phase...")
    
    try:
        from database_data_fetcher import DatabaseDataFetcher
        
        # Initialize components with database storage
        fetcher = DatabaseDataFetcher(use_database=True)
        
        # Load configuration using the new config system
        from formfinder.config import FormFinderConfig
        ff_config = FormFinderConfig.from_yaml('config.yaml')
        league_ids = ff_config.get_league_ids()
        
        logger.info(f"Initialized DataFetcher for {len(league_ids)} leagues")
        
        # Fetch data for all configured leagues
        results = {}
        for league_id in league_ids:
            logger.info(f"Fetching data for league {league_id}...")
            
            # Fetch fixtures
            fixtures = await fetcher.fetch_upcoming_fixtures_for_league(league_id, f"League-{league_id}")
            if fixtures:
                results[f'league_{league_id}_fixtures'] = fixtures
                logger.info(f"Fetched {len(fixtures)} fixtures for league {league_id}")
            
            # Fetch standings
            standings = await fetcher.fetch_standings_async(league_id, f"League-{league_id}")
            if standings:
                results[f'league_{league_id}_standings'] = standings
                logger.info(f"Fetched standings for league {league_id}")
            
            # Add delay between leagues
            if config['processing']['inter_league_delay'] > 0:
                await asyncio.sleep(config['processing']['inter_league_delay'])
        
        logger.info(f"Data fetching completed. Retrieved data for {len(results)} datasets.")
        return results
        
    except Exception as e:
        logger.error(f"Data fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_data(fetched_data, logger):
    """Process data using DataProcessor"""
    logger.info("Starting data processing phase...")
    
    try:
        from formfinder.DataProcessor import DataProcessor
        
        processor = DataProcessor()
        logger.info("Initialized DataProcessor")
        
        processed_results = {}
        
        # Process fixtures for each league
        for key, data in fetched_data.items():
            if 'fixtures' in key:
                league_id = key.split('_')[1]
                logger.info(f"Processing fixtures for league {league_id}...")
                
                processed_teams = processor.process_league(league_id=int(league_id), fixtures=data)
                processed_results[f'league_{league_id}_processed'] = processed_teams
                logger.info(f"Processed {len(processed_teams)} high-form teams for league {league_id}")
        
        logger.info(f"Data processing completed. Processed {len(processed_results)} datasets.")
        return processed_results
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_predictions(processed_data, logger):
    """Generate predictions using PredictorOutputter"""
    logger.info("Starting prediction generation phase...")
    
    try:
        from formfinder.PredictorOutputter import PredictorOutputter
        
        outputter = PredictorOutputter('leagues.json')
        logger.info(f"Initialized PredictorOutputter with {len(outputter.leagues_data)} leagues")
        
        all_predictions = []
        
        # Generate predictions for each processed league
        for key, data in processed_data.items():
            league_id = key.split('_')[1]
            logger.info(f"Generating predictions for league {league_id}...")
            
            # Create predictions based on high-form teams
            predictions = []
            if not data.empty:
                for _, team_row in data.iterrows():
                    prediction = {
                        'league_id': league_id,
                        'league_name': outputter.leagues_data.get(int(league_id), {}).get('name', 'Unknown'),
                        'country': outputter.leagues_data.get(int(league_id), {}).get('country', 'Unknown'),
                        'team_id': team_row.get('team_id', 'unknown'),
                        'team_name': team_row.get('team_name', 'Unknown'),
                        'win_rate': team_row.get('win_rate', 0.0),
                        'prediction_type': 'high_form_team',
                        'confidence': team_row.get('win_rate', 0.0),
                        'timestamp': datetime.now().isoformat()
                    }
                    predictions.append(prediction)
            
            all_predictions.extend(predictions)
            logger.info(f"Generated {len(predictions)} predictions for league {league_id}")
        
        # Save predictions to file
        predictions_file = 'data/predictions/latest_predictions.json'
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        logger.info(f"Prediction generation completed. Generated {len(all_predictions)} total predictions.")
        return all_predictions
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def send_notifications(predictions, logger):
    """Send notifications using Notifier"""
    logger.info("Starting notification phase...")
    
    try:
        from formfinder.notifier import Notifier
        
        notifier = Notifier()
        logger.info("Initialized Notifier")
        
        # In a real implementation, this would send actual notifications
        # For now, we'll just log the predictions
        logger.info(f"Would send notifications for {len(predictions)} predictions")
        
        # Show sample predictions
        for i, pred in enumerate(predictions[:5]):  # Show first 5
            logger.info(f"Prediction {i+1}: {pred['home_team']} vs {pred['away_team']} - {pred['prediction']} ({pred['confidence']:.2f})")
        
        logger.info("Notification phase completed.")
        return True
        
    except Exception as e:
        logger.error(f"Notification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_formfinder_pipeline():
    """Run the complete FormFinder pipeline"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("STARTING FORMFINDER PIPELINE")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            return False
        logger.info("Configuration loaded successfully")
        
        # Create necessary directories
        os.makedirs('data/logs', exist_ok=True)
        os.makedirs('data/predictions', exist_ok=True)
        
        # Phase 1: Data Fetching
        fetched_data = await fetch_data(config, logger)
        if not fetched_data:
            logger.error("Data fetching failed. Exiting.")
            return False
        
        # Phase 2: Data Processing
        processed_data = process_data(fetched_data, logger)
        if not processed_data:
            logger.error("Data processing failed. Exiting.")
            return False
        
        # Phase 3: Prediction Generation
        predictions = generate_predictions(processed_data, logger)
        if not predictions:
            logger.error("Prediction generation failed. Exiting.")
            return False
        
        # Phase 4: Notifications
        notification_success = send_notifications(predictions, logger)
        if not notification_success:
            logger.warning("Notification phase had issues, but continuing...")
        
        # Pipeline completed successfully
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("FORMFINDER PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the FormFinder pipeline."""
    print("FormFinder - Football Prediction Pipeline")
    print("=========================================")
    
    try:
        success = asyncio.run(run_formfinder_pipeline())
        if success:
            print("\n🎉 FormFinder pipeline completed successfully!")
            print("Check data/predictions/latest_predictions.json for results.")
            sys.exit(0)
        else:
            print("\n❌ FormFinder pipeline failed.")
            print("Check data/logs/formfinder_main.log for details.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()