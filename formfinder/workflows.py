"""Prefect workflow orchestration for FormFinder pipeline."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.system import Secret
from sqlalchemy.orm import Session

from .config import FormFinderConfig, get_config
from .database import get_db_session, DataFetchLog, init_database
from .DataFetcher import DataFetcher, DataFetcherConfig
from .DataProcessor import DataProcessor
from .PredictorOutputter import PredictorOutputter
from .notifier import Notifier

logger = logging.getLogger(__name__)


@task(retries=3, retry_delay_seconds=60)
def fetch_league_data(league_id: int, data_fetcher: DataFetcher) -> dict:
    """Fetch data for a single league."""
    task_logger = get_run_logger()
    task_logger.info(f"Fetching data for league {league_id}")
    
    start_time = datetime.utcnow()
    
    try:
        # Fetch fixtures
        fixtures = data_fetcher.fetch_fixtures(league_id)
        task_logger.info(f"Fetched {len(fixtures)} fixtures for league {league_id}")
        
        # Fetch standings
        standings = data_fetcher.fetch_standings(league_id)
        task_logger.info(f"Fetched {len(standings)} standings for league {league_id}")
        
        # Fetch teams
        teams = data_fetcher.fetch_teams(league_id)
        task_logger.info(f"Fetched {len(teams)} teams for league {league_id}")
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Log successful fetch
        with get_db_session() as session:
            log_entry = DataFetchLog(
                league_id=league_id,
                data_type='complete',
                status='success',
                records_fetched=len(fixtures) + len(standings) + len(teams),
                duration_seconds=duration
            )
            session.add(log_entry)
            session.commit()
        
        return {
            'league_id': league_id,
            'fixtures': fixtures,
            'standings': standings,
            'teams': teams,
            'status': 'success'
        }
        
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        task_logger.error(f"Failed to fetch data for league {league_id}: {str(e)}")
        
        # Log failed fetch
        with get_db_session() as session:
            log_entry = DataFetchLog(
                league_id=league_id,
                data_type='complete',
                status='error',
                records_fetched=0,
                error_message=str(e),
                duration_seconds=duration
            )
            session.add(log_entry)
            session.commit()
        
        return {
            'league_id': league_id,
            'status': 'error',
            'error': str(e)
        }


@task(retries=2, retry_delay_seconds=30)
def process_league_data(league_data: dict, data_processor: DataProcessor) -> dict:
    """Process data for a single league."""
    task_logger = get_run_logger()
    league_id = league_data['league_id']
    
    if league_data['status'] != 'success':
        task_logger.warning(f"Skipping processing for league {league_id} due to fetch error")
        return {'league_id': league_id, 'status': 'skipped', 'reason': 'fetch_failed'}
    
    task_logger.info(f"Processing data for league {league_id}")
    
    try:
        # Process the league data
        processed_data = data_processor.process_league(
            league_id=league_id,
            fixtures=league_data['fixtures'],
            standings=league_data['standings'],
            teams=league_data['teams']
        )
        
        task_logger.info(f"Successfully processed data for league {league_id}")
        
        return {
            'league_id': league_id,
            'processed_data': processed_data,
            'status': 'success'
        }
        
    except Exception as e:
        task_logger.error(f"Failed to process data for league {league_id}: {str(e)}")
        return {
            'league_id': league_id,
            'status': 'error',
            'error': str(e)
        }


@task(retries=2, retry_delay_seconds=30)
def generate_predictions_task(processed_data_list: List[dict], predictor_outputter: PredictorOutputter) -> dict:
    """Generate predictions from processed data."""
    task_logger = get_run_logger()
    task_logger.info("Generating predictions from processed data")
    
    try:
        # Filter successful processing results
        successful_data = [
            data for data in processed_data_list 
            if data['status'] == 'success'
        ]
        
        if not successful_data:
            task_logger.warning("No successful data processing results to generate predictions from")
            return {'status': 'skipped', 'reason': 'no_data'}
        
        # Generate predictions
        predictions = predictor_outputter.generate_predictions(successful_data)
        
        task_logger.info(f"Generated {len(predictions)} predictions")
        
        return {
            'predictions': predictions,
            'status': 'success',
            'count': len(predictions)
        }
        
    except Exception as e:
        task_logger.error(f"Failed to generate predictions: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


@task(retries=3, retry_delay_seconds=60)
def send_notifications(predictions_result: dict, notifier: Notifier) -> dict:
    """Send notifications with predictions."""
    task_logger = get_run_logger()
    
    if predictions_result['status'] != 'success':
        task_logger.warning("Skipping notifications due to prediction generation failure")
        return {'status': 'skipped', 'reason': 'prediction_failed'}
    
    task_logger.info("Sending notifications")
    
    try:
        # Send notifications
        notification_result = notifier.send_notifications(predictions_result['predictions'])
        
        task_logger.info("Notifications sent successfully")
        
        return {
            'status': 'success',
            'notification_result': notification_result
        }
        
    except Exception as e:
        task_logger.error(f"Failed to send notifications: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


@flow(
    name="FormFinder Data Pipeline",
    description="Complete FormFinder data fetching, processing, and prediction pipeline",
    task_runner=ConcurrentTaskRunner()
)
def run_main_pipeline(
    config_path: str = "config.yaml",
    league_ids: Optional[List[int]] = None,
    force_refresh: bool = False,
    skip_notifications: bool = False
) -> dict:
    """Main FormFinder pipeline workflow."""
    flow_logger = get_run_logger()

    # Load config from path
    from .config import load_config
    config = load_config(config_path)
    
    if league_ids is None:
        league_ids = config.get_league_ids()
    
    flow_logger.info(f"Starting FormFinder pipeline for {len(league_ids)} leagues")
    
    # Initialize database with config
    init_database(config.get_database_url())
    
    # Initialize components
    # Create DataFetcherConfig from FormFinderConfig
    # Ensure league_ids are properly set in processing config
    processing_data = config.processing.model_dump()
    if processing_data.get('league_ids') is None:
        processing_data['league_ids'] = league_ids
    
    data_fetcher_config = DataFetcherConfig(
        api=config.api.model_dump(),
        processing=processing_data
    )
    data_fetcher = DataFetcher(data_fetcher_config)
    data_processor = DataProcessor()
    predictor_outputter = PredictorOutputter(leagues_filepath=config.leagues_path)
    notifier = Notifier() if not skip_notifications else None
    
    # Phase 1: Fetch data for all leagues concurrently
    flow_logger.info("Phase 1: Fetching data for all leagues")
    fetch_results = []
    for league_id in league_ids:
        result = fetch_league_data.submit(league_id, data_fetcher)
        fetch_results.append(result)
    
    # Wait for all fetch tasks to complete
    fetch_data = [result.result() for result in fetch_results]
    
    # Log fetch summary
    successful_fetches = sum(1 for data in fetch_data if data['status'] == 'success')
    flow_logger.info(f"Fetch phase completed: {successful_fetches}/{len(league_ids)} leagues successful")
    
    # Phase 2: Process data for all leagues concurrently
    flow_logger.info("Phase 2: Processing data for all leagues")
    process_results = []
    for league_data in fetch_data:
        result = process_league_data.submit(league_data, data_processor)
        process_results.append(result)
    
    # Wait for all processing tasks to complete
    processed_data = [result.result() for result in process_results]
    
    # Log processing summary
    successful_processing = sum(1 for data in processed_data if data['status'] == 'success')
    flow_logger.info(f"Processing phase completed: {successful_processing}/{len(processed_data)} leagues successful")
    
    # Phase 3: Generate predictions
    flow_logger.info("Phase 3: Generating predictions")
    predictions_result = generate_predictions_task(processed_data, predictor_outputter)
    
    # Phase 4: Send notifications (if enabled)
    notification_result = None
    if not skip_notifications and notifier is not None:
        flow_logger.info("Phase 4: Sending notifications")
        notification_result = send_notifications(predictions_result, notifier)
    else:
        flow_logger.info("Phase 4: Skipping notifications")
    
    # Compile final results
    pipeline_result = {
        'pipeline_start': datetime.utcnow().isoformat(),
        'leagues_processed': len(league_ids),
        'successful_fetches': successful_fetches,
        'successful_processing': successful_processing,
        'predictions_generated': predictions_result.get('count', 0),
        'predictions_status': predictions_result['status'],
        'notifications_sent': notification_result is not None and notification_result['status'] == 'success',
        'overall_status': 'success' if predictions_result['status'] == 'success' else 'partial_failure'
    }
    
    flow_logger.info(f"Pipeline completed with status: {pipeline_result['overall_status']}")
    
    return pipeline_result


@flow(
    name="FormFinder Quick Update",
    description="Quick update for specific leagues"
)
def run_quick_update(config_path: str = "config.yaml", league_ids: List[int] = None) -> dict:
    """Quick update pipeline for specific leagues."""
    flow_logger = get_run_logger()
    flow_logger.info(f"Starting quick update for leagues: {league_ids}")
    
    return run_main_pipeline(config_path=config_path, league_ids=league_ids, skip_notifications=True)


@flow(
    name="FormFinder Health Check",
    description="Health check for FormFinder components"
)
def run_health_check(config_path: str = "config.yaml") -> dict:
    """Health check pipeline to verify all components are working."""
    flow_logger = get_run_logger()
    flow_logger.info("Starting FormFinder health check")
    
    health_status = {
        'timestamp': datetime.utcnow().isoformat(),
        'config': 'unknown',
        'database': 'unknown',
        'api': 'unknown',
        'overall': 'unknown'
    }
    
    try:
        # Load config from path
        from .config import load_config
        config = load_config(config_path)
        
        # Check configuration
        health_status['config'] = 'healthy'
        flow_logger.info("Configuration: OK")
        
        # Initialize database with config
        init_database(config.get_database_url())
        
        # Check database connection
        with get_db_session() as session:
            session.execute("SELECT 1")
        health_status['database'] = 'healthy'
        flow_logger.info("Database: OK")
        
        # Check API connectivity (fetch one league)
        # Create DataFetcherConfig from FormFinderConfig
        # Ensure league_ids are properly set in processing config
        processing_data = config.processing.model_dump()
        league_ids = config.get_league_ids()
        if processing_data.get('league_ids') is None:
            processing_data['league_ids'] = league_ids
        
        data_fetcher_config = DataFetcherConfig(
            api=config.api.model_dump(),
            processing=processing_data
        )
        data_fetcher = DataFetcher(data_fetcher_config)
        test_league_id = league_ids[0] if league_ids else 203
        # Test API connectivity by fetching fixtures
        data_fetcher.fetch_fixtures(test_league_id, limit=1)
        health_status['api'] = 'healthy'
        flow_logger.info("API: OK")
        
        health_status['overall'] = 'healthy'
        flow_logger.info("Health check completed: All systems healthy")
        
    except Exception as e:
        health_status['overall'] = 'unhealthy'
        health_status['error'] = str(e)
        flow_logger.error(f"Health check failed: {str(e)}")
    
    return health_status


def schedule_daily_pipeline(config_path: str = "config.yaml", cron_schedule: str = "0 6 * * *"):
    """Schedule the daily FormFinder pipeline."""
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
    
    deployment = Deployment.build_from_flow(
        flow=run_main_pipeline,
        name="daily-formfinder-pipeline",
        schedule=CronSchedule(cron=cron_schedule),
        work_queue_name="formfinder",
        tags=["formfinder", "daily", "production"],
        parameters={"config_path": config_path}
    )
    
    deployment_id = deployment.apply()
    logger.info("Daily FormFinder pipeline scheduled")
    return deployment_id


def schedule_health_check(config_path: str = "config.yaml"):
    """Schedule regular health checks."""
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
    
    deployment = Deployment.build_from_flow(
        flow=run_health_check,
        name="formfinder-health-check",
        schedule=CronSchedule(cron="*/30 * * * *"),  # Run every 30 minutes
        work_queue_name="formfinder",
        tags=["formfinder", "health-check", "monitoring"],
        parameters={"config_path": config_path}
    )
    
    deployment_id = deployment.apply()
    logger.info("FormFinder health check scheduled")
    return deployment_id



schedule_health_checks = schedule_health_check