"""Main Application Entry Point for FormFinder2

This module provides the main application entry point that coordinates all components
of the FormFinder2 system including data collection, feature computation, training,
and monitoring.

Author: FormFinder2 Team
Created: 2025-01-01
Purpose: Main application coordination
"""

import asyncio
import logging
import sys
import argparse
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
from sqlalchemy import text

from formfinder.config import load_config, FormFinderConfig
from formfinder.exceptions import (
    FormFinderError, ConfigurationError, DatabaseError,
    FeatureComputationError, TrainingError
)
from formfinder.feature_precomputer import FeaturePrecomputer
from formfinder.training_engine import TrainingEngine
from formfinder.monitoring import SystemMonitor
from formfinder.scheduler import TaskScheduler
from formfinder.orchestrator import WorkflowOrchestrator
from formfinder.database import get_db_session, get_db_manager
from formfinder.clients.api_client import SoccerDataAPIClient


class FormFinderApp:
    """Main FormFinder2 application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the FormFinder2 application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[FormFinderConfig] = None
        self.logger: Optional[logging.Logger] = None
        
        # Components
        self.feature_precomputer: Optional[FeaturePrecomputer] = None
        self.training_engine: Optional[TrainingEngine] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.scheduler: Optional[TaskScheduler] = None
        self.orchestrator: Optional[WorkflowOrchestrator] = None
        
        # Application state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the application."""
        try:
            # Load configuration
            self.config = load_config(self.config_path)
            
            # Setup logging
            self._setup_logging()
            
            self.logger.info("FormFinder2 application initialized")
            self.logger.info(f"Configuration loaded from: {self.config_path or 'default'}")
            
        except Exception as e:
            print(f"Failed to initialize FormFinder2: {str(e)}")
            sys.exit(1)
    
    def _setup_logging(self) -> None:
        """Setup application logging."""
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format=self.config.logging.format,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Add file handler if configured
        if self.config.logging.file_handler.enabled:
            log_dir = Path(self.config.logging.file_handler.directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "formfinder.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.logging.level.upper()))
            file_handler.setFormatter(logging.Formatter(self.config.logging.format))
            
            logging.getLogger().addHandler(file_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self) -> None:
        """Initialize system components."""
        try:
            self.logger.info("Initializing system components")
            
            # Initialize database manager
            self.db_manager = get_db_manager()
            self.logger.info("Database manager initialized")
            
            # Initialize API client
            self.api_client = SoccerDataAPIClient(self.db_manager.get_session())
            self.logger.info("API client initialized")
            
            # Initialize feature precomputer
            self.feature_precomputer = FeaturePrecomputer(self.db_manager.get_session())
            self.logger.info("Feature precomputer initialized")
            
            # Initialize training engine
            self.training_engine = TrainingEngine(self.config_path)
            self.logger.info("Training engine initialized")
            
            # Initialize system monitor
            self.system_monitor = SystemMonitor(self.config_path)
            self.logger.info("System monitor initialized")
            
            # Initialize scheduler
            self.scheduler = TaskScheduler(self.config_path)
            self.logger.info("Task scheduler initialized")
            
            # Initialize orchestrator
            self.orchestrator = WorkflowOrchestrator(self.config_path)
            self.logger.info("Workflow orchestrator initialized")
            
            self.logger.info("All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise FormFinderError(f"Component initialization failed: {str(e)}")
    
    async def run_feature_computation(self, fixture_ids: Optional[list] = None) -> Dict[str, Any]:
        """Run feature computation.
        
        Args:
            fixture_ids: Specific fixture IDs to process (if None, processes all pending)
            
        Returns:
            Computation results
        """
        if not self.feature_precomputer:
            self._initialize_components()
        
        self.logger.info("Starting feature computation")
        
        try:
            if fixture_ids:
                # Process specific fixtures
                results = []
                for fixture_id in fixture_ids:
                    result = await self.feature_precomputer.compute_features_for_fixture(fixture_id)
                    results.append(result)
                
                return {
                    'status': 'completed',
                    'processed_fixtures': len(fixture_ids),
                    'results': results
                }
            else:
                # Process all pending fixtures
                # Get fixtures that need feature computation
                query = text("""
                    SELECT f.id
                    FROM fixtures f
                    LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
                    WHERE f.match_date >= CURRENT_DATE
                    AND f.match_date <= CURRENT_DATE + INTERVAL '7 days'
                    AND pcf.fixture_id IS NULL
                    ORDER BY f.match_date
                    LIMIT 100
                """)
                
                with get_db_session() as session:
                    pending_fixtures = session.execute(query).fetchall()
                    fixture_ids = [row[0] for row in pending_fixtures]
                
                if not fixture_ids:
                    return {
                        'status': 'completed',
                        'processed_fixtures': 0,
                        'computation_time': 0
                    }
                
                start_time = time.time()
                result = self.feature_precomputer.compute_all_features(fixture_ids)
                computation_time = time.time() - start_time
                
                return {
                    'status': 'completed',
                    'processed_fixtures': result.get('successful_computations', 0),
                    'computation_time': computation_time
                }
                
        except Exception as e:
            self.logger.error(f"Feature computation failed: {str(e)}")
            raise FeatureComputationError(f"Feature computation failed: {str(e)}")
    
    async def run_model_training(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Run model training.
        
        Args:
            force_retrain: Force retraining even if recent model exists
            
        Returns:
            Training results
        """
        if not self.training_engine:
            self._initialize_components()
        
        self.logger.info("Starting model training")
        
        try:
            # Check if retraining is needed
            if not force_retrain:
                last_training = await self.training_engine.get_last_training_info()
                if last_training and self._is_recent_training(last_training):
                    self.logger.info("Recent training found, skipping")
                    return {
                        'status': 'skipped',
                        'reason': 'Recent training exists',
                        'last_training': last_training
                    }
            
            # Run training pipeline
            model, metrics = await asyncio.to_thread(
                self.training_engine.train_model_pipeline
            )
            
            self.logger.info(f"Model training completed with RMSE: {metrics.get('validation_rmse', 'N/A')}")
            
            return {
                'status': 'completed',
                'metrics': metrics,
                'model_path': str(self.training_engine.model_path)
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise TrainingError(f"Model training failed: {str(e)}")
    
    def _is_recent_training(self, last_training: Dict[str, Any]) -> bool:
        """Check if last training is recent enough.
        
        Args:
            last_training: Last training information
            
        Returns:
            True if training is recent
        """
        if not last_training.get('timestamp'):
            return False
        
        last_time = datetime.fromisoformat(last_training['timestamp'])
        hours_since = (datetime.now() - last_time).total_seconds() / 3600
        
        # Consider training recent if within configured hours
        max_hours = self.config.training.model_selection.retrain_threshold_hours
        return hours_since < max_hours
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run system health check.
        
        Returns:
            Health check results
        """
        if not self.system_monitor:
            self._initialize_components()
        
        self.logger.info("Running system health check")
        
        try:
            health_results = await self.system_monitor.run_health_checks()
            status = self.system_monitor.get_system_status()
            
            return {
                'status': 'completed',
                'overall_health': status['overall_health'],
                'health_checks': {
                    component: {
                        'status': result.status.value,
                        'message': result.message,
                        'response_time_ms': result.response_time_ms
                    }
                    for component, result in health_results.items()
                },
                'alerts': status['alerts']
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def start_scheduler(self) -> None:
        """Start the task scheduler."""
        if not self.scheduler:
            self._initialize_components()
        
        self.logger.info("Starting task scheduler")
        
        try:
            await self.scheduler.start()
        except Exception as e:
            self.logger.error(f"Scheduler failed: {str(e)}")
            raise FormFinderError(f"Scheduler failed: {str(e)}")
    
    async def start_monitoring(self, check_interval: int = 300) -> None:
        """Start continuous monitoring.
        
        Args:
            check_interval: Monitoring check interval in seconds
        """
        if not self.system_monitor:
            self._initialize_components()
        
        self.logger.info("Starting continuous monitoring")
        
        try:
            await self.system_monitor.start_monitoring(check_interval)
        except Exception as e:
            self.logger.error(f"Monitoring failed: {str(e)}")
            raise FormFinderError(f"Monitoring failed: {str(e)}")
    
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete data processing and training pipeline.
        
        Returns:
            Pipeline execution results
        """
        self.logger.info("Starting full pipeline execution")
        
        results = {
            'pipeline_start': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: Feature computation
            self.logger.info("Pipeline step 1: Feature computation")
            feature_result = await self.run_feature_computation()
            results['steps']['feature_computation'] = feature_result
            
            # Step 2: Data quality check
            self.logger.info("Pipeline step 2: Health check")
            health_result = await self.run_health_check()
            results['steps']['health_check'] = health_result
            
            # Step 3: Model training (if health is good)
            if health_result.get('overall_health') in ['healthy', 'warning']:
                self.logger.info("Pipeline step 3: Model training")
                training_result = await self.run_model_training()
                results['steps']['model_training'] = training_result
            else:
                self.logger.warning("Skipping model training due to health issues")
                results['steps']['model_training'] = {
                    'status': 'skipped',
                    'reason': 'Health check failed'
                }
            
            results['pipeline_end'] = datetime.now().isoformat()
            results['status'] = 'completed'
            
            self.logger.info("Full pipeline execution completed")
            
        except Exception as e:
            results['pipeline_end'] = datetime.now().isoformat()
            results['status'] = 'failed'
            results['error'] = str(e)
            
            self.logger.error(f"Pipeline execution failed: {str(e)}")
        
        return results
    
    async def start_daemon(self) -> None:
        """Start the application as a daemon with scheduler and monitoring."""
        self.logger.info("Starting FormFinder2 daemon")
        
        self.is_running = True
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Start scheduler and monitoring concurrently
            tasks = [
                asyncio.create_task(self.start_scheduler()),
                asyncio.create_task(self.start_monitoring())
            ]
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Daemon execution failed: {str(e)}")
            raise FormFinderError(f"Daemon failed: {str(e)}")
        
        finally:
            self.is_running = False
            self.logger.info("FormFinder2 daemon stopped")
    
    async def shutdown(self) -> None:
        """Shutdown the application gracefully."""
        self.logger.info("Shutting down FormFinder2 application")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Shutdown components
        if self.scheduler:
            await self.scheduler.shutdown()
        
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        self.logger.info("FormFinder2 application shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status.
        
        Returns:
            Application status information
        """
        status = {
            'application': {
                'name': 'FormFinder2',
                'version': '2.0.0',
                'is_running': self.is_running,
                'config_path': self.config_path,
                'timestamp': datetime.now().isoformat()
            },
            'components': {
                'feature_precomputer': self.feature_precomputer is not None,
                'training_engine': self.training_engine is not None,
                'system_monitor': self.system_monitor is not None,
                'scheduler': self.scheduler is not None,
                'orchestrator': self.orchestrator is not None
            }
        }
        
        # Add scheduler status if available
        if self.scheduler:
            try:
                status['scheduler'] = self.scheduler.get_job_status()
            except Exception as e:
                status['scheduler'] = {'error': str(e)}
        
        # Add system status if available
        if self.system_monitor:
            try:
                status['system'] = self.system_monitor.get_system_status()
            except Exception as e:
                status['system'] = {'error': str(e)}
        
        return status


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="FormFinder2 - Advanced Football Form Analysis and Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --daemon                    # Run as daemon with scheduler
  %(prog)s --features                  # Compute features for pending fixtures
  %(prog)s --train                     # Train the prediction model
  %(prog)s --health                    # Run system health check
  %(prog)s --pipeline                  # Run full pipeline
  %(prog)s --status                    # Show application status
  %(prog)s --config config.yaml       # Use custom configuration
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    
    # Operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        "--daemon", 
        action="store_true",
        help="Run as daemon with scheduler and monitoring"
    )
    
    mode_group.add_argument(
        "--features", 
        action="store_true",
        help="Compute features for pending fixtures"
    )
    
    mode_group.add_argument(
        "--train", 
        action="store_true",
        help="Train the prediction model"
    )
    
    mode_group.add_argument(
        "--health", 
        action="store_true",
        help="Run system health check"
    )
    
    mode_group.add_argument(
        "--pipeline", 
        action="store_true",
        help="Run complete pipeline (features + training)"
    )
    
    mode_group.add_argument(
        "--status", 
        action="store_true",
        help="Show application status"
    )
    
    mode_group.add_argument(
        "--monitor", 
        action="store_true",
        help="Start continuous monitoring"
    )
    
    # Options
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force operation (e.g., force retrain)"
    )
    
    parser.add_argument(
        "--fixtures", 
        nargs="+",
        type=int,
        help="Specific fixture IDs to process"
    )
    
    parser.add_argument(
        "--interval", 
        type=int,
        default=300,
        help="Monitoring check interval in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


async def main() -> None:
    """Main application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create application
    config_path = args.config or "config.yaml"
    app = FormFinderApp(config_path)
    
    # Override log level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.daemon:
            # Run as daemon
            await app.start_daemon()
            
        elif args.features:
            # Compute features
            result = await app.run_feature_computation(args.fixtures)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.train:
            # Train model
            result = await app.run_model_training(args.force)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.health:
            # Health check
            result = await app.run_health_check()
            print(json.dumps(result, indent=2, default=str))
            
        elif args.pipeline:
            # Full pipeline
            result = await app.run_full_pipeline()
            print(json.dumps(result, indent=2, default=str))
            
        elif args.status:
            # Show status
            status = app.get_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.monitor:
            # Start monitoring
            await app.start_monitoring(args.interval)
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        await app.shutdown()
        
    except FormFinderError as e:
        print(f"FormFinder2 Error: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logging.getLogger(__name__).exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())