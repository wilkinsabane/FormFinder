# FormFinder2 - Data Collection and Training Separation

## Overview

This document describes the enhanced FormFinder2 system that implements complete separation between data collection and model training phases, as outlined in the Data Collection and Training Separation PRD.

## Architecture

### Core Components

1. **Feature Pre-computation Engine** (`formfinder/feature_precomputer.py`)
   - Pre-computes and caches all features to eliminate API dependency during training
   - Handles form analysis, head-to-head statistics, and preview features
   - Implements batch processing and data quality monitoring

2. **Enhanced Training Engine** (`formfinder/training_engine.py`)
   - Reads pre-computed features from database without making API calls
   - Implements advanced feature engineering and model validation
   - Supports XGBoost with hyperparameter optimization

3. **System Monitor** (`formfinder/monitoring.py`)
   - Comprehensive health checks for database, API, and system resources
   - Performance monitoring and alerting
   - Data quality validation

4. **Task Scheduler** (`formfinder/scheduler.py`)
   - Cron-based scheduling for feature computation, training, and maintenance
   - Job management and dependency handling
   - Integration with workflow orchestrator

5. **Workflow Orchestrator** (`formfinder/orchestrator.py`)
   - Coordinates data collection and training processes
   - Task queue management and priority handling
   - Graceful error handling and recovery

6. **Main Application** (`formfinder/main.py`)
   - Central entry point that coordinates all components
   - CLI interface for various operation modes
   - Daemon mode for continuous operation

### Database Schema

The system introduces several new tables to support the separation:

#### Core Tables

- **`pre_computed_features`**: Stores all pre-computed features for fixtures
- **`h2h_cache_enhanced`**: Enhanced head-to-head statistics with detailed metrics
- **`feature_computation_log`**: Tracks feature computation status and performance
- **`api_usage_log`**: Monitors API quota usage and rate limiting
- **`data_quality_metrics`**: Tracks data quality indicators
- **`feature_computation_queue`**: Manages feature computation tasks

#### Views

- **`training_ready_features`**: Features ready for model training
- **`feature_computation_stats`**: Computation performance statistics
- **`daily_api_usage`**: Daily API usage summaries

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL or SQLite database
- Football-Data.org API key

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Apply database migrations:**
   ```bash
   python -c "from formfinder.database import apply_migration; apply_migration('migrations/20250101_data_collection_training_separation.sql')"
   ```

3. **Configure the system:**
   - Copy `config.yaml.example` to `config.yaml`
   - Update configuration with your database and API settings

## Usage

### Command Line Interface

The system provides a comprehensive CLI through `run_formfinder.py`:

#### Basic Operations

```bash
# Run as daemon with scheduler and monitoring
python run_formfinder.py --daemon

# Compute features for pending fixtures
python run_formfinder.py --features

# Train the prediction model
python run_formfinder.py --train

# Run system health check
python run_formfinder.py --health

# Run complete pipeline (features + training)
python run_formfinder.py --pipeline

# Show application status
python run_formfinder.py --status

# Start continuous monitoring
python run_formfinder.py --monitor
```

#### Advanced Options

```bash
# Use custom configuration
python run_formfinder.py --config custom_config.yaml --features

# Process specific fixtures
python run_formfinder.py --features --fixtures 12345 12346 12347

# Force model retraining
python run_formfinder.py --train --force

# Verbose logging
python run_formfinder.py --pipeline --verbose

# Custom monitoring interval
python run_formfinder.py --monitor --interval 600
```

### Programmatic Usage

```python
from formfinder.main import FormFinderApp
import asyncio

# Create application instance
app = FormFinderApp('config.yaml')

# Run feature computation
result = await app.run_feature_computation()
print(f"Processed {result['processed_fixtures']} fixtures")

# Train model
training_result = await app.run_model_training()
print(f"Model RMSE: {training_result['metrics']['validation_rmse']}")

# Run health check
health = await app.run_health_check()
print(f"System health: {health['overall_health']}")
```

## Configuration

The system uses a comprehensive configuration system defined in `formfinder/config.py`:

### Key Configuration Sections

#### Feature Computation
```yaml
feature_computation:
  batch_processing:
    batch_size: 50
    max_concurrent_batches: 3
    retry_attempts: 3
  
  form_analysis:
    lookback_days: 30
    min_matches_required: 3
  
  h2h_analysis:
    max_h2h_matches: 10
    min_h2h_matches: 1
```

#### Training Configuration
```yaml
training:
  data_settings:
    min_training_samples: 1000
    validation_split: 0.2
    test_split: 0.1
  
  xgboost_params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
```

#### Monitoring
```yaml
monitoring:
  health_checks:
    database_timeout: 5.0
    api_timeout: 10.0
    disk_space_threshold: 0.9
    memory_threshold: 0.85
```

#### Scheduling
```yaml
scheduler:
  schedules:
    feature_computation: "0 */6 * * *"  # Every 6 hours
    model_training: "0 2 * * *"         # Daily at 2 AM
    health_checks: "*/15 * * * *"       # Every 15 minutes
```

## Workflow

### Data Collection Phase

1. **Feature Pre-computation**
   - Scheduler triggers feature computation every 6 hours
   - System fetches fixture data and computes all features
   - Features are stored in `pre_computed_features` table
   - API usage is logged and monitored

2. **Data Quality Monitoring**
   - Continuous validation of computed features
   - Detection of missing or invalid data
   - Alerting for data quality issues

### Training Phase

1. **Model Training**
   - Scheduled daily or triggered manually
   - Reads pre-computed features from database
   - No API calls during training
   - Validates model performance before deployment

2. **Performance Monitoring**
   - Tracks training metrics and model performance
   - Alerts for significant performance degradation
   - Automatic retraining triggers

### Continuous Operations

1. **Health Monitoring**
   - Regular system health checks
   - Resource usage monitoring
   - API quota tracking

2. **Maintenance Tasks**
   - Cache cleanup and optimization
   - Database maintenance
   - Log rotation

## Monitoring and Alerting

### Health Checks

The system performs comprehensive health checks:

- **Database Connectivity**: Connection and query performance
- **API Status**: Quota usage and response times
- **System Resources**: CPU, memory, and disk usage
- **Data Quality**: Feature completeness and validity
- **Training Performance**: Model accuracy and training times

### Alerting

Alerts are sent via:
- Email notifications
- SMS (via Twilio, optional)
- System logs
- Application status endpoints

### Performance Metrics

Key metrics tracked:
- Feature computation time and success rate
- API response times and error rates
- Model training performance
- System resource utilization
- Data quality scores

## Error Handling

The system implements comprehensive error handling:

### Custom Exceptions
- `FeatureComputationError`: Feature computation failures
- `DataQualityError`: Data validation issues
- `TrainingError`: Model training problems
- `RateLimitError`: API rate limiting
- `DatabaseError`: Database connectivity issues

### Recovery Mechanisms
- Automatic retry with exponential backoff
- Graceful degradation for non-critical failures
- Task queue for failed operations
- Manual intervention triggers

## Performance Optimization

### Database Optimization
- Comprehensive indexing strategy
- Query optimization for large datasets
- Connection pooling
- Batch processing for bulk operations

### Caching Strategy
- Multi-level caching (memory, database, file)
- Cache invalidation policies
- Preemptive cache warming
- Cache performance monitoring

### Async Operations
- Asynchronous API calls
- Concurrent feature computation
- Non-blocking database operations
- Parallel model training preparation

## Security

### API Security
- Secure API key management
- Rate limiting compliance
- Request/response logging
- Error handling without data exposure

### Database Security
- Connection encryption
- Parameterized queries
- Access control
- Audit logging

### Configuration Security
- Environment variable support
- Encrypted configuration options
- Secure defaults
- Validation of security settings

## Troubleshooting

### Common Issues

1. **Feature Computation Failures**
   ```bash
   # Check computation logs
   python run_formfinder.py --health
   
   # Retry failed computations
   python run_formfinder.py --features --force
   ```

2. **Training Performance Issues**
   ```bash
   # Check data quality
   python run_formfinder.py --health
   
   # Force retrain with verbose logging
   python run_formfinder.py --train --force --verbose
   ```

3. **API Rate Limiting**
   ```bash
   # Check API usage
   python run_formfinder.py --status
   
   # Monitor API quota
   python run_formfinder.py --monitor
   ```

### Logs and Debugging

- Application logs: `logs/formfinder.log`
- Error logs: `logs/errors.log`
- Performance logs: `logs/performance.log`
- Database logs: Check database-specific logging

### Performance Tuning

1. **Batch Size Optimization**
   - Adjust `feature_computation.batch_processing.batch_size`
   - Monitor memory usage and processing time

2. **Concurrency Tuning**
   - Adjust `feature_computation.batch_processing.max_concurrent_batches`
   - Consider system resources and API limits

3. **Cache Configuration**
   - Tune cache sizes and TTL values
   - Monitor cache hit rates

## Migration from Previous Version

To migrate from the previous FormFinder version:

1. **Backup existing data**
2. **Apply new database schema**
3. **Update configuration files**
4. **Run initial feature computation**
5. **Validate system operation**

Detailed migration guide available in `MIGRATION.md`.

## Development

### Code Structure

```
formfinder/
├── __init__.py
├── config.py              # Configuration management
├── exceptions.py          # Custom exceptions
├── feature_precomputer.py # Feature pre-computation engine
├── training_engine.py     # Enhanced training engine
├── monitoring.py          # System monitoring
├── scheduler.py           # Task scheduling
├── orchestrator.py        # Workflow orchestration
└── main.py               # Main application entry point

migrations/
└── 20250101_data_collection_training_separation.sql

run_formfinder.py         # CLI runner script
config.yaml.example       # Example configuration
requirements.txt          # Python dependencies
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=formfinder

# Run specific test categories
pytest tests/test_feature_computation.py
pytest tests/test_training.py
pytest tests/test_monitoring.py
```

### Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure all health checks pass

## Support

For issues and questions:
- Check the troubleshooting section
- Review system logs
- Run health checks
- Consult the configuration documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.