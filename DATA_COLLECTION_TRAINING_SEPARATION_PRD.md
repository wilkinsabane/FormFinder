# Data Collection and Training Separation PRD

## Executive Summary

This Product Requirements Document (PRD) outlines the architectural redesign of the FormFinder2 system to separate data collection from model training, addressing the critical API rate limiting bottleneck of 75 requests per day that currently causes training delays of up to an hour per game.

## Problem Statement

### Current Issues
- **API Rate Limiting Bottleneck**: 75 requests/day limit severely constrains training pipeline
- **Mixed Responsibilities**: Training script performs both data fetching and model training
- **Inconsistent Training Times**: Network issues and API delays make training unpredictable
- **Resource Waste**: API quota consumed on repeated training runs
- **Poor Developer Experience**: Cannot iterate quickly on model improvements

### Impact
- Training a single game can take 1+ hours due to API waits
- Model experimentation is severely limited
- Development velocity is significantly reduced
- Training results are not reproducible

## Solution Overview

### Architecture Principles
1. **Separation of Concerns**: Data collection and training are completely independent
2. **Database-Centric**: All features pre-computed and stored in PostgreSQL
3. **API Efficiency**: API calls only for fresh data collection, never during training
4. **Reproducibility**: Training results must be consistent and repeatable
5. **Performance**: Training should complete in minutes, not hours

## Technical Architecture

### Environment Setup
- **Conda Environment**: `formfinder`
- **Database**: PostgreSQL database named `formfinder`
- **No CSV Dependencies**: All data persistence through PostgreSQL

### Component Separation

#### 1. Enhanced Data Collection Layer

**Purpose**: Comprehensive data fetching and feature pre-computation

**Components**:
- Enhanced Historical Fetcher
- Enhanced Upcoming Fetcher
- Feature Pre-computation Engine
- H2H Statistics Calculator
- Preview Metrics Collector

**Responsibilities**:
- Fetch all match data from API
- Pre-compute rolling form features
- Calculate and cache H2H statistics
- Collect preview metrics and weather data
- Store all features in optimized database tables
- Manage API rate limiting and caching

#### 2. Pure Training Layer

**Purpose**: Fast, reproducible model training using pre-computed features

**Components**:
- Database-Only Feature Engine
- Model Training Pipeline
- Model Validation Framework
- Performance Monitoring

**Responsibilities**:
- Read features from database only
- Perform feature engineering on cached data
- Train and validate models
- Generate performance metrics
- Save trained models

## Database Schema Enhancements

### New Tables

#### `pre_computed_features`
```sql
CREATE TABLE pre_computed_features (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    match_date TIMESTAMP NOT NULL,
    league_id INTEGER NOT NULL,
    
    -- Home team form features
    home_avg_goals_scored DECIMAL(4,2) DEFAULT 0.0,
    home_avg_goals_conceded DECIMAL(4,2) DEFAULT 0.0,
    home_avg_goals_scored_home DECIMAL(4,2) DEFAULT 0.0,
    home_avg_goals_conceded_home DECIMAL(4,2) DEFAULT 0.0,
    home_form_last_5_games TEXT, -- JSON array of last 5 results
    
    -- Away team form features
    away_avg_goals_scored DECIMAL(4,2) DEFAULT 0.0,
    away_avg_goals_conceded DECIMAL(4,2) DEFAULT 0.0,
    away_avg_goals_scored_away DECIMAL(4,2) DEFAULT 0.0,
    away_avg_goals_conceded_away DECIMAL(4,2) DEFAULT 0.0,
    away_form_last_5_games TEXT, -- JSON array of last 5 results
    
    -- H2H features
    h2h_overall_games INTEGER DEFAULT 0,
    h2h_avg_total_goals DECIMAL(4,2) DEFAULT 0.0,
    h2h_overall_home_goals DECIMAL(4,2) DEFAULT 0.0,
    h2h_overall_away_goals DECIMAL(4,2) DEFAULT 0.0,
    h2h_home_advantage DECIMAL(4,2) DEFAULT 0.0,
    h2h_last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Preview and weather features
    excitement_rating DECIMAL(4,2) DEFAULT 0.0,
    weather_temp_c DECIMAL(4,1) DEFAULT 21.0,
    weather_temp_f DECIMAL(4,1) DEFAULT 69.8,
    weather_humidity DECIMAL(4,1) DEFAULT 50.0,
    weather_wind_speed DECIMAL(4,1) DEFAULT 5.0,
    weather_precipitation DECIMAL(4,1) DEFAULT 0.0,
    weather_condition VARCHAR(50) DEFAULT 'Clear',
    
    -- Target variables (for completed matches)
    total_goals INTEGER,
    over_2_5 BOOLEAN,
    
    -- Metadata
    features_computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    
    UNIQUE(fixture_id),
    INDEX idx_match_date (match_date),
    INDEX idx_teams (home_team_id, away_team_id),
    INDEX idx_league_date (league_id, match_date)
);
```

#### `h2h_cache_enhanced`
```sql
CREATE TABLE h2h_cache_enhanced (
    id SERIAL PRIMARY KEY,
    team1_id INTEGER NOT NULL,
    team2_id INTEGER NOT NULL,
    competition_id INTEGER,
    
    -- Overall statistics
    total_games INTEGER DEFAULT 0,
    team1_wins INTEGER DEFAULT 0,
    team2_wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    
    -- Goal statistics
    avg_total_goals DECIMAL(4,2) DEFAULT 0.0,
    avg_team1_goals DECIMAL(4,2) DEFAULT 0.0,
    avg_team2_goals DECIMAL(4,2) DEFAULT 0.0,
    
    -- Home/Away breakdown
    team1_home_games INTEGER DEFAULT 0,
    team1_home_wins INTEGER DEFAULT 0,
    team1_home_goals DECIMAL(4,2) DEFAULT 0.0,
    team2_home_games INTEGER DEFAULT 0,
    team2_home_wins INTEGER DEFAULT 0,
    team2_home_goals DECIMAL(4,2) DEFAULT 0.0,
    
    -- Recent form (last 5 meetings)
    recent_games_data TEXT, -- JSON array
    
    -- Cache metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cache_expires_at TIMESTAMP,
    data_source VARCHAR(20) DEFAULT 'api',
    
    UNIQUE(team1_id, team2_id, competition_id),
    INDEX idx_teams_comp (team1_id, team2_id, competition_id),
    INDEX idx_cache_expiry (cache_expires_at)
);
```

#### `feature_computation_log`
```sql
CREATE TABLE feature_computation_log (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL,
    computation_type VARCHAR(50) NOT NULL, -- 'form', 'h2h', 'preview', 'weather'
    status VARCHAR(20) NOT NULL, -- 'success', 'failed', 'skipped'
    error_message TEXT,
    computation_time_ms INTEGER,
    api_calls_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_fixture_type (fixture_id, computation_type),
    INDEX idx_status_date (status, created_at)
);
```

## Implementation Plan

### Phase 1: Database Schema Enhancement (Week 1)

#### Tasks:
1. **Create Migration Scripts**
   - Design and implement new table schemas
   - Create indexes for optimal query performance
   - Add foreign key constraints
   - Implement data validation triggers

2. **Database Performance Optimization**
   - Analyze query patterns for feature retrieval
   - Create composite indexes for common queries
   - Implement partitioning for large tables if needed
   - Set up connection pooling optimization

#### Deliverables:
- Migration scripts in `migrations/` directory
- Database schema documentation
- Performance benchmarking results
- Rollback procedures

### Phase 2: Enhanced Data Collection Layer (Week 2-3)

#### 2.1 Feature Pre-computation Engine

**File**: `formfinder/feature_precomputer.py`

```python
class FeaturePrecomputer:
    """Pre-computes and caches all features for training."""
    
    def __init__(self, db_session, api_client):
        self.db_session = db_session
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
    
    async def compute_all_features(self, fixture_ids: List[int]) -> Dict[str, int]:
        """Compute all features for given fixtures."""
        
    async def compute_form_features(self, fixture_id: int) -> bool:
        """Compute rolling form features for both teams."""
        
    async def compute_h2h_features(self, fixture_id: int) -> bool:
        """Compute and cache H2H statistics."""
        
    async def compute_preview_features(self, fixture_id: int) -> bool:
        """Compute preview and weather features."""
```

#### 2.2 Enhanced Historical Fetcher

**Enhancements to**: `formfinder/historical_fetcher.py`

- Add feature pre-computation after data fetching
- Implement batch processing for efficiency
- Add comprehensive error handling and retry logic
- Include progress tracking and detailed logging

#### 2.3 Enhanced Upcoming Fetcher

**Enhancements to**: `formfinder/upcoming_fetcher.py`

- Pre-compute features for upcoming matches
- Handle partial feature computation gracefully
- Implement smart caching strategies
- Add notification system for failed computations

#### 2.4 H2H Statistics Manager

**File**: `formfinder/h2h_manager.py`

```python
class H2HManager:
    """Manages H2H statistics computation and caching."""
    
    def __init__(self, db_session, api_client):
        self.db_session = db_session
        self.api_client = api_client
        self.cache_ttl = timedelta(hours=24)
    
    async def get_or_compute_h2h(self, team1_id: int, team2_id: int, 
                                competition_id: int) -> Dict[str, Any]:
        """Get H2H stats from cache or compute via API."""
        
    async def batch_compute_h2h(self, team_pairs: List[Tuple[int, int]], 
                               competition_id: int) -> Dict[str, int]:
        """Batch compute H2H statistics for multiple team pairs."""
```

### Phase 3: Pure Training Layer (Week 4)

#### 3.1 Database-Only Feature Engine

**File**: `scripts/database_feature_engine.py`

```python
class DatabaseFeatureEngine:
    """Retrieves pre-computed features from database only."""
    
    def __init__(self, db_session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
    
    def load_training_features(self, leagues: List[int], 
                             start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """Load pre-computed features for training."""
        
    def validate_feature_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that all required features are present."""
        
    def get_feature_quality_report(self, fixture_ids: List[int]) -> Dict[str, Any]:
        """Generate data quality report for features."""
```

#### 3.2 Enhanced Training Script

**Enhancements to**: `scripts/train_model.py`

- Remove all API client dependencies
- Replace `build_features()` with database-only feature loading
- Add comprehensive feature validation
- Implement fast feature engineering pipeline
- Add model performance tracking

```python
def load_features_from_database(db_session, leagues: List[int], 
                               start_date: datetime, 
                               end_date: datetime) -> pd.DataFrame:
    """Load all pre-computed features from database.
    
    This replaces the previous build_features() function that made API calls.
    """
    feature_engine = DatabaseFeatureEngine(db_session)
    
    # Load pre-computed features
    features_df = feature_engine.load_training_features(leagues, start_date, end_date)
    
    # Validate feature completeness
    quality_report = feature_engine.validate_feature_completeness(features_df)
    
    if quality_report['missing_features_pct'] > 0.1:  # More than 10% missing
        logger.warning(f"High missing feature rate: {quality_report['missing_features_pct']:.2%}")
    
    return features_df
```

### Phase 4: Orchestration and Automation (Week 5)

#### 4.1 Data Collection Scheduler

**File**: `scripts/data_collection_scheduler.py`

```python
class DataCollectionScheduler:
    """Orchestrates automated data collection and feature computation."""
    
    def __init__(self):
        self.config = load_config()
        self.db_session = self._create_db_session()
        self.api_client = SoccerDataAPIClient()
    
    async def daily_data_collection(self):
        """Run daily data collection and feature computation."""
        
    async def weekly_historical_update(self):
        """Run weekly historical data updates."""
        
    async def real_time_upcoming_update(self):
        """Update upcoming fixtures and compute features."""
```

#### 4.2 Workflow Integration

**Enhancements to**: `formfinder/workflows.py`

- Add data collection workflows
- Implement feature computation pipelines
- Add monitoring and alerting
- Create health check endpoints

### Phase 5: Monitoring and Optimization (Week 6)

#### 5.1 Performance Monitoring

**File**: `scripts/performance_monitor.py`

```python
class PerformanceMonitor:
    """Monitors system performance and data quality."""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance and quality report."""
        
    def check_feature_freshness(self) -> Dict[str, Any]:
        """Check if features are up to date."""
        
    def monitor_api_usage(self) -> Dict[str, Any]:
        """Monitor API usage and rate limiting."""
```

#### 5.2 Data Quality Assurance

**File**: `scripts/data_quality_checker.py`

```python
class DataQualityChecker:
    """Ensures data quality and feature integrity."""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    def validate_feature_distributions(self) -> Dict[str, Any]:
        """Validate that feature distributions are reasonable."""
        
    def check_data_consistency(self) -> Dict[str, Any]:
        """Check for data consistency issues."""
        
    def identify_outliers(self) -> Dict[str, Any]:
        """Identify potential data outliers."""
```

## Configuration Management

### Enhanced Configuration

**File**: `formfinder/config.py`

```python
class FeatureComputationConfig:
    """Configuration for feature computation."""
    
    # Feature computation settings
    FORM_LOOKBACK_GAMES = 5
    H2H_CACHE_TTL_HOURS = 24
    PREVIEW_CACHE_TTL_HOURS = 2
    WEATHER_CACHE_TTL_HOURS = 6
    
    # Batch processing settings
    FEATURE_BATCH_SIZE = 100
    MAX_CONCURRENT_COMPUTATIONS = 5
    COMPUTATION_TIMEOUT_SECONDS = 30
    
    # Data quality thresholds
    MIN_FEATURE_COMPLETENESS = 0.9
    MAX_MISSING_H2H_RATE = 0.2
    MAX_COMPUTATION_FAILURE_RATE = 0.1
    
    # API usage optimization
    DAILY_API_QUOTA = 75
    API_QUOTA_BUFFER = 10  # Reserve 10 requests for urgent needs
    ENABLE_AGGRESSIVE_CACHING = True
```

## Error Handling and Resilience

### Comprehensive Error Handling Strategy

1. **Graceful Degradation**
   - Use cached/default values when API fails
   - Continue training with partial features
   - Log all failures for later retry

2. **Retry Mechanisms**
   - Exponential backoff for API calls
   - Automatic retry for failed feature computations
   - Circuit breaker pattern for API failures

3. **Data Validation**
   - Validate all computed features
   - Check for reasonable value ranges
   - Flag suspicious data for manual review

4. **Monitoring and Alerting**
   - Real-time monitoring of computation success rates
   - Alerts for high failure rates
   - Daily data quality reports

## Performance Expectations

### Before Implementation (Current State)
- **Training Time**: 1+ hours per training run
- **API Dependency**: 100% dependent on API availability
- **Reproducibility**: Poor due to network variability
- **Development Velocity**: Severely limited

### After Implementation (Target State)
- **Training Time**: 2-5 minutes per training run
- **API Dependency**: 0% during training
- **Reproducibility**: 100% consistent results
- **Development Velocity**: 10-50x improvement

### Specific Performance Targets

1. **Data Collection Performance**
   - Process 1000 fixtures in < 30 minutes
   - Feature computation success rate > 95%
   - API quota utilization < 90%

2. **Training Performance**
   - Load 10,000 pre-computed features in < 30 seconds
   - Complete full training pipeline in < 5 minutes
   - Zero API calls during training

3. **System Reliability**
   - 99.9% uptime for data collection services
   - < 1% data quality issues
   - Automatic recovery from transient failures

## Testing Strategy

### Unit Tests
- Test all feature computation functions
- Mock API responses for consistent testing
- Validate database operations
- Test error handling scenarios

### Integration Tests
- End-to-end data collection pipeline
- Training pipeline with pre-computed features
- Database performance under load
- API rate limiting behavior

### Performance Tests
- Benchmark feature computation speed
- Load testing for concurrent operations
- Memory usage optimization
- Database query performance

## Migration Strategy

### Phase-by-Phase Migration

1. **Phase 1**: Set up new database tables alongside existing ones
2. **Phase 2**: Implement data collection enhancements with dual writing
3. **Phase 3**: Create new training pipeline using pre-computed features
4. **Phase 4**: Gradually migrate from old to new training pipeline
5. **Phase 5**: Remove old API-dependent training code

### Rollback Plan
- Maintain old training pipeline during migration
- Feature flags to switch between old and new systems
- Database rollback scripts for schema changes
- Comprehensive backup strategy

## Success Metrics

### Primary Metrics
1. **Training Speed**: Reduce training time from hours to minutes
2. **API Efficiency**: Eliminate API calls during training
3. **Reproducibility**: 100% consistent training results
4. **Development Velocity**: Enable rapid model experimentation

### Secondary Metrics
1. **Data Quality**: Maintain > 95% feature completeness
2. **System Reliability**: Achieve 99.9% uptime
3. **Resource Utilization**: Optimize API quota usage
4. **Developer Experience**: Reduce setup and debugging time

## Risk Assessment and Mitigation

### High-Risk Areas

1. **Database Performance**
   - **Risk**: Slow queries with large feature tables
   - **Mitigation**: Comprehensive indexing, query optimization, partitioning

2. **Data Consistency**
   - **Risk**: Inconsistent features between collection and training
   - **Mitigation**: Atomic transactions, data validation, integrity checks

3. **API Quota Management**
   - **Risk**: Exceeding daily API limits
   - **Mitigation**: Smart quota tracking, prioritization, caching

### Medium-Risk Areas

1. **Feature Computation Failures**
   - **Risk**: High failure rates affecting data quality
   - **Mitigation**: Robust error handling, retry mechanisms, fallback values

2. **Storage Requirements**
   - **Risk**: Rapid growth of feature tables
   - **Mitigation**: Data archiving, compression, cleanup procedures

## Conclusion

This comprehensive separation of data collection and training will transform the FormFinder2 system from a slow, API-dependent training pipeline to a fast, reliable, and scalable machine learning platform. The implementation will eliminate the 75 requests/day bottleneck, enable rapid model experimentation, and provide a solid foundation for future enhancements.

The phased approach ensures minimal disruption to current operations while delivering immediate benefits as each phase is completed. The focus on PostgreSQL database optimization and comprehensive error handling will ensure the system is robust and maintainable.

**Expected Timeline**: 6 weeks for complete implementation
**Expected ROI**: 10-50x improvement in development velocity
**Risk Level**: Medium (well-mitigated through phased approach)
**Maintenance Overhead**: Low (automated monitoring and self-healing capabilities)