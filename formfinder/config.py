"""Configuration management using Pydantic for validation and YAML loading."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class APIConfig(BaseModel):
    """API configuration settings."""
    auth_token: str
    base_url: str = "https://api.soccerdataapi.com"
    h2h_path: str = "/head-to-head/"
    preview_path: str = "/match-preview/"
    rate_limit_requests: int = 300
    rate_limit_period: int = 60
    timeout: int = 15
    max_retries: int = 3
    h2h_ttl_seconds: int = 24 * 3600
    preview_ttl_seconds: int = 2 * 3600


class FeatureComputationConfig(BaseModel):
    """Configuration for feature computation and pre-processing."""
    
    # Feature computation settings
    form_lookback_games: int = 5
    h2h_cache_ttl_hours: int = 24
    preview_cache_ttl_hours: int = 2
    weather_cache_ttl_hours: int = 6
    
    # Batch processing settings
    feature_batch_size: int = 100
    max_concurrent_computations: int = 5
    computation_timeout_seconds: int = 30
    
    # Data quality thresholds
    min_feature_completeness: float = Field(ge=0.0, le=1.0, default=0.9)
    max_missing_h2h_rate: float = Field(ge=0.0, le=1.0, default=0.2)
    max_computation_failure_rate: float = Field(ge=0.0, le=1.0, default=0.1)
    
    # API usage optimization
    daily_api_quota: int = 75
    api_quota_buffer: int = 10  # Reserve 10 requests for urgent needs
    enable_aggressive_caching: bool = True
    
    # Feature value ranges for validation
    feature_ranges: dict = Field(default_factory=lambda: {
        'home_avg_goals_scored': (0.0, 8.0),
        'home_avg_goals_conceded': (0.0, 8.0),
        'away_avg_goals_scored': (0.0, 8.0),
        'away_avg_goals_conceded': (0.0, 8.0),
        'h2h_total_matches': (0, 50),
        'h2h_avg_total_goals': (0.0, 10.0),
        'excitement_rating': (0.0, 10.0),
        'weather_temp_c': (-20.0, 50.0),
        'weather_humidity': (0.0, 100.0),
        'weather_wind_speed': (0.0, 100.0),
        'weather_precipitation': (0.0, 100.0)
    })


class ProcessingConfig(BaseModel):
    """Data processing configuration."""
    league_ids: Optional[List[int]] = None  # Will be loaded from leagues.json if not provided
    season_year: str = "2024-2025"
    max_concurrent_requests: int = 10
    inter_league_delay: int = 0
    cache_ttl_hours: int = 24
    recent_period: int = 10
    win_rate_threshold: float = Field(ge=0.0, le=1.0, default=0.70)
    
    def get_league_ids(self, leagues_path: str = "leagues.json") -> List[int]:
        """Get league IDs from config or load from leagues.json file."""
        if self.league_ids is not None:
            return self.league_ids
        
        # Load from leagues.json
        leagues_file = Path(leagues_path)
        if not leagues_file.exists():
            raise FileNotFoundError(f"Leagues file not found: {leagues_path}")
        
        try:
            with open(leagues_file, 'r', encoding='utf-8') as f:
                leagues_data = json.load(f)
            
            # Extract IDs from the results array
            if 'results' in leagues_data and isinstance(leagues_data['results'], list):
                return [league['id'] for league in leagues_data['results'] if 'id' in league]
            else:
                raise ValueError("Invalid leagues.json format: expected 'results' array")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in leagues file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading leagues from {leagues_path}: {e}")


class SQLiteConfig(BaseModel):
    """SQLite database configuration."""
    path: str = "data/formfinder.db"


class PostgreSQLConfig(BaseModel):
    """PostgreSQL database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "formfinder"
    username: str
    password: str


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = Field(pattern="^(sqlite|postgresql)$")
    sqlite: Optional[SQLiteConfig] = None
    postgresql: Optional[PostgreSQLConfig] = None

    @field_validator('sqlite')
    @classmethod
    def validate_sqlite_config(cls, v, info):
        if info.data.get('type') == 'sqlite' and v is None:
            return SQLiteConfig()
        return v

    @field_validator('postgresql')
    @classmethod
    def validate_postgresql_config(cls, v, info):
        if info.data.get('type') == 'postgresql' and v is None:
            raise ValueError('PostgreSQL configuration required when type is postgresql')
        return v


class EmailConfig(BaseModel):
    """Email notification configuration."""
    enabled: bool = True
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str
    password: str
    from_address: str
    to_addresses: List[str]
    max_matches_in_email_body: int = 10


class TwilioConfig(BaseModel):
    """Twilio SMS configuration."""
    account_sid: str
    auth_token: str
    from_number: str
    to_number: str


class SMSConfig(BaseModel):
    """SMS notification configuration."""
    enabled: bool = False
    provider: str = Field(pattern="^(twilio|aws_sns)$", default="twilio")
    twilio: Optional[TwilioConfig] = None
    max_matches_in_sms: int = 1


class NotificationsConfig(BaseModel):
    """Notifications configuration."""
    email: EmailConfig
    sms: SMSConfig
    predictions_dir: str = "data/predictions"


class FileHandlerConfig(BaseModel):
    """File handler logging configuration."""
    enabled: bool = True
    directory: str = "data/logs"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class ConsoleHandlerConfig(BaseModel):
    """Console handler logging configuration."""
    enabled: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", default="INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: FileHandlerConfig
    console_handler: ConsoleHandlerConfig


class DirectoriesConfig(BaseModel):
    """Directory paths configuration."""
    data_root: str = "data"
    logs: str = "data/logs"
    cache: str = "data/cache"
    predictions: str = "data/predictions"
    # Legacy CSV directories (will be phased out)
    historical: str = "data/historical"
    fixtures: str = "data/fixtures"
    standings: str = "data/standings"
    processed_data: str = "processed_data"


class WorkflowConfig(BaseModel):
    """Workflow orchestration configuration."""
    orchestrator: str = Field(pattern="^(prefect|dagster|local)$", default="local")
    retry_attempts: int = 3
    retry_delay_seconds: int = 60
    task_timeout_seconds: int = 3600
    enable_notifications: bool = True


class TestingConfig(BaseModel):
    """Testing configuration."""
    test_database_url: str = "sqlite:///data/test_formfinder.db"
    mock_api_responses: bool = True
    test_data_dir: str = "tests/data"


class NewsProviderConfig(BaseModel):
    """Configuration for a single news provider."""
    enabled: bool = True
    api_key: str = ""
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    priority: int = 1  # Lower number = higher priority


class SentimentAnalysisConfig(BaseModel):
    """Sentiment analysis configuration with multi-provider support."""
    enabled: bool = False
    cache_hours: int = 24
    prediction_weights: dict = Field(default_factory=lambda: {"form": 0.7, "sentiment": 0.3})
    
    # Multi-provider configuration
    providers: Dict[str, NewsProviderConfig] = Field(default_factory=lambda: {
        "newsapi": NewsProviderConfig(
            enabled=True,
            api_key="",
            rate_limit_per_minute=60,
            rate_limit_per_day=1000,
            priority=1
        ),
        "newsdata_io": NewsProviderConfig(
            enabled=True,
            api_key="",
            rate_limit_per_minute=120,  # 1800 credits per 15 min = 120/min
            rate_limit_per_day=200,     # 200 credits per day
            priority=2
        ),
        "thenewsapi": NewsProviderConfig(
            enabled=True,
            api_key="",
            rate_limit_per_minute=50,
            rate_limit_per_day=1000,    # Depends on plan
            priority=3
        )
    })
    
    # Failover settings
    enable_failover: bool = True
    load_balancing_strategy: str = Field(pattern="^(priority|round_robin|random|least_used)$", default="priority")
    provider_health_check_interval: int = 300  # 5 minutes
    provider_cooldown_minutes: int = 15  # Cool down after rate limit hit
    
    # Legacy support (for backward compatibility)
    news_api_key: str = ""  # Will be used for newsapi provider if providers.newsapi.api_key is empty
    
    def get_provider_config(self, provider_name: str) -> Optional[NewsProviderConfig]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider_name)
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider names sorted by priority."""
        enabled = [(name, config) for name, config in self.providers.items() if config.enabled]
        return [name for name, config in sorted(enabled, key=lambda x: x[1].priority)]


class FeatureComputationConfig(BaseModel):
    """Configuration for feature computation engine."""
    
    # Batch processing settings
    feature_batch_size: int = 50
    max_concurrent_computations: int = 5
    
    # Form analysis settings
    form_lookback_games: int = 5
    form_weight_decay: float = 0.8
    
    # H2H cache settings
    h2h_cache_ttl_hours: int = 168  # 1 week
    h2h_min_games_threshold: int = 3
    
    # Data quality thresholds
    min_feature_completeness: float = 0.7
    min_data_quality_score: float = 0.6
    
    # API usage limits
    daily_api_quota: int = 75
    api_rate_limit_per_minute: int = 10
    api_backoff_factor: float = 2.0
    max_api_retries: int = 3
    
    # Cache settings
    enable_feature_caching: bool = True
    cache_refresh_threshold_hours: int = 24
    
    # Computation priorities
    priority_leagues: List[int] = Field(default_factory=lambda: [2021, 2014, 2002, 2019, 2015])
    skip_low_priority_fixtures: bool = False


class TrainingConfig(BaseModel):
    """Model training configuration."""
    
    # Data settings
    min_training_samples: int = 100
    train_test_split: float = Field(ge=0.1, le=0.9, default=0.8)
    validation_split: float = Field(ge=0.1, le=0.5, default=0.2)
    
    # Feature settings
    feature_selection_threshold: float = 0.01
    enable_feature_scaling: bool = True
    handle_missing_values: str = Field(pattern="^(median|mean|drop)$", default="median")
    
    # Model settings
    model_type: str = Field(pattern="^(xgboost|lightgbm|random_forest)$", default="xgboost")
    cross_validation_folds: int = 5
    
    # XGBoost specific
    xgboost_params: dict = Field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    })
    
    # Training performance
    max_training_time_minutes: int = 30
    early_stopping_rounds: int = 50


class DynamicTrainingConfig(BaseModel):
    """Dynamic training configuration with rolling date ranges."""
    
    # Core training period settings
    default_months_back: int = Field(ge=12, le=36, default=20)  # 20 months for optimal balance
    min_months_back: int = Field(ge=6, le=24, default=12)  # Minimum fallback period
    max_months_back: int = Field(ge=18, le=48, default=30)  # Maximum extension period
    
    # Sample validation settings
    min_training_samples: int = Field(ge=50, default=50)
    target_training_samples: int = Field(ge=100, default=300)
    
    # Seasonal boundary settings
    respect_season_boundaries: bool = True
    season_start_month: int = Field(ge=1, le=12, default=8)  # August
    season_end_month: int = Field(ge=1, le=12, default=5)   # May
    
    # League-specific adjustments
    league_adjustments: dict = Field(default_factory=lambda: {
        # Major European leagues (longer seasons)
        2021: {"months_back": 22, "season_start": 8, "season_end": 5},  # Premier League
        2014: {"months_back": 22, "season_start": 8, "season_end": 5},  # La Liga
        2002: {"months_back": 22, "season_start": 8, "season_end": 5},  # Bundesliga
        2019: {"months_back": 22, "season_start": 8, "season_end": 5},  # Serie A
        2015: {"months_back": 22, "season_start": 8, "season_end": 5},  # Ligue 1
        
        # MLS (different season pattern)
        2020: {"months_back": 18, "season_start": 2, "season_end": 11},
        
        # Brazilian leagues (calendar year)
        2013: {"months_back": 18, "season_start": 1, "season_end": 12},
    })
    
    # Recency weighting settings
    enable_recency_weighting: bool = True
    recency_decay_factor: float = Field(ge=0.1, le=1.0, default=0.95)
    recent_months_boost: int = Field(ge=1, le=6, default=3)
    
    def calculate_training_dates(self, 
                               reference_date: Optional[datetime] = None,
                               league_id: Optional[int] = None,
                               league_ids: Optional[List[int]] = None) -> Tuple[datetime, datetime]:
        """Calculate dynamic training date range.
        
        Args:
            reference_date: Reference date for calculation (defaults to now)
            league_id: League ID for specific adjustments (deprecated, use league_ids)
            league_ids: List of league IDs for multi-league adjustments
            
        Returns:
            Tuple of (start_date, end_date) for training data
        """
        if reference_date is None:
            reference_date = datetime.now()
            
        # Handle backward compatibility
        if league_ids is None and league_id is not None:
            league_ids = [league_id]
            
        # Get league-specific settings with enhanced multi-league support
        months_back = self._get_league_adjusted_period(league_ids)
        season_start, season_end = self._get_league_season_pattern(league_ids)
        
        # Calculate initial date range
        end_date = reference_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date = reference_date - timedelta(days=months_back * 30.44)  # Average month length
        
        # Adjust for seasonal boundaries if enabled
        if self.respect_season_boundaries:
            start_date, end_date = self._adjust_for_season_boundaries(
                start_date, end_date, season_start, season_end
            )
        
        return start_date.replace(hour=0, minute=0, second=0, microsecond=0), end_date
    
    def _adjust_for_season_boundaries(self, 
                                    start_date: datetime, 
                                    end_date: datetime,
                                    season_start: int,
                                    season_end: int) -> Tuple[datetime, datetime]:
        """Adjust dates to respect football season boundaries.
        
        Args:
            start_date: Initial start date
            end_date: Initial end date
            season_start: Season start month (1-12)
            season_end: Season end month (1-12)
            
        Returns:
            Adjusted (start_date, end_date)
        """
        current_year = end_date.year
        current_month = end_date.month
        
        # Determine current season boundaries
        if season_start <= season_end:  # Same calendar year season
            if current_month >= season_start:
                season_end_date = datetime(current_year, season_end, 28)
            else:
                season_end_date = datetime(current_year - 1, season_end, 28)
        else:  # Cross-year season (e.g., Aug to May)
            if current_month >= season_start:
                season_end_date = datetime(current_year + 1, season_end, 28)
            else:
                season_end_date = datetime(current_year, season_end, 28)
        
        # Don't extend beyond current season end
        if end_date > season_end_date:
            end_date = season_end_date
        
        # Calculate how many complete seasons to include
        seasons_to_include = max(1, int(self.default_months_back / 10))  # ~2 seasons for 20 months
        
        # Adjust start date to season boundary
        if season_start <= season_end:
            target_start_year = end_date.year - seasons_to_include
            start_date = datetime(target_start_year, season_start, 1)
        else:
            if end_date.month >= season_start:
                target_start_year = end_date.year - seasons_to_include
            else:
                target_start_year = end_date.year - seasons_to_include - 1
            start_date = datetime(target_start_year, season_start, 1)
        
        return start_date, end_date
    
    def _get_league_adjusted_period(self, league_ids: Optional[List[int]] = None) -> int:
        """Get training period adjusted for specific leagues.
        
        Args:
            league_ids: List of league IDs
            
        Returns:
            Adjusted months back for training period
        """
        if not league_ids:
            return self.default_months_back
            
        # Enhanced league-specific adjustments based on season patterns and data availability
        league_adjustments = {
            # European top leagues (longer seasons, consistent data)
            2021: 20,  # Premier League
            2014: 20,  # La Liga  
            2002: 20,  # Bundesliga
            2019: 20,  # Serie A
            2015: 20,  # Ligue 1
            
            # Second tier European leagues (more volatile, need more data)
            2016: 22,  # Championship
            2017: 22,  # Serie B
            2018: 22,  # La Liga 2
            
            # MLS (shorter season, different pattern)
            2020: 24,  # MLS
            
            # International competitions (less frequent matches)
            2001: 30,  # Champions League
            2018: 30,  # Europa League
            
            # Emerging leagues (less predictable, need more historical context)
            2013: 26,  # Brazilian Serie A
            2012: 26,  # Argentine Primera
        }
        
        # Calculate weighted average based on leagues present
        total_adjustment = 0
        count = 0
        
        for league_id in league_ids:
            if league_id in league_adjustments:
                total_adjustment += league_adjustments[league_id]
                count += 1
                
        if count > 0:
            return int(total_adjustment / count)
        else:
            # Default for unknown leagues
            return self.default_months_back
            
    def _get_league_season_pattern(self, league_ids: Optional[List[int]] = None) -> Tuple[int, int]:
        """Get season start/end months for specific leagues.
        
        Args:
            league_ids: List of league IDs
            
        Returns:
            Tuple of (season_start_month, season_end_month)
        """
        if not league_ids:
            return self.season_start_month, self.season_end_month
            
        # League-specific season patterns
        season_patterns = {
            # European leagues (August to May)
            'european': {
                'start_month': 8, 
                'end_month': 5,
                'leagues': [2021, 2014, 2002, 2019, 2015, 2016, 2017, 2018]
            },
            # MLS (February/March to November)
            'mls': {
                'start_month': 3, 
                'end_month': 11,
                'leagues': [2020]
            },
            # Nordic leagues (March/April to October)
            'nordic': {
                'start_month': 4, 
                'end_month': 10,
                'leagues': [2113, 2119]
            },
            # South American (varies, often calendar year or March-December)
            'south_american': {
                'start_month': 3, 
                'end_month': 12,
                'leagues': [2013, 2012]
            },
            # International competitions (September to May/June)
            'international': {
                'start_month': 9,
                'end_month': 6, 
                'leagues': [2001, 2018]
            }
        }
        
        # Find the most relevant pattern
        for pattern_name, pattern_info in season_patterns.items():
            if any(league_id in pattern_info['leagues'] for league_id in league_ids):
                return pattern_info['start_month'], pattern_info['end_month']
                
        # Default pattern
        return self.season_start_month, self.season_end_month
    
    def get_recency_weights(self, match_dates: List[datetime], 
                          reference_date: Optional[datetime] = None) -> List[float]:
        """Calculate recency weights for training samples.
        
        Args:
            match_dates: List of match dates
            reference_date: Reference date for weight calculation
            
        Returns:
            List of weights (higher for more recent matches)
        """
        if not self.enable_recency_weighting:
            return [1.0] * len(match_dates)
            
        if reference_date is None:
            reference_date = datetime.now()
            
        weights = []
        recent_cutoff = reference_date - timedelta(days=self.recent_months_boost * 30.44)
        
        for match_date in match_dates:
            # Calculate days difference
            days_diff = (reference_date - match_date).days
            
            # Base exponential decay
            weight = self.recency_decay_factor ** (days_diff / 30.44)  # Monthly decay
            
            # Boost recent matches
            if match_date >= recent_cutoff:
                weight *= 1.5
                
            weights.append(max(0.1, weight))  # Minimum weight of 0.1
            
        return weights


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval_minutes: int = 5
    
    # Performance monitoring
    track_performance_metrics: bool = True
    metrics_retention_days: int = 30
    
    # Alerting thresholds
    api_quota_warning_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    data_quality_warning_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    computation_failure_threshold: float = Field(ge=0.0, le=1.0, default=0.2)
    
    # Notification settings
    enable_email_alerts: bool = False
    alert_email_recipients: List[str] = Field(default_factory=list)


class SchedulerConfig(BaseModel):
    """Task scheduler configuration."""
    
    # Feature computation schedule
    feature_computation_cron: str = "0 2 * * *"  # Daily at 2 AM
    historical_backfill_cron: str = "0 3 * * 0"  # Weekly on Sunday at 3 AM
    
    # Data quality checks
    quality_check_cron: str = "0 */6 * * *"  # Every 6 hours
    
    # Cache maintenance
    cache_cleanup_cron: str = "0 1 * * *"  # Daily at 1 AM
    
    # Monitoring
    metrics_aggregation_cron: str = "*/15 * * * *"  # Every 15 minutes
    
    # Scheduler settings
    max_concurrent_jobs: int = 3
    job_timeout_minutes: int = 120
    enable_job_persistence: bool = True


class RSSFeedConfig(BaseModel):
    """Configuration for a single RSS feed source."""
    name: str
    url: str
    enabled: bool = True
    update_interval_minutes: int = Field(ge=5, le=1440, default=60)  # 5 min to 24 hours
    timeout_seconds: int = Field(ge=5, le=120, default=30)
    max_retries: int = Field(ge=1, le=10, default=3)
    backoff_factor: float = Field(ge=1.0, le=5.0, default=2.0)
    priority: int = Field(ge=1, le=20, default=5)  # Lower number = higher priority
    
    # Content filtering
    keywords: List[str] = Field(default_factory=list)  # Required keywords
    exclude_keywords: List[str] = Field(default_factory=list)  # Excluded keywords
    min_article_length: int = Field(ge=50, default=100)
    max_article_age_hours: int = Field(ge=1, le=168, default=48)  # 1 hour to 1 week
    
    # Quality settings
    min_content_quality_score: float = Field(ge=0.0, le=1.0, default=0.6)
    require_team_mention: bool = True
    

class RSSContentFilterConfig(BaseModel):
    """RSS content filtering configuration."""
    # Language filtering
    allowed_languages: List[str] = Field(default_factory=lambda: ["en", "es", "fr", "de", "it"])
    detect_language: bool = True
    
    # Content quality filters
    min_word_count: int = Field(ge=10, default=50)
    max_word_count: int = Field(ge=100, default=2000)
    min_sentences: int = Field(ge=1, default=3)
    
    # Spam detection
    max_duplicate_percentage: float = Field(ge=0.0, le=1.0, default=0.8)
    spam_keywords: List[str] = Field(default_factory=lambda: [
        "click here", "subscribe now", "limited time", "act now",
        "free trial", "special offer", "advertisement", "sponsored"
    ])
    
    # Team name matching
    fuzzy_match_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    require_exact_team_match: bool = False
    team_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    

class RSSCacheConfig(BaseModel):
    """RSS content caching configuration."""
    # Cache retention
    article_retention_days: int = Field(ge=1, le=365, default=30)
    sentiment_cache_hours: int = Field(ge=1, le=168, default=24)
    feed_health_retention_days: int = Field(ge=1, le=90, default=7)
    
    # Cache performance
    max_cached_articles: int = Field(ge=100, default=10000)
    cache_cleanup_interval_hours: int = Field(ge=1, le=24, default=6)
    enable_compression: bool = True
    
    # Deduplication
    similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.85)
    title_similarity_weight: float = Field(ge=0.0, le=1.0, default=0.6)
    content_similarity_weight: float = Field(ge=0.0, le=1.0, default=0.4)
    

class RSSMonitoringConfig(BaseModel):
    """RSS monitoring and health check configuration."""
    # Health monitoring
    enable_feed_health_monitoring: bool = True
    health_check_interval_minutes: int = Field(ge=5, le=60, default=15)
    feed_timeout_threshold_seconds: int = Field(ge=10, le=300, default=60)
    
    # Quality monitoring
    enable_content_quality_monitoring: bool = True
    quality_check_sample_size: int = Field(ge=1, le=100, default=10)
    min_quality_score_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Sentiment monitoring
    enable_sentiment_monitoring: bool = True
    sentiment_accuracy_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    min_articles_for_sentiment: int = Field(ge=1, le=50, default=5)
    
    # Alerting
    enable_alerts: bool = True
    alert_on_feed_failure: bool = True
    alert_on_quality_drop: bool = True
    consecutive_failures_threshold: int = Field(ge=1, le=10, default=3)
    
    # Reporting
    generate_daily_reports: bool = True
    generate_weekly_reports: bool = True
    report_retention_days: int = Field(ge=7, le=365, default=30)
    

class RSSConfig(BaseModel):
    """Complete RSS system configuration."""
    # System settings
    enabled: bool = False
    fallback_priority: int = Field(ge=1, le=20, default=3)  # Priority when APIs fail
    max_concurrent_feeds: int = Field(ge=1, le=20, default=5)
    
    # Feed sources
    feeds: List[RSSFeedConfig] = Field(default_factory=lambda: [
        RSSFeedConfig(
            name="BBC Sport Football",
            url="http://feeds.bbci.co.uk/sport/football/rss.xml",
            keywords=["football", "soccer", "match", "game"],
            priority=1
        ),
        RSSFeedConfig(
            name="Sky Sports Football",
            url="https://www.skysports.com/rss/12040",
            keywords=["football", "soccer", "premier league"],
            priority=2
        ),
        RSSFeedConfig(
            name="ESPN Football",
            url="https://www.espn.com/espn/rss/soccer/news",
            keywords=["soccer", "football", "match"],
            priority=3
        ),
        # International Football Leagues
        # RSSFeedConfig(
        #     name="Albania Superliga",
        #     url="https://www.livesoccertv.com/rss/league/albania-superliga/",
        #     keywords=["albania", "superliga", "football"],
        #     priority=4
        # ),  # Disabled due to 403 Forbidden error
        RSSFeedConfig(
            name="Argentina Liga Profesional",
            url="https://www.ole.com.ar/rss/futbol-primera/",
            keywords=["argentina", "liga profesional", "football"],
            priority=4
        ),
        RSSFeedConfig(
            name="Brazil Serie A",
            url="https://feeds.folha.uol.com.br/emcimadahora/rss091.xml",
            keywords=["brazil", "serie a", "football"],
            priority=4
        ),
        # English Football Leagues
        RSSFeedConfig(
            name="Premier League News",
            url="http://feeds.feedburner.com/PremierLeagueFootballNews",
            keywords=["premier league", "england", "football"],
            priority=3
        ),
        RSSFeedConfig(
            name="Championship News",
            url="http://feeds.feedburner.com/ChampionshipFootballNews",
            keywords=["championship", "england", "football"],
            priority=4
        ),
        RSSFeedConfig(
            name="League One News",
            url="http://feeds.feedburner.com/LeagueOneFootballNews",
            keywords=["league one", "england", "football"],
            priority=5
        ),
        RSSFeedConfig(
            name="League Two News",
            url="http://feeds.feedburner.com/LeagueTwoFootballNews",
            keywords=["league two", "england", "football"],
            priority=5
        ),
        # German Football Leagues
        RSSFeedConfig(
            name="Bundesliga Kicker",
            url="https://newsfeed.kicker.de/news/bundesliga",
            keywords=["bundesliga", "germany", "football"],
            priority=3
        ),
        RSSFeedConfig(
            name="2. Bundesliga Kicker",
            url="https://newsfeed.kicker.de/news/2-bundesliga",
            keywords=["2 bundesliga", "germany", "football"],
            priority=4
        ),
        RSSFeedConfig(
            name="3. Liga Kicker",
            url="https://newsfeed.kicker.de/news/3-liga",
            keywords=["3 liga", "germany", "football"],
            priority=5
        ),
        # International Competitions
        RSSFeedConfig(
            name="Copa Libertadores",
            url="https://www.ole.com.ar/rss/futbol-internacional/libertadores",
            keywords=["copa libertadores", "south america", "football"],
            priority=3
        ),
        RSSFeedConfig(
            name="Liga MX Reddit",
            url="https://www.reddit.com/r/LigaMX/.rss",
            keywords=["liga mx", "mexico", "football"],
            priority=4
        ),
        # RSSFeedConfig(
        #     name="Portugal Primeira Liga",
        #     url="https://rss.com/podcasts/longballfutebol/1611201/",
        #     keywords=["primeira liga", "portugal", "football"],
        #     priority=4
        # )  # Disabled due to XML syntax error
    ])
    
    # Configuration components
    content_filter: RSSContentFilterConfig = Field(default_factory=RSSContentFilterConfig)
    cache: RSSCacheConfig = Field(default_factory=RSSCacheConfig)
    monitoring: RSSMonitoringConfig = Field(default_factory=RSSMonitoringConfig)
    
    # Integration settings
    sentiment_integration: bool = True
    sentiment_weight: float = Field(ge=0.0, le=1.0, default=0.2)  # Weight in final prediction
    fallback_timeout_seconds: int = Field(ge=30, le=300, default=120)
    
    def get_enabled_feeds(self) -> List[RSSFeedConfig]:
        """Get list of enabled feeds sorted by priority."""
        enabled_feeds = [feed for feed in self.feeds if feed.enabled]
        return sorted(enabled_feeds, key=lambda x: x.priority)
    
    def get_feed_by_name(self, name: str) -> Optional[RSSFeedConfig]:
        """Get feed configuration by name."""
        for feed in self.feeds:
            if feed.name == name:
                return feed
        return None


class FormFinderConfig(BaseSettings):
    """Main FormFinder configuration."""
    api: APIConfig
    processing: ProcessingConfig
    database: DatabaseConfig
    notifications: NotificationsConfig
    logging: LoggingConfig
    directories: DirectoriesConfig
    workflow: WorkflowConfig
    testing: TestingConfig
    sentiment_analysis: SentimentAnalysisConfig = Field(default_factory=SentimentAnalysisConfig)
    feature_computation: FeatureComputationConfig = Field(default_factory=FeatureComputationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dynamic_training: DynamicTrainingConfig = Field(default_factory=DynamicTrainingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    rss: RSSConfig = Field(default_factory=RSSConfig)
    leagues_path: str = "leagues.json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "FormFinderConfig":
        """Load configuration from YAML file with environment variable substitution."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Substitute environment variables
        config_data = cls._substitute_env_vars(config_data)
        
        return cls(**config_data)

    @staticmethod
    def _substitute_env_vars(data):
        """Recursively substitute environment variables in configuration data."""
        if isinstance(data, dict):
            return {key: FormFinderConfig._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [FormFinderConfig._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            env_var = data[2:-1]
            return os.getenv(env_var, data)  # Return original if env var not found
        else:
            return data

    def get_database_url(self) -> str:
        """Get the database URL based on configuration."""
        if self.database.type == "sqlite":
            return f"sqlite:///{self.database.sqlite.path}"
        elif self.database.type == "postgresql":
            pg_config = self.database.postgresql
            return f"postgresql://{pg_config.username}:{pg_config.password}@{pg_config.host}:{pg_config.port}/{pg_config.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.directories.data_root,
            self.directories.logs,
            self.directories.cache,
            self.directories.predictions,
            self.directories.historical,
            self.directories.fixtures,
            self.directories.standings,
            self.directories.processed_data,
            self.testing.test_data_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_league_ids(self) -> List[int]:
        """Get league IDs using the configured leagues_path."""
        return self.processing.get_league_ids(self.leagues_path)


# Global configuration instance
_config: Optional[FormFinderConfig] = None


def get_config() -> FormFinderConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config


def load_config(config_path: Union[str, Path] = "config.yaml") -> FormFinderConfig:
    """Load and set the global configuration."""
    global _config
    _config = FormFinderConfig.from_yaml(config_path)
    # _config.ensure_directories()
    return _config


def reload_config(config_path: Union[str, Path] = "config.yaml") -> FormFinderConfig:
    """Reload the global configuration."""
    return load_config(config_path)