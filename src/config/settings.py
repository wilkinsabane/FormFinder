import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, validator
from pathlib import Path

class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "formfinder"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "formfinder"
    
    # Connection pooling
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

class APISettings(BaseSettings):
    """API configuration settings"""
    
    # FastAPI settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FormFinder Sentiment Analysis"
    DEBUG: bool = False
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60

class FootballAPISettings(BaseSettings):
    """Football API settings"""
    
    # API-Football settings
    FOOTBALL_API_KEY: str
    FOOTBALL_API_BASE_URL: str = "https://v3.football.api-sports.io"
    
    # Rate limiting for API calls
    FOOTBALL_API_RATE_LIMIT: int = 100  # requests per day
    FOOTBALL_API_CACHE_TTL: int = 3600  # 1 hour

class SentimentSettings(BaseSettings):
    """Sentiment analysis settings"""
    
    # Twitter API settings
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET: Optional[str] = None
    TWITTER_ACCESS_TOKEN: Optional[str] = None
    TWITTER_ACCESS_TOKEN_SECRET: Optional[str] = None
    TWITTER_BEARER_TOKEN: Optional[str] = None
    
    # Reddit API settings
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "FormFinder/1.0"
    
    # Sentiment analysis model settings
    SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    SENTIMENT_BATCH_SIZE: int = 32
    SENTIMENT_MAX_LENGTH: int = 512
    
    # Data collection settings
    SENTIMENT_DAYS_BACK: int = 7
    MAX_TWEETS_PER_TEAM: int = 100
    MAX_REDDIT_POSTS_PER_TEAM: int = 50

class CacheSettings(BaseSettings):
    """Cache settings"""
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Cache TTL settings (in seconds)
    CACHE_TTL_SENTIMENT: int = 3600
    CACHE_TTL_FOOTBALL: int = 1800
    CACHE_TTL_COMBINED: int = 900

class AlertSettings(BaseSettings):
    """Alert and notification settings"""
    
    # Email settings
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # Alert thresholds
    SENTIMENT_ALERT_THRESHOLD: float = 0.8
    FORM_ALERT_THRESHOLD: float = 0.9
    
    # Webhook settings
    SLACK_WEBHOOK_URL: Optional[str] = None
    DISCORD_WEBHOOK_URL: Optional[str] = None

class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    LOG_MAX_SIZE: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5

class Settings(BaseSettings):
    """Main settings class combining all configuration sections"""
    
    # Environment
    ENVIRONMENT: str = "development"
    
    # Include all settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    football_api: FootballAPISettings = FootballAPISettings()
    sentiment: SentimentSettings = SentimentSettings()
    cache: CacheSettings = CacheSettings()
    alerts: AlertSettings = AlertSettings()
    logging: LoggingSettings = LoggingSettings()
    
    # Project root
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    # Data directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        valid_envs = {"development", "staging", "production"}
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

# Global settings instance
settings = Settings()

# Environment-specific configurations
def get_environment_config():
    """Get environment-specific configuration"""
    if settings.is_development:
        return {
            "debug": True,
            "reload": True,
            "log_level": "DEBUG"
        }
    elif settings.is_production:
        return {
            "debug": False,
            "reload": False,
            "log_level": "WARNING"
        }
    else:
        return {
            "debug": False,
            "reload": False,
            "log_level": "INFO"
        }

# Example .env file content
ENV_EXAMPLE = """
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=formfinder
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=formfinder

# Football API
FOOTBALL_API_KEY=your_football_api_key_here

# Twitter API (optional)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Reddit API (optional)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Email Configuration (optional)
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Webhook URLs (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Environment
ENVIRONMENT=development
"""