"""Configuration management using Pydantic for validation and YAML loading."""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class APIConfig(BaseModel):
    """API configuration settings."""
    auth_token: str
    base_url: str = "https://api.soccerdataapi.com"
    rate_limit_requests: int = 300
    rate_limit_period: int = 60
    timeout: int = 15
    max_retries: int = 2


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