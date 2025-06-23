from pydantic import BaseModel, Field, FilePath, DirectoryPath
from typing import List, Optional, Dict
import yaml
import os

# Helper to construct default paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class APIConfig(BaseModel):
    auth_token: str = Field(..., min_length=1)
    base_url: str = "https://api.soccerdataapi.com"
    rate_limit_requests: int = Field(100, ge=1)
    rate_limit_period: int = Field(3600, ge=1) # seconds
    timeout: int = Field(30, ge=1) # seconds
    max_retries: int = Field(5, ge=1)

class DataFetcherProcessingConfig(BaseModel):
    league_ids: List[int] = Field(..., min_items=1)
    season_year: str = Field(..., pattern=r'^(\d{4}|\d{4}-\d{4})$')
    max_concurrent_requests: int = Field(5, ge=1, le=20)
    # inter_league_delay is no longer used as tasks are per league type
    cache_ttl_hours: int = Field(24, ge=1)

class DataFetcherAppConfig(BaseModel):
    api: APIConfig
    processing: DataFetcherProcessingConfig
    # Directories for logs and cache, data files are now in DB
    log_dir: DirectoryPath = Field(default=os.path.join(PROJECT_ROOT, "data/logs"))
    cache_dir: DirectoryPath = Field(default=os.path.join(PROJECT_ROOT, "data/cache"))


class DataProcessorAppConfig(BaseModel):
    recent_period: int = Field(10, ge=1)
    win_rate_threshold: float = Field(0.7, ge=0.0, le=1.0)
    season_year: str # This should match DataFetcher's season_year
    log_dir: DirectoryPath = Field(default=os.path.join(PROJECT_ROOT, "data/logs"))


class PredictorOutputterAppConfig(BaseModel):
    days_ahead: int = Field(7, ge=1)
    output_dir: DirectoryPath = Field(default=os.path.join(PROJECT_ROOT, "data/predictions"))
    season_year: str # Match DataFetcher/Processor
    recent_period_for_form: int # Match DataProcessor's recent_period
    log_dir: DirectoryPath = Field(default=os.path.join(PROJECT_ROOT, "data/logs"))


class EmailNotifierConfig(BaseModel):
    enabled: bool = True
    sender_email: Optional[str] = None
    sender_password: Optional[str] = None
    receiver_email: Optional[str] = None
    smtp_server: Optional[str] = "smtp.gmail.com"
    smtp_port: Optional[int] = 587
    subject_prefix: str = "[FormFinder]"
    max_matches_in_email_body: Optional[int] = 20

class SMSNotifierConfig(BaseModel):
    enabled: bool = False
    service_provider: str = "twilio" # Example
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    receiver_phone_number: Optional[str] = None
    max_matches_in_sms: Optional[int] = 1


class NotifierAppConfig(BaseModel):
    email: EmailNotifierConfig = EmailNotifierConfig()
    sms: SMSNotifierConfig = SMSNotifierConfig()
    predictions_dir: DirectoryPath = Field(default=os.path.join(PROJECT_ROOT, "data/predictions"))
    log_dir: DirectoryPath = Field(default=os.path.join(PROJECT_ROOT, "data/logs"))


class AppConfig(BaseModel):
    data_fetcher: DataFetcherAppConfig
    data_processor: DataProcessorAppConfig
    predictor_outputter: PredictorOutputterAppConfig
    notifier: NotifierAppConfig
    database_url: str = f"sqlite:///{os.path.join(PROJECT_ROOT, 'formfinder.db')}"

    @classmethod
    deffrom_yaml(cls, config_path: FilePath) -> 'AppConfig':
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            if data is None: # Handle empty YAML file
                raise ValueError(f"Configuration file is empty or invalid: {config_path}")
        return cls(**data)

# Global config instance (can be loaded once and imported by other modules)
# Load default config path, can be overridden by environment variable if needed
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
CONFIG_FILE_PATH = os.getenv("FORM FINDER_CONFIG_PATH", DEFAULT_CONFIG_PATH)

# Initialize a global config object.
# Modules can `from config.config import app_config`
# This will raise an error at import time if config.yaml is missing or invalid,
# which is good for failing fast.
try:
    app_config = AppConfig.from_yaml(CONFIG_FILE_PATH)
except FileNotFoundError:
    # This is a fallback for when config.yaml might not exist yet (e.g. first run, tests)
    # In a real scenario, you'd ensure config.yaml exists or handle this more gracefully.
    print(f"WARNING: Configuration file {CONFIG_FILE_PATH} not found. Some components might not work without it or use defaults.")
    # Create a dummy AppConfig if needed for parts of the code to import without error,
    # but this is risky as it won't have real values.
    # For now, let it be None or raise, to make it explicit that config is missing.
    app_config = None # Or raise an error immediately.
    # raise FileNotFoundError(f"CRITICAL: Configuration file {CONFIG_FILE_PATH} not found. Application cannot start.")
except Exception as e:
    print(f"CRITICAL: Error loading configuration from {CONFIG_FILE_PATH}: {e}")
    app_config = None # Or raise
    # raise


if __name__ == "__main__":
    # Example of loading and accessing config
    # This part will only run if config.yaml exists and is valid
    if app_config:
        print("Config loaded successfully!")
        print(f"Data Fetcher API Token: {app_config.data_fetcher.api.auth_token[:5]}...") # Print only a snippet
        print(f"Data Processor Recent Period: {app_config.data_processor.recent_period}")
        print(f"Notifier Email Enabled: {app_config.notifier.email.enabled}")
        print(f"Database URL: {app_config.database_url}")

        # Show how Pydantic creates default directories if they are part of the model
        # (though these are just strings until used to create dirs by the app)
        print(f"Log dir for fetcher: {app_config.data_fetcher.log_dir}")

        # Test creation of directories from config (example)
        # os.makedirs(app_config.data_fetcher.log_dir, exist_ok=True)
        # print(f"Ensured log directory exists: {app_config.data_fetcher.log_dir}")
    else:
        print("Failed to load app_config. Check warnings/errors above.")
