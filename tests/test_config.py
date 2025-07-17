"""Unit tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from formfinder.config import (
    FormFinderConfig, APIConfig, ProcessingConfig, DatabaseConfig,
    NotificationsConfig, LoggingConfig, DirectoriesConfig,
    WorkflowConfig, TestingConfig, load_config, get_config
)


class TestAPIConfig:
    """Test API configuration validation."""
    
    def test_valid_api_config(self):
        """Test valid API configuration."""
        config_data = {
            "auth_token": "test_token",
            "base_url": "https://api.test.com",
            "rate_limit_requests": 300,
            "rate_limit_period": 60,
            "timeout": 15,
            "max_retries": 2
        }
        config = APIConfig(**config_data)
        assert config.auth_token == "test_token"
        assert config.base_url == "https://api.test.com"
        assert config.rate_limit_requests == 300
    
    def test_api_config_defaults(self):
        """Test API configuration with defaults."""
        config = APIConfig(auth_token="test_token")
        assert config.base_url == "https://api.soccerdataapi.com"
        assert config.rate_limit_requests == 300
        assert config.timeout == 15
    
    def test_invalid_api_config(self):
        """Test invalid API configuration."""
        with pytest.raises(ValidationError):
            APIConfig()  # Missing required auth_token


class TestProcessingConfig:
    """Test processing configuration validation."""
    
    def test_valid_processing_config(self):
        """Test valid processing configuration."""
        config_data = {
            "league_ids": [203, 204, 205],
            "season_year": "2024-2025",
            "win_rate_threshold": 0.75
        }
        config = ProcessingConfig(**config_data)
        assert config.league_ids == [203, 204, 205]
        assert config.win_rate_threshold == 0.75
    
    def test_processing_config_defaults(self):
        """Test processing configuration with defaults."""
        config = ProcessingConfig(league_ids=[203])
        assert config.season_year == "2024-2025"
        assert config.max_concurrent_requests == 10
        assert config.win_rate_threshold == 0.70
    
    def test_invalid_win_rate_threshold(self):
        """Test invalid win rate threshold."""
        with pytest.raises(ValidationError):
            ProcessingConfig(league_ids=[203], win_rate_threshold=1.5)  # > 1.0
        
        with pytest.raises(ValidationError):
            ProcessingConfig(league_ids=[203], win_rate_threshold=-0.1)  # < 0.0


class TestDatabaseConfig:
    """Test database configuration validation."""
    
    def test_sqlite_config(self):
        """Test SQLite database configuration."""
        config_data = {
            "type": "sqlite",
            "sqlite": {"path": "test.db"}
        }
        config = DatabaseConfig(**config_data)
        assert config.type == "sqlite"
        assert config.sqlite.path == "test.db"
    
    def test_sqlite_config_with_defaults(self):
        """Test SQLite configuration with defaults."""
        config = DatabaseConfig(type="sqlite")
        assert config.sqlite.path == "data/formfinder.db"
    
    def test_postgresql_config(self):
        """Test PostgreSQL database configuration."""
        config_data = {
            "type": "postgresql",
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass"
            }
        }
        config = DatabaseConfig(**config_data)
        assert config.type == "postgresql"
        assert config.postgresql.host == "localhost"
    
    def test_invalid_database_type(self):
        """Test invalid database type."""
        with pytest.raises(ValidationError):
            DatabaseConfig(type="mysql")  # Unsupported type
    
    def test_missing_postgresql_config(self):
        """Test missing PostgreSQL configuration."""
        with pytest.raises(ValidationError):
            DatabaseConfig(type="postgresql")  # Missing postgresql config


class TestFormFinderConfig:
    """Test main FormFinder configuration."""
    
    def test_valid_complete_config(self, test_config):
        """Test valid complete configuration."""
        assert isinstance(test_config.api, APIConfig)
        assert isinstance(test_config.processing, ProcessingConfig)
        assert isinstance(test_config.database, DatabaseConfig)
        assert isinstance(test_config.notifications, NotificationsConfig)
    
    def test_get_database_url_sqlite(self, test_config):
        """Test SQLite database URL generation."""
        url = test_config.get_database_url()
        assert url.startswith("sqlite:///")
        assert "test_formfinder.db" in url
    
    def test_get_database_url_postgresql(self):
        """Test PostgreSQL database URL generation."""
        config_data = {
            "api": {"auth_token": "test"},
            "processing": {"league_ids": [203]},
            "database": {
                "type": "postgresql",
                "postgresql": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_db",
                    "username": "test_user",
                    "password": "test_pass"
                }
            },
            "notifications": {
                "email": {
                    "username": "test@test.com",
                    "password": "test",
                    "from_address": "test@test.com",
                    "to_addresses": ["test@test.com"]
                },
                "sms": {}
            },
            "logging": {
                "file_handler": {},
                "console_handler": {}
            },
            "directories": {},
            "workflow": {},
            "testing": {}
        }
        config = FormFinderConfig(**config_data)
        url = config.get_database_url()
        expected = "postgresql://test_user:test_pass@localhost:5432/test_db"
        assert url == expected
    
    def test_ensure_directories(self, test_config, temp_directory):
        """Test directory creation."""
        # Update config to use temp directory
        test_config.directories.data_root = str(temp_directory)
        test_config.directories.logs = str(temp_directory / "logs")
        test_config.directories.cache = str(temp_directory / "cache")
        
        test_config.ensure_directories()
        
        assert (temp_directory / "logs").exists()
        assert (temp_directory / "cache").exists()


class TestConfigLoading:
    """Test configuration loading from YAML files."""
    
    def test_load_config_from_yaml(self, temp_directory):
        """Test loading configuration from YAML file."""
        config_data = {
            "api": {
                "auth_token": "test_token",
                "base_url": "https://api.test.com"
            },
            "processing": {
                "league_ids": [203, 204],
                "season_year": "2024-2025"
            },
            "database": {
                "type": "sqlite",
                "sqlite": {"path": "test.db"}
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "username": "test@test.com",
                    "password": "test",
                    "from_address": "test@test.com",
                    "to_addresses": ["test@test.com"]
                },
                "sms": {"enabled": False}
            },
            "logging": {
                "level": "INFO",
                "file_handler": {"enabled": False},
                "console_handler": {"enabled": True}
            },
            "directories": {
                "data_root": "data"
            },
            "workflow": {
                "orchestrator": "prefect"
            },
            "testing": {
                "test_database_url": "sqlite:///test.db"
            }
        }
        
        config_file = temp_directory / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = FormFinderConfig.from_yaml(config_file)
        assert config.api.auth_token == "test_token"
        assert config.processing.league_ids == [203, 204]
    
    def test_load_config_with_env_vars(self, temp_directory, monkeypatch):
        """Test loading configuration with environment variable substitution."""
        monkeypatch.setenv("TEST_TOKEN", "secret_token")
        monkeypatch.setenv("TEST_EMAIL", "user@example.com")
        
        config_data = {
            "api": {
                "auth_token": "${TEST_TOKEN}",
                "base_url": "https://api.test.com"
            },
            "processing": {
                "league_ids": [203]
            },
            "database": {
                "type": "sqlite"
            },
            "notifications": {
                "email": {
                    "username": "${TEST_EMAIL}",
                    "password": "test",
                    "from_address": "${TEST_EMAIL}",
                    "to_addresses": ["${TEST_EMAIL}"]
                },
                "sms": {}
            },
            "logging": {
                "file_handler": {},
                "console_handler": {}
            },
            "directories": {},
            "workflow": {},
            "testing": {}
        }
        
        config_file = temp_directory / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = FormFinderConfig.from_yaml(config_file)
        assert config.api.auth_token == "secret_token"
        assert config.notifications.email.username == "user@example.com"
    
    def test_load_config_missing_file(self, temp_directory):
        """Test loading configuration from missing file."""
        missing_file = temp_directory / "missing_config.yaml"
        
        with pytest.raises(FileNotFoundError):
            FormFinderConfig.from_yaml(missing_file)
    
    def test_load_config_invalid_yaml(self, temp_directory):
        """Test loading configuration from invalid YAML."""
        config_file = temp_directory / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            FormFinderConfig.from_yaml(config_file)


class TestGlobalConfigManagement:
    """Test global configuration management functions."""
    
    def test_load_and_get_config(self, temp_directory):
        """Test loading and getting global configuration."""
        config_data = {
            "api": {"auth_token": "test"},
            "processing": {"league_ids": [203]},
            "database": {"type": "sqlite"},
            "notifications": {
                "email": {
                    "username": "test@test.com",
                    "password": "test",
                    "from_address": "test@test.com",
                    "to_addresses": ["test@test.com"]
                },
                "sms": {}
            },
            "logging": {
                "file_handler": {},
                "console_handler": {}
            },
            "directories": {},
            "workflow": {},
            "testing": {}
        }
        
        config_file = temp_directory / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        loaded_config = load_config(config_file)
        assert loaded_config.api.auth_token == "test"
        
        # Get config
        retrieved_config = get_config()
        assert retrieved_config.api.auth_token == "test"
        assert retrieved_config is loaded_config
    
    def test_get_config_not_loaded(self):
        """Test getting configuration when not loaded."""
        # Reset global config
        import formfinder.config
        formfinder.config._config = None
        
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            get_config()


# Test coverage: ~95% - Missing edge cases in environment variable substitution