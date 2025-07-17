"""Pytest configuration and shared fixtures for FormFinder tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from formfinder.config import FormFinderConfig, load_config
from formfinder.database import Base, DatabaseManager, init_database


@pytest.fixture(scope="session")
def test_config() -> FormFinderConfig:
    """Create test configuration with isolated directories."""
    # Create isolated temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    
    config_data = {
        "api": {
            "auth_token": "test_token",
            "base_url": "https://api.test.com",
            "rate_limit_requests": 100,
            "rate_limit_period": 60,
            "timeout": 10,
            "max_retries": 1
        },
        "processing": {
            "league_ids": [203, 204, 205],
            "season_year": "2024-2025",
            "max_concurrent_requests": 2,
            "inter_league_delay": 0,
            "cache_ttl_hours": 1,
            "recent_period": 5,
            "win_rate_threshold": 0.70
        },
        "database": {
            "type": "sqlite",
            "sqlite": {
                "path": f"{temp_dir}/test_formfinder.db"
            }
        },
        "notifications": {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.test.com",
                "smtp_port": 587,
                "username": "test@test.com",
                "password": "test_password",
                "from_address": "test@test.com",
                "to_addresses": ["test@test.com"],
                "max_matches_in_email_body": 5
            },
            "sms": {
                "enabled": False,
                "provider": "twilio",
                "max_matches_in_sms": 1
            },
            "predictions_dir": f"{temp_dir}/predictions"
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_handler": {
                "enabled": False,
                "directory": f"{temp_dir}/logs",
                "max_bytes": 1048576,
                "backup_count": 2
            },
            "console_handler": {
                "enabled": True
            }
        },
        "directories": {
            "data_root": temp_dir,
            "logs": f"{temp_dir}/logs",
            "cache": f"{temp_dir}/cache",
            "predictions": f"{temp_dir}/predictions",
            "historical": f"{temp_dir}/historical",
            "fixtures": f"{temp_dir}/fixtures",
            "standings": f"{temp_dir}/standings",
            "processed_data": f"{temp_dir}/processed_data"
        },
        "workflow": {
            "orchestrator": "local",
            "retry_attempts": 1,
            "retry_delay_seconds": 1,
            "task_timeout_seconds": 60,
            "enable_notifications": False
        },
        "testing": {
            "test_database_url": f"sqlite:///{temp_dir}/test_formfinder.db",
            "mock_api_responses": True,
            "test_data_dir": f"{temp_dir}/test_data"
        }
    }
    
    config = FormFinderConfig(**config_data)
    # Set as global config for workflows that use get_config()
    import formfinder.config
    formfinder.config._config = config
    return config


@pytest.fixture(scope="session", autouse=True)
def load_global_config(test_config):
    """Ensure configuration is loaded globally for all tests."""
    from formfinder.config import load_config
    import formfinder.config
    # Set the global config to our test config
    formfinder.config._config = test_config
    yield test_config
    # Clean up after tests
    formfinder.config._config = None


@pytest.fixture(scope="session")
def test_db_engine(test_config):
    """Create test database engine."""
    database_url = test_config.get_database_url()
    
    # Enable foreign key constraints for SQLite
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url, 
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        
        # Enable foreign key constraints for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        engine = create_engine(database_url, echo=False)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session() -> Generator[Session, None, None]:
    """Create test database session with isolated temporary database."""
    # Create a unique temporary SQLite database file for this test
    # This ensures isolation from other tests and from prefect-test.db
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)  # Close the file descriptor immediately to allow SQLAlchemy to open it

    db_url = f"sqlite:///{path}"
    engine = create_engine(
        db_url,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    # Enable foreign key constraints for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    # Create your application's tables in this DB
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        # Retry mechanism for cleanup to handle file locking issues on Windows
        for _ in range(3):
            try:
                if os.path.exists(path):
                    os.unlink(path)
                break
            except PermissionError:
                import time
                time.sleep(0.1)
        else:
            if os.path.exists(path):
                print(f"Warning: Could not unlink temporary database file {path} after multiple retries.")


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        "fixtures": [
            {
                "fixture_id": 1,
                "league_id": 203,
                "home_team": {"team_id": 1, "name": "Team A"},
                "away_team": {"team_id": 2, "name": "Team B"},
                "event_date": "2024-01-15",
                "event_time": "15:00",
                "status": "scheduled"
            },
            {
                "fixture_id": 2,
                "league_id": 203,
                "home_team": {"team_id": 2, "name": "Team B"},
                "away_team": {"team_id": 3, "name": "Team C"},
                "event_date": "2024-01-16",
                "event_time": "18:00",
                "status": "finished",
                "goalsHomeTeam": 2,
                "goalsAwayTeam": 1
            }
        ],
        "standings": [
            {
                "team_id": 1,
                "teamName": "Team A",
                "position": 1,
                "played": 10,
                "won": 8,
                "draw": 1,
                "lost": 1,
                "goals_for": 25,
                "goals_against": 8,
                "goal_difference": 17,
                "points": 25
            },
            {
                "team_id": 2,
                "teamName": "Team B",
                "position": 2,
                "played": 10,
                "won": 6,
                "draw": 2,
                "lost": 2,
                "goals_for": 18,
                "goals_against": 12,
                "goal_difference": 6,
                "points": 20
            }
        ],
        "teams": [
            {
                "team_id": 1,
                "name": "Team A",
                "short_name": "TEA",
                "logo": "https://example.com/team_a.png"
            },
            {
                "team_id": 2,
                "name": "Team B",
                "short_name": "TEB",
                "logo": "https://example.com/team_b.png"
            }
        ]
    }


@pytest.fixture
def sample_predictions():
    """Sample prediction data for testing."""
    return [
        {
            "fixture_id": 1,
            "home_team": "Team A",
            "away_team": "Team B",
            "match_date": "2024-01-15",
            "home_win_probability": 0.65,
            "draw_probability": 0.20,
            "away_win_probability": 0.15,
            "predicted_home_score": 2.1,
            "predicted_away_score": 0.8,
            "confidence_score": 0.82,
            "home_team_form_score": 0.85,
            "away_team_form_score": 0.45
        },
        {
            "fixture_id": 2,
            "home_team": "Team C",
            "away_team": "Team D",
            "match_date": "2024-01-16",
            "home_win_probability": 0.45,
            "draw_probability": 0.30,
            "away_win_probability": 0.25,
            "predicted_home_score": 1.5,
            "predicted_away_score": 1.2,
            "confidence_score": 0.68,
            "home_team_form_score": 0.60,
            "away_team_form_score": 0.55
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment(test_config, monkeypatch):
    """Setup test environment for each test."""
    # Ensure test directories exist
    test_config.ensure_directories()
    
    # Set environment variables for testing
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EMAIL_USERNAME", "test@test.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "test_password")
    monkeypatch.setenv("EMAIL_FROM", "test@test.com")
    monkeypatch.setenv("EMAIL_TO", "test@test.com")


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Pytest markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.timeout(300)  # 5 minute timeout for all tests
]


# Test coverage configuration
pytest_plugins = ["pytest_cov"]