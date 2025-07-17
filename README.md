# FormFinder 2.0

🚀 **A next-generation football prediction system with database storage, workflow orchestration, and comprehensive testing.**

FormFinder 2.0 represents a complete architectural overhaul, transforming from a simple script-based system into a robust, production-ready data pipeline with modern DevOps practices.

## ✨ Key Features

### 🏗️ **Modern Architecture**
- **Database Storage**: SQLite for development, PostgreSQL for production
- **Workflow Orchestration**: Prefect-powered data pipelines with automatic retries and monitoring
- **Centralized Configuration**: Single YAML configuration with Pydantic validation
- **Comprehensive Testing**: Unit, integration, and performance tests with 90%+ coverage

### 📊 **Core Functionality**
- **Multi-League Support**: Premier League, La Liga, Bundesliga, Serie A, Ligue 1 (configurable via leagues.json)
- **Real-time Data Fetching**: Automated data collection with rate limiting and caching
- **Advanced Analytics**: Statistical analysis and prediction generation
- **Smart Notifications**: Email and SMS alerts for predictions and system status
- **Performance Monitoring**: Built-in health checks and performance metrics

### 🔧 **Developer Experience**
- **CLI Interface**: Comprehensive command-line tools for all operations
- **CI/CD Pipeline**: Automated testing, linting, and security checks
- **Docker Support**: Containerized deployment with Docker Compose
- **Rich Documentation**: API docs, tutorials, and deployment guides (see ARCHITECTURE.md, CHANGELOG.md, README_POSTGRESQL_SETUP.md)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- Git
- (Optional) PostgreSQL for production deployment
- (Optional) Docker for containerized deployment

### Installation

1. **Clone and Install**:
```bash
git clone <repository-url>
cd FormFinder
pip install -e .[dev]  # Install in development mode with all dependencies
```

2. **Initialize Configuration**:
```bash
formfinder config-template -o config.yaml
# Edit config.yaml with your API token and preferences
```

3. **Initialize System**:
```bash
formfinder init  # Creates database and directories
```

4. **Run Your First Pipeline**:
```bash
make run  # Execute the main prediction pipeline using Makefile
```

### Quick Commands

```bash
# Run health checks
formfinder health-check

# Quick update for recent data
formfinder quick-update

# Schedule automatic daily runs
formfinder schedule

# Database management
formfinder db status
formfinder db reset

# View all available commands
formfinder --help
```

## 📋 Configuration

FormFinder 2.0 uses a single, comprehensive YAML configuration file:

```yaml
api:
  token: "YOUR_API_TOKEN"  # Required: Get from football-data.org
  league_ids: [2021, 2014, 2002, 2019, 2015]  # Premier League, La Liga, etc.

database:
  type: "sqlite"  # or "postgresql" for production
  sqlite:
    path: "data/formfinder.db"

workflow:
  max_retries: 3
  task_timeout: 3600

notifications:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    # ... additional email settings
```

Generate a complete template:
```bash
formfinder config-template > config.yaml
```

## 🏗️ Architecture Overview

### Database Schema
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Leagues   │    │    Teams     │    │  Fixtures   │
├─────────────┤    ├──────────────┤    ├─────────────┤
│ id (PK)     │◄──►│ id (PK)      │◄──►│ id (PK)     │
│ name        │    │ name         │    │ home_team   │
│ country     │    │ league_id    │    │ away_team   │
│ season      │    │ founded      │    │ match_date  │
└─────────────┘    └──────────────┘    │ status      │
                                       │ home_score  │
                                       │ away_score  │
                                       └─────────────┘
```

### Workflow Pipeline
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Fetch  │───►│ Processing  │───►│ Prediction  │───►│ Notification│
│             │    │             │    │ Generation  │    │             │
│ • API calls │    │ • Clean data│    │ • ML models │    │ • Email/SMS │
│ • Rate limit│    │ • Transform │    │ • Analytics │    │ • Alerts    │
│ • Cache     │    │ • Validate  │    │ • Confidence│    │ • Reports   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 🧪 Testing

FormFinder 2.0 includes comprehensive testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=formfinder --cov-report=html

# Run specific test categories
pytest tests/test_database.py      # Database tests
pytest tests/test_workflows.py     # Workflow tests
pytest tests/test_integration.py   # Integration tests

# Performance benchmarks
pytest tests/test_integration.py::test_large_dataset_performance --benchmark-only
```

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability scanning

## 🚀 Deployment

### Development
```bash
# Local development with SQLite
formfinder init
formfinder run
```

### Production
```bash
# With PostgreSQL
export DATABASE_TYPE=postgresql
export DATABASE_HOST=your-postgres-host
export DATABASE_PASSWORD=your-password

formfinder init
formfinder schedule  # Set up automated runs
```

### Docker
```bash
docker-compose up -d  # For base setup
docker-compose -f docker-compose.postgres.yml up -d  # For PostgreSQL integration
```

## 📊 Monitoring & Observability

- **Health Checks**: Automated system status monitoring
- **Performance Metrics**: Database query performance, API response times
- **Error Tracking**: Comprehensive logging and error reporting
- **Workflow Monitoring**: Prefect UI for pipeline visualization

## 🔧 Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black formfinder/
isort formfinder/
flake8 formfinder/
mypy formfinder/
```

### Project Structure
```
FormFinder/
├── .dockerignore
├── .github/workflows/         # CI/CD pipeline
│   └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── ARCHITECTURE.md
├── CHANGELOG.md
├── DATABASE_MIGRATION_SUMMARY.md
├── DB Migration Files/        # Migration scripts
│   ├── migrate_csv_to_db.py
│   └── migrate_csv_to_db_fixed.py
├── Dev-Test Files/            # Development and test files
│   ├── test_core_functionality.py
│   ├── test_database_fetcher.py
│   ├── test_full_pipeline.py
│   └── test_postgresql_connection.py
├── Dockerfile
├── Makefile
├── README.md
├── README_POSTGRESQL_SETUP.md
├── config.yaml
├── config.yaml.example
├── database_data_fetcher.py
├── docker-compose.postgres.yml
├── docker-compose.yml
├── docker/                    # Docker configuration files
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── datasources/
│   ├── postgres/
│   │   └── init.sql
│   └── prometheus/
│       └── prometheus.yml
├── formfinder/                # Main package
│   ├── DataFetcher.py
│   ├── DataProcessor.py
│   ├── PredictorOutputter.py
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── database.py
│   ├── notifier.py
│   └── workflows.py
├── full_pipeline_test_issue.md
├── leagues.json
├── main.py
├── makerun_error.md
├── pyproject.toml
├── requirements.txt
├── run_formfinder.sh
├── sdata_config.json
├── sdata_init_config.json
├── setup.cfg
├── setup.py
├── setup_database.py
├── setup_database_native.py
├── setup_postgres_unix.sh
├── setup_postgres_windows.bat
├── setup_postgresql.md
├── setup_postgresql_native.md
├── test_full_pipeline.py
└── tests/                     # Comprehensive test suite
    ├── __init__.py
    ├── conftest.py
    ├── test_config.py
    ├── test_database.py
    ├── test_integration.py
    └── test_workflows.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Quality Standards
- **Test Coverage**: Minimum 90% coverage required
- **Type Hints**: All functions must have type annotations
- **Documentation**: Docstrings for all public functions and classes
- **Code Style**: Black formatting, isort imports, flake8 linting

## 📈 Roadmap

### Phase 2: Advanced Analytics (Q2 2024)
- [ ] Machine learning model improvements
- [ ] Advanced statistical analysis
- [ ] Custom prediction algorithms
- [ ] Performance optimization

### Phase 3: Web Interface (Q3 2024)
- [ ] Web dashboard for predictions
- [ ] Real-time data visualization
- [ ] User management system
- [ ] API endpoints for external access

### Phase 4: Scaling & Enterprise (Q4 2024)
- [ ] Kubernetes deployment
- [ ] Multi-tenant support
- [ ] Advanced monitoring and alerting
- [ ] Enterprise security features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Football-Data.org](https://www.football-data.org/) for providing the API
- [Prefect](https://www.prefect.io/) for workflow orchestration
- [SQLAlchemy](https://www.sqlalchemy.org/) for database management
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

## 📞 Support

- **Documentation**: [Coming Soon]
- **Issues**: [GitHub Issues](https://github.com/yourusername/FormFinder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/FormFinder/discussions)

---

**FormFinder 2.0** - From simple scripts to production-ready data pipelines 🚀