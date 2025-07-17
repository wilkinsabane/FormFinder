# FormFinder Development Makefile
# Provides convenient commands for common development tasks

.PHONY: help install install-dev clean test test-unit test-integration test-performance
.PHONY: lint format type-check security-check coverage docs
.PHONY: build package upload-test upload-prod
.PHONY: docker-build docker-run docker-test
.PHONY: db-init db-reset db-status db-migrate
.PHONY: run quick-update health-check schedule
.PHONY: pre-commit setup-dev

# Default target
help: ## Show this help message
	@echo "FormFinder Development Commands"
	@echo "=============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment Variables:"
	@echo "  API_TOKEN          - Your football data API token"
	@echo "  DATABASE_TYPE      - Database type (sqlite/postgresql)"
	@echo "  DATABASE_HOST      - Database host (for PostgreSQL)"
	@echo "  DATABASE_PASSWORD  - Database password (for PostgreSQL)"

# Installation and Setup
install: ## Install the package in production mode
	pip install .

install-dev: ## Install the package in development mode with all dependencies
	pip install -e .[dev,postgresql,sms,docs,monitoring]

setup-dev: install-dev ## Complete development environment setup
	pre-commit install
	@echo "Development environment setup complete!"
	@echo "Next steps:"
	@echo "  1. Copy config.yaml.example to config.yaml and configure"
	@echo "  2. Run 'make db-init' to initialize the database"
	@echo "  3. Run 'make test' to verify everything works"

# Cleaning
clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "*.cover" -delete
	find . -type f -name "*.log" -delete

clean-data: ## Clean data directories (use with caution!)
	@echo "WARNING: This will delete all data files!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/; \
		rm -rf processed_data/; \
		echo "Data directories cleaned."; \
	else \
		echo "Cancelled."; \
	fi

# Testing
test: install-dev ## Run all tests
	python -m pytest

test-unit: ## Run unit tests only
	pytest tests/ -m "unit or not integration"

test-integration: ## Run integration tests only
	pytest tests/ -m "integration"

test-performance: ## Run performance tests only
	pytest tests/ -m "performance" --benchmark-only

test-watch: ## Run tests in watch mode
	pytest-watch

test-parallel: ## Run tests in parallel
	pytest -n auto

coverage: ## Run tests with coverage report
	pytest --cov=formfinder --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

coverage-xml: ## Generate XML coverage report for CI
	pytest --cov=formfinder --cov-report=xml

# Code Quality
lint: ## Run linting checks
	flake8 formfinder/ tests/

format: ## Format code with black and isort
	black formfinder/ tests/
	isort formfinder/ tests/

format-check: ## Check if code is properly formatted
	black --check formfinder/ tests/
	isort --check-only formfinder/ tests/

type-check: ## Run type checking with mypy
	mypy formfinder/

security-check: ## Run security checks
	bandit -r formfinder/
	safety check

quality: format lint type-check security-check ## Run all code quality checks

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Documentation
docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs && make livehtml

docs-clean: ## Clean documentation build
	cd docs && make clean

# Database Management
db-init: ## Initialize database and directories
	formfinder init

db-reset: ## Reset database (drops and recreates tables)
	formfinder db reset

db-status: ## Show database status
	formfinder db status

db-create: ## Create database tables
	formfinder db create

db-drop: ## Drop database tables
	formfinder db drop

# Application Commands
run: ## Run the main pipeline
	python -m formfinder.cli run

run-force: ## Run pipeline with force refresh
	formfinder run --force

quick-update: ## Run quick update for recent data
	formfinder quick-update

health-check: ## Run system health checks
	formfinder health-check

schedule: ## Schedule automatic pipeline runs
	formfinder schedule

config-template: ## Generate configuration template
	formfinder config-template -o config.yaml.example

# Building and Packaging
build: clean ## Build the package
	python -m build

package: build ## Create distribution packages
	twine check dist/*

upload-test: package ## Upload to Test PyPI
	twine upload --repository testpypi dist/*

upload-prod: package ## Upload to Production PyPI
	twine upload dist/*

# Docker Commands (when Docker support is added)
docker-build: ## Build Docker image
	docker build -t formfinder:latest .

docker-run: ## Run FormFinder in Docker
	docker run --rm -v $(PWD)/config.yaml:/app/config.yaml formfinder:latest

docker-test: ## Run tests in Docker
	docker run --rm formfinder:latest pytest

docker-dev: ## Run development environment in Docker
	docker-compose up -d

docker-down: ## Stop Docker development environment
	docker-compose down

# Development Utilities
shell: ## Start Python shell with FormFinder imported
	python -c "import formfinder; print('FormFinder imported. Available modules:'); print(dir(formfinder))"

notebook: ## Start Jupyter notebook for development
	jupyter notebook

profile: ## Profile the application performance
	python -m cProfile -o profile.stats -m formfinder.cli run
	@echo "Profile saved to profile.stats"
	@echo "View with: python -c 'import pstats; pstats.Stats(\"profile.stats\").sort_stats(\"cumulative\").print_stats(20)'"

memory-profile: ## Profile memory usage
	mprof run formfinder run
	mprof plot

# CI/CD Simulation
ci-test: ## Simulate CI pipeline locally
	make format-check
	make lint
	make type-check
	make security-check
	make test
	make coverage

# Environment Management
env-create: ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"

env-requirements: ## Generate requirements.txt from current environment
	pip freeze > requirements-frozen.txt
	@echo "Current environment requirements saved to requirements-frozen.txt"

# Monitoring and Logs
logs: ## View recent logs
	tail -f data/logs/formfinder.log

logs-error: ## View recent error logs
	grep -i error data/logs/formfinder.log | tail -20

monitor: ## Monitor system resources during execution
	psutil formfinder run

# Backup and Restore
backup: ## Backup data and configuration
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	cp -r data/ backups/$(shell date +%Y%m%d_%H%M%S)/
	cp config.yaml backups/$(shell date +%Y%m%d_%H%M%S)/
	@echo "Backup created in backups/$(shell date +%Y%m%d_%H%M%S)/"

restore: ## Restore from backup (specify BACKUP_DIR)
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo "Usage: make restore BACKUP_DIR=backups/20240101_120000"; \
		exit 1; \
	fi
	cp -r $(BACKUP_DIR)/data/ .
	cp $(BACKUP_DIR)/config.yaml .
	@echo "Restored from $(BACKUP_DIR)"

# Performance Benchmarks
benchmark: ## Run performance benchmarks
	pytest tests/test_integration.py::test_large_dataset_performance --benchmark-only --benchmark-sort=mean

benchmark-compare: ## Compare benchmarks with previous run
	pytest tests/test_integration.py::test_large_dataset_performance --benchmark-only --benchmark-compare

# Development Workflow
dev-setup: setup-dev config-template db-init ## Complete development setup
	@echo "Development setup complete!"
	@echo "Edit config.yaml with your settings and run 'make run' to start."

dev-test: format lint type-check test ## Run development test suite

dev-commit: dev-test ## Prepare for commit (run all checks)
	@echo "All checks passed! Ready to commit."

# Release Management
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

release-check: ## Check if ready for release
	make ci-test
	make docs
	@echo "Release checks passed!"

# Quick Start
quickstart: ## Quick start for new users
	@echo "FormFinder Quick Start"
	@echo "====================="
	@echo "1. Setting up development environment..."
	make setup-dev
	@echo "2. Generating configuration template..."
	make config-template
	@echo "3. Initializing database..."
	make db-init
	@echo "4. Running tests..."
	make test
	@echo ""
	@echo "✅ Quick start complete!"
	@echo "Next steps:"
	@echo "  1. Edit config.yaml with your API token"
	@echo "  2. Run 'make run' to execute the pipeline"
	@echo "  3. Run 'make help' to see all available commands"