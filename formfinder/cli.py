"""Command-line interface for FormFinder."""

import click
import sys
from pathlib import Path
from typing import Optional

from .config import FormFinderConfig
from .database import DatabaseManager, get_db_session
from .workflows import (
    run_main_pipeline,
    run_quick_update,
    run_health_check,
    schedule_daily_pipeline,
    schedule_health_checks,
)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="config.yaml",
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx: click.Context, config: Path) -> None:
    """FormFinder: Football prediction system with workflow orchestration."""
    ctx.ensure_object(dict)

    click.echo(f"Invoked subcommand: {ctx.invoked_subcommand}")  # DEBUG

    # Skip config loading for commands that don't need it
    if ctx.invoked_subcommand == "config_template":
        click.echo("Skipping config loading for config_template")  # DEBUG
        return

    try:
        click.echo(f"Loading config from: {config}")  # DEBUG
        config_obj = FormFinderConfig.from_yaml(config)
        ctx.obj["config"] = config_obj
        ctx.obj["db_manager"] = DatabaseManager(config_obj.get_database_url())
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize the FormFinder database and directories."""
    config: FormFinderConfig = ctx.obj["config"]
    db_manager: DatabaseManager = ctx.obj["db_manager"]
    
    click.echo("Initializing FormFinder...")
    
    # Create directories
    config.ensure_directories()
    click.echo("✓ Created directories")
    
    # Initialize database
    db_manager.create_tables()
    click.echo("✓ Initialized database")
    
    click.echo("FormFinder initialization complete!")


@cli.command()
@click.option(
    "--leagues",
    "-l",
    multiple=True,
    help="Specific league IDs to process (can be used multiple times)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force processing even if recent data exists",
)
@click.pass_context
def run(ctx: click.Context, leagues: tuple[str, ...], force: bool) -> None:
    """Run the main FormFinder pipeline."""
    config: FormFinderConfig = ctx.obj["config"]
    
    click.echo("Starting FormFinder pipeline...")
    
    # Convert leagues to list of integers if provided
    league_ids = [int(league) for league in leagues] if leagues else None
    
    try:
        result = run_main_pipeline(
            config_path="config.yaml",
            league_ids=league_ids,
            force_refresh=force,
        )
        
        if result.get('overall_status') == 'success':
            click.echo("✓ Pipeline completed successfully")
            click.echo(f"  Leagues processed: {result.get('leagues_processed', 0)}")
            click.echo(f"  Predictions generated: {result.get('predictions_generated', 0)}")
        else:
            click.echo("✗ Pipeline completed with errors", err=True)
            click.echo(f"  Status: {result.get('overall_status', 'unknown')}", err=True)
            click.echo(f"  Successful fetches: {result.get('successful_fetches', 0)}", err=True)
            click.echo(f"  Successful processing: {result.get('successful_processing', 0)}", err=True)
    except Exception as e:
        click.echo(f"Pipeline failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--leagues",
    "-l",
    multiple=True,
    help="Specific league IDs to update (can be used multiple times)",
)
@click.pass_context
def quick_update(ctx: click.Context, leagues: tuple[str, ...]) -> None:
    """Run a quick update for recent fixtures and standings."""
    config: FormFinderConfig = ctx.obj["config"]
    
    click.echo("Starting quick update...")
    
    # Convert leagues to list of integers if provided
    league_ids = [int(league) for league in leagues] if leagues else None
    
    try:
        result = run_quick_update(
            config_path="config.yaml",
            league_ids=league_ids,
        )
        
        if result.get('overall_status') == 'success':
            click.echo("✓ Quick update completed successfully")
            click.echo(f"  Leagues processed: {result.get('leagues_processed', 0)}")
        else:
            click.echo("✗ Quick update completed with errors", err=True)
            click.echo(f"  Status: {result.get('overall_status', 'unknown')}", err=True)
    except Exception as e:
        click.echo(f"Quick update failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def health_check(ctx: click.Context) -> None:
    """Run system health checks."""
    config: FormFinderConfig = ctx.obj["config"]
    
    click.echo("Running health checks...")
    
    try:
        result = run_health_check(config_path="config.yaml")
        
        if result.get('overall') == 'healthy':
            click.echo("✓ All health checks passed")
            for component, status in result.items():
                if component != 'overall':
                    click.echo(f"  {component}: {status}")
        else:
            click.echo("✗ Some health checks failed", err=True)
            for component, status in result.items():
                if component != 'overall':
                    click.echo(f"  {component}: {status}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Health check failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--cron",
    default="0 6 * * *",
    help="Cron expression for pipeline schedule (default: daily at 6 AM)",
)
@click.pass_context
def schedule(ctx: click.Context, cron: str) -> None:
    """Schedule the FormFinder pipeline to run automatically."""
    config: FormFinderConfig = ctx.obj["config"]
    
    click.echo(f"Scheduling pipeline with cron: {cron}")
    
    try:
        # Schedule main pipeline
        deployment_id = schedule_daily_pipeline(
            config_path="config.yaml",
            cron_schedule=cron,
        )
        click.echo(f"✓ Pipeline scheduled (deployment: {deployment_id})")
        
        # Schedule health checks
        health_deployment_id = schedule_health_checks(config_path="config.yaml")
        click.echo(f"✓ Health checks scheduled (deployment: {health_deployment_id})")
        
        click.echo("\nScheduling complete! Use 'prefect deployment ls' to view active deployments.")
    except Exception as e:
        click.echo(f"Scheduling failed: {e}", err=True)
        sys.exit(1)


@cli.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command()
@click.pass_context
def create(ctx: click.Context) -> None:
    """Create database tables."""
    db_manager: DatabaseManager = ctx.obj["db_manager"]
    
    try:
        db_manager.create_tables()
        click.echo("✓ Database tables created")
    except Exception as e:
        click.echo(f"Failed to create tables: {e}", err=True)
        sys.exit(1)


@db.command()
@click.confirmation_option(
    prompt="Are you sure you want to drop all tables? This will delete all data."
)
@click.pass_context
def drop(ctx: click.Context) -> None:
    """Drop all database tables."""
    db_manager: DatabaseManager = ctx.obj["db_manager"]
    
    try:
        db_manager.drop_tables()
        click.echo("✓ Database tables dropped")
    except Exception as e:
        click.echo(f"Failed to drop tables: {e}", err=True)
        sys.exit(1)


@db.command()
@click.pass_context
def refresh_view(ctx: click.Context) -> None:
    """Refresh the materialized view."""
    db_manager: DatabaseManager = ctx.obj["db_manager"]
    
    try:
        db_manager.refresh_materialized_view()
        click.echo("✓ Materialized view refreshed")
    except Exception as e:
        click.echo(f"Failed to refresh view: {e}", err=True)
        sys.exit(1)

@db.command()
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Reset database (drop and recreate tables)."""
    db_manager: DatabaseManager = ctx.obj["db_manager"]
    
    try:
        db_manager.drop_tables()
        click.echo("✓ Dropped existing tables")
        
        db_manager.create_tables()
        click.echo("✓ Created new tables")
        
        click.echo("Database reset complete!")
    except Exception as e:
        click.echo(f"Failed to reset database: {e}", err=True)
        sys.exit(1)


@db.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show database status and statistics."""
    db_manager: DatabaseManager = ctx.obj["db_manager"]
    
    try:
        with get_db_session() as session:
            from .database import League, Team, Fixture, Standing, Prediction, DataFetchLog
            
            # Count records in each table
            leagues_count = session.query(League).count()
            teams_count = session.query(Team).count()
            fixtures_count = session.query(Fixture).count()
            standings_count = session.query(Standing).count()
            predictions_count = session.query(Prediction).count()
            logs_count = session.query(DataFetchLog).count()
        
        click.echo("Database Status:")
        click.echo(f"  Leagues: {leagues_count}")
        click.echo(f"  Teams: {teams_count}")
        click.echo(f"  Fixtures: {fixtures_count}")
        click.echo(f"  Standings: {standings_count}")
        click.echo(f"  Predictions: {predictions_count}")
        click.echo(f"  Fetch Logs: {logs_count}")
        
        # Show recent activity
        recent_log = session.query(DataFetchLog).order_by(
            DataFetchLog.fetch_date.desc()
        ).first()
        
        if recent_log:
            click.echo(f"\nLast fetch: {recent_log.fetch_date} ({recent_log.data_type})")
        else:
            click.echo("\nNo fetch activity recorded")
        
        session.close()
    except Exception as e:
        if 'session' in locals():
            session.close()
        click.echo(f"Failed to get database status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for configuration template",
)
def config_template(output: Optional[Path]) -> None:
    """Generate a configuration template file."""
    template = '''
# FormFinder Configuration Template
# Copy this file to config.yaml and customize for your environment

api:
  token: "YOUR_API_TOKEN_HERE"  # Get from your sports data provider
  base_url: "https://api.football-data.org/v4"
  rate_limit_requests: 10
  rate_limit_period: 60
  timeout: 30
  retry_attempts: 3
  retry_delay: 1
  league_ids:
    - 2021  # Premier League
    - 2014  # La Liga
    - 2002  # Bundesliga
    - 2019  # Serie A
    - 2015  # Ligue 1

processing:
  recent_period: 30
  win_rate_threshold: 0.6
  min_games_for_prediction: 5
  prediction_confidence_threshold: 0.7
  batch_size: 100
  parallel_workers: 4

database:
  type: "sqlite"  # or "postgresql"
  sqlite:
    path: "data/formfinder.db"
  postgresql:
    host: "localhost"
    port: 5432
    database: "formfinder"
    username: "formfinder"
    password: "your_password"
  pool_size: 5
  max_overflow: 10
  echo: false

notifications:
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your_email@gmail.com"
    password: "your_app_password"
    from_address: "your_email@gmail.com"
    to_addresses:
      - "recipient@example.com"
  sms:
    enabled: false
    twilio_account_sid: "your_account_sid"
    twilio_auth_token: "your_auth_token"
    from_number: "+1234567890"
    to_numbers:
      - "+0987654321"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_enabled: true
  console_enabled: true
  max_file_size: 10485760  # 10MB
  backup_count: 5

directories:
  data: "data"
  logs: "data/logs"
  cache: "data/cache"
  predictions: "data/predictions"
  fixtures: "data/fixtures"
  standings: "data/standings"
  historical: "data/historical"

workflow:
  prefect_api_url: "http://localhost:4200/api"
  work_pool: "default-agent-pool"
  deployment_name: "formfinder-pipeline"
  max_retries: 3
  retry_delay: 300
  task_timeout: 3600

testing:
  test_data_dir: "tests/data"
  mock_api_responses: true
  test_database_url: "sqlite:///:memory:"
  parallel_test_workers: 4
'''
    
    if output:
        output.write_text(template.strip(), encoding="utf-8")
        click.echo(f"Configuration template written to {output}")
    else:
        click.echo(template.strip())


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()