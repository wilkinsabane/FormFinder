#!/usr/bin/env python3
"""
Interactive Database Table Data Clearer

A modern, robust, and user-friendly script for clearing data from specific
database tables without dropping the table structure. Provides interactive
table selection with detailed summaries and safety confirmations.

Usage:
    python clear_table_data.py
    python clear_table_data.py --table fixtures --confirm
    python clear_table_data.py --dry-run

Features:
- Interactive table selection with data summaries
- Safety confirmations before deletion
- Dry-run mode for testing
- Comprehensive logging
- Support for both SQLite and PostgreSQL
- Backup recommendations
- Detailed progress reporting
"""

import argparse
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json

from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from rich.console import Console
from rich.table import Table as RichTable
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from formfinder.database import get_db_manager, get_db_session
from formfinder.config import get_config, load_config


class TableDataClearer:
    """Main class for handling table data clearing operations."""
    
    def __init__(self):
        """Initialize the table data clearer with database connection."""
        self.console = Console()
        self.db_manager = get_db_manager()
        self.engine = self.db_manager.engine
        self.inspector = inspect(self.engine)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging for the clearer."""
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"table_clearer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if not hasattr(sys, '_called_from_test') else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TableDataClearer initialized with database: {self.engine.url}")
        
    def get_all_tables(self) -> List[str]:
        """Get all table names in the database."""
        return sorted(self.inspector.get_table_names())
    
    def get_table_summary(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive summary information for a table."""
        summary = {
            'table_name': table_name,
            'row_count': 0,
            'columns': [],
            'indexes': [],
            'foreign_keys': [],
            'size_estimate': '0 B',
            'created_at': None,
            'last_update': None
        }
        
        try:
            # Get row count
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                summary['row_count'] = result.scalar()
                
                # Get size estimate (SQLite vs PostgreSQL)
                if str(self.engine.url).startswith('sqlite'):
                    result = conn.execute(text(f"SELECT SUM(pgsize) FROM dbstat WHERE name='{table_name}'"))
                    size_bytes = result.scalar() or 0
                    if size_bytes > 0:
                        summary['size_estimate'] = self._format_bytes(size_bytes)
                else:
                    # PostgreSQL size query
                    result = conn.execute(text(f"SELECT pg_total_relation_size('{table_name}')"))
                    size_bytes = result.scalar() or 0
                    if size_bytes > 0:
                        summary['size_estimate'] = self._format_bytes(size_bytes)
            
            # Get column information
            columns = self.inspector.get_columns(table_name)
            summary['columns'] = [
                {
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'primary_key': col.get('primary_key', False)
                }
                for col in columns
            ]
            
            # Get indexes
            indexes = self.inspector.get_indexes(table_name)
            summary['indexes'] = [
                {
                    'name': idx['name'],
                    'unique': idx['unique'],
                    'columns': idx['column_names']
                }
                for idx in indexes
            ]
            
            # Get foreign keys
            fks = self.inspector.get_foreign_keys(table_name)
            summary['foreign_keys'] = [
                {
                    'name': fk['name'],
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns'],
                    'constrained_columns': fk['constrained_columns']
                }
                for fk in fks
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting summary for table {table_name}: {e}")
            
        return summary
    
    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes into human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} TB"
    
    def display_table_summary(self, table_name: str, summary: Dict[str, Any]):
        """Display detailed table summary using rich formatting."""
        self.console.print(f"\n[bold blue]Table: {table_name}[/bold blue]")
        
        # Create summary table
        table = RichTable(title=f"Summary for {table_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Row Count", f"{summary['row_count']:,}")
        table.add_row("Size Estimate", summary['size_estimate'])
        table.add_row("Columns", str(len(summary['columns'])))
        table.add_row("Indexes", str(len(summary['indexes'])))
        table.add_row("Foreign Keys", str(len(summary['foreign_keys'])))
        
        self.console.print(table)
        
        # Column details
        if summary['columns']:
            self.console.print(f"\n[bold]Columns:[/bold]")
            col_table = RichTable()
            col_table.add_column("Name", style="cyan")
            col_table.add_column("Type", style="yellow")
            col_table.add_column("Nullable", style="green")
            col_table.add_column("Primary Key", style="red")
            
            for col in summary['columns']:
                col_table.add_row(
                    col['name'],
                    col['type'],
                    "✓" if col['nullable'] else "✗",
                    "✓" if col['primary_key'] else "✗"
                )
            
            self.console.print(col_table)
    
    def check_dependencies(self, table_name: str) -> List[Dict[str, str]]:
        """Check for foreign key dependencies on this table."""
        dependencies = []
        
        try:
            all_tables = self.get_all_tables()
            for other_table in all_tables:
                if other_table == table_name:
                    continue
                    
                fks = self.inspector.get_foreign_keys(other_table)
                for fk in fks:
                    if fk['referred_table'] == table_name:
                        dependencies.append({
                            'table': other_table,
                            'fk_name': fk['name'],
                            'columns': ', '.join(fk['constrained_columns']),
                            'referred_columns': ', '.join(fk['referred_columns'])
                        })
        except Exception as e:
            self.logger.error(f"Error checking dependencies for {table_name}: {e}")
            
        return dependencies
    
    def clear_table_data(self, table_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Clear all data from a specific table."""
        result = {
            'success': False,
            'rows_deleted': 0,
            'duration': 0,
            'error': None,
            'backup_created': False
        }
        
        start_time = datetime.now()
        
        try:
            # Get initial row count
            with self.engine.connect() as conn:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                initial_count = count_result.scalar()
                
                if initial_count == 0:
                    result['rows_deleted'] = 0
                    result['success'] = True
                    return result
                
                if dry_run:
                    result['rows_deleted'] = initial_count
                    result['success'] = True
                    return result
                
                # Create backup if configured
                backup_path = self._create_backup(table_name)
                if backup_path:
                    result['backup_created'] = True
                    result['backup_path'] = str(backup_path)
                
                # Disable foreign key checks for SQLite
                if str(self.engine.url).startswith('sqlite'):
                    conn.execute(text("PRAGMA foreign_keys=OFF"))
                
                # Delete all data
                delete_result = conn.execute(text(f"DELETE FROM {table_name}"))
                result['rows_deleted'] = delete_result.rowcount
                
                # Re-enable foreign key checks for SQLite
                if str(self.engine.url).startswith('sqlite'):
                    conn.execute(text("PRAGMA foreign_keys=ON"))
                
                # Reset sequences for PostgreSQL
                if not str(self.engine.url).startswith('sqlite'):
                    conn.execute(text(f"ALTER SEQUENCE {table_name}_id_seq RESTART WITH 1"))
                
                conn.commit()
                
                result['success'] = True
                
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error clearing table {table_name}: {e}")
            
        result['duration'] = (datetime.now() - start_time).total_seconds()
        return result
    
    def _create_backup(self, table_name: str) -> Optional[Path]:
        """Create a backup of table data before clearing."""
        try:
            backup_dir = Path("data/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{table_name}_backup_{timestamp}.json"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name}"))
                rows = [dict(row._mapping) for row in result]
                
                if rows:
                    with open(backup_file, 'w') as f:
                        json.dump(rows, f, indent=2, default=str)
                    
                    self.logger.info(f"Backup created: {backup_file}")
                    return backup_file
                    
        except Exception as e:
            self.logger.error(f"Error creating backup for {table_name}: {e}")
            
        return None
    
    def interactive_table_selection(self) -> Optional[str]:
        """Interactive table selection with detailed information."""
        tables = self.get_all_tables()
        
        if not tables:
            self.console.print("[red]No tables found in the database![/red]")
            return None
        
        self.console.print(Panel.fit(
            "[bold green]Database Table Data Clearer[/bold green]\n\n"
            "Select a table to clear all data from. The table structure will be preserved.",
            border_style="green"
        ))
        
        # Display tables with summaries
        table_list = []
        for i, table_name in enumerate(tables, 1):
            summary = self.get_table_summary(table_name)
            table_list.append({
                'index': i,
                'name': table_name,
                'rows': summary['row_count'],
                'size': summary['size_estimate']
            })
        
        # Create selection table
        selection_table = RichTable(title="Available Tables")
        selection_table.add_column("#", style="cyan", justify="right")
        selection_table.add_column("Table Name", style="green")
        selection_table.add_column("Rows", style="yellow", justify="right")
        selection_table.add_column("Size", style="magenta")
        
        for item in table_list:
            selection_table.add_row(
                str(item['index']),
                item['name'],
                f"{item['rows']:,}",
                item['size']
            )
        
        self.console.print(selection_table)
        
        # Get user selection
        while True:
            choice = Prompt.ask(
                "\n[yellow]Enter table number (or 'q' to quit)[/yellow]",
                choices=[str(i) for i in range(1, len(tables)+1)] + ['q', 'quit'],
                show_choices=False
            )
            
            if choice.lower() in ['q', 'quit']:
                return None
                
            try:
                selected_index = int(choice)
                selected_table = table_list[selected_index - 1]['name']
                
                # Show detailed summary
                summary = self.get_table_summary(selected_table)
                self.display_table_summary(selected_table, summary)
                
                # Check dependencies
                dependencies = self.check_dependencies(selected_table)
                if dependencies:
                    self.console.print(f"\n[bold red]⚠️  WARNING: Dependencies found![/bold red]")
                    dep_table = RichTable()
                    dep_table.add_column("Dependent Table", style="red")
                    dep_table.add_column("Foreign Key", style="yellow")
                    dep_table.add_column("Columns", style="cyan")
                    
                    for dep in dependencies:
                        dep_table.add_row(
                            dep['table'],
                            dep['fk_name'],
                            dep['columns']
                        )
                    
                    self.console.print(dep_table)
                    
                    if not Confirm.ask("\n[yellow]Continue despite dependencies?[/yellow]"):
                        continue
                
                # Final confirmation
                if Confirm.ask(
                    f"\n[red bold]Are you sure you want to clear all {summary['row_count']:,} rows from '{selected_table}'?[/red bold]",
                    default=False
                ):
                    return selected_table
                    
            except (ValueError, IndexError):
                self.console.print("[red]Invalid selection. Please try again.[/red]")
    
    def run_interactive(self):
        """Run the interactive table clearer."""
        try:
            selected_table = self.interactive_table_selection()
            if selected_table:
                self.clear_selected_table(selected_table)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.console.print(f"[red]Error: {e}[/red]")
    
    def clear_selected_table(self, table_name: str, dry_run: bool = False):
        """Clear data from the selected table with progress display."""
        self.console.print(f"\n[bold yellow]Clearing data from '{table_name}'...[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Clearing table data...", total=None)
            
            result = self.clear_table_data(table_name, dry_run)
            
            if result['success']:
                if dry_run:
                    self.console.print(
                        f"[green]✓ Dry run complete: Would delete {result['rows_deleted']:,} rows[/green]"
                    )
                else:
                    self.console.print(
                        f"[green]✓ Successfully cleared {result['rows_deleted']:,} rows from '{table_name}'[/green]"
                    )
                    self.console.print(f"[dim]Duration: {result['duration']:.2f}s[/dim]")
                    
                    if result.get('backup_created'):
                        self.console.print(f"[dim]Backup created: {result['backup_path']}[/dim]")
            else:
                self.console.print(f"[red]✗ Error: {result['error']}[/red]")


def main():
    """Main entry point for the script."""
    # Load configuration first
    try:
        load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Clear data from database tables while preserving structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive mode
  %(prog)s --table fixtures   # Clear specific table
  %(prog)s --dry-run          # Show what would be deleted
  %(prog)s --list             # List all tables with summaries
        """
    )
    
    parser.add_argument(
        '--table', '-t',
        help='Specific table name to clear (skips interactive selection)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all tables with summaries and exit'
    )
    
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompts (use with caution)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    clearer = TableDataClearer()
    
    try:
        if args.list:
            tables = clearer.get_all_tables()
            if tables:
                console = Console()
                console.print("\n[bold green]Database Tables Summary[/bold green]")
                
                for table_name in tables:
                    summary = clearer.get_table_summary(table_name)
                    clearer.display_table_summary(table_name, summary)
            else:
                print("No tables found in the database.")
            return
        
        if args.table:
            # Check if table exists
            if args.table not in clearer.get_all_tables():
                print(f"Error: Table '{args.table}' not found in database.")
                print("Available tables:", ', '.join(clearer.get_all_tables()))
                sys.exit(1)
            
            # Get summary
            summary = clearer.get_table_summary(args.table)
            clearer.display_table_summary(args.table, summary)
            
            # Check dependencies
            dependencies = clearer.check_dependencies(args.table)
            if dependencies:
                print(f"\n⚠️  WARNING: {len(dependencies)} dependencies found!")
                for dep in dependencies:
                    print(f"   - {dep['table']} depends on {args.table}")
            
            # Confirmation
            if not args.confirm:
                confirm = input(f"\nClear {summary['row_count']:,} rows from '{args.table}'? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("Operation cancelled.")
                    return
            
            clearer.clear_selected_table(args.table, args.dry_run)
        else:
            # Interactive mode
            clearer.run_interactive()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()