#!/usr/bin/env python3
"""
Truncate team_performance_states table.
"""

import sys
from pathlib import Path
from sqlalchemy import text

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session
from formfinder.config import load_config

def main():
    """Truncate team_performance_states table."""
    try:
        # Load configuration first
        load_config()
        
        with get_db_session() as session:
            session.execute(text('TRUNCATE TABLE team_performance_states'))
            session.commit()
            print('team_performance_states table truncated successfully')
    except Exception as e:
        print(f'Error truncating table: {e}')
        raise

if __name__ == '__main__':
    main()