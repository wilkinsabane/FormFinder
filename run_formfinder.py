#!/usr/bin/env python3
"""FormFinder2 CLI Runner

Convenience script to run FormFinder2 application with various modes.

Usage:
    python run_formfinder.py --daemon
    python run_formfinder.py --features
    python run_formfinder.py --train
    python run_formfinder.py --health
    python run_formfinder.py --pipeline
    python run_formfinder.py --status

Author: FormFinder2 Team
Created: 2025-01-01
"""

import sys
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.main import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Application failed: {str(e)}")
        sys.exit(1)