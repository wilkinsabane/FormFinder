
#!/usr/bin/env python3
"""Update training configuration for better data utilization."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
import yaml
from pathlib import Path

def update_training_config():
    """Update training configuration for better performance."""
    config_path = Path("config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update dynamic training settings
        if 'dynamic_training' not in config_data:
            config_data['dynamic_training'] = {}
        
        config_data['dynamic_training'].update({
            'default_months_back': 24,  # Increase from 20 to 24 months
            'min_training_samples': 100,  # Increase from 50 to 100
            'target_training_samples': 500,  # Increase from 300 to 500
            'max_months_back': 36  # Increase from 30 to 36 months
        })
        
        # Update training config
        if 'training' not in config_data:
            config_data['training'] = {}
        
        config_data['training'].update({
            'min_training_samples': 100,  # Increase minimum samples
            'enable_sample_weighting': True,  # Enable recency weighting
            'recency_decay_factor': 0.95  # Weight recent data more heavily
        })
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"✅ Updated training configuration in {config_path}")
    else:
        print(f"⚠️ Config file not found at {config_path}")

if __name__ == "__main__":
    update_training_config()
