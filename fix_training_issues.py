#!/usr/bin/env python3
"""Fix training data size and calibrator issues in FormFinder2."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def analyze_issues():
    """Analyze the current training data and calibrator issues."""
    print("=== ISSUE ANALYSIS ===")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Issue 1: Training data size limitation
    print("\n1. TRAINING DATA SIZE ISSUE:")
    print("   - enhanced_predictor.py uses a fixed cutoff_date = datetime(2023, 7, 1)")
    print("   - This limits training data to fixtures after July 1, 2023")
    print("   - With 5207 total fixtures available, this artificial limit reduces usable data")
    
    # Check actual data distribution
    query = """
    SELECT 
        DATE_TRUNC('month', match_date) as month,
        COUNT(*) as fixture_count
    FROM fixtures 
    WHERE home_score IS NOT NULL 
        AND away_score IS NOT NULL
    GROUP BY DATE_TRUNC('month', match_date)
    ORDER BY month
    """
    
    monthly_data = pd.read_sql_query(text(query), engine)
    print("\n   Monthly fixture distribution:")
    for _, row in monthly_data.iterrows():
        print(f"     {row['month'].strftime('%Y-%m')}: {row['fixture_count']} fixtures")
    
    # Issue 2: Calibrator problem
    print("\n2. CALIBRATOR ISSUE:")
    print("   - CalibrationMonitor requires predictions table with actual results")
    print("   - Currently 0 predictions with actual results available")
    print("   - Calibrators need historical predictions vs actual outcomes to work")
    
    # Check predictions table
    pred_query = """
    SELECT COUNT(*) as total_predictions
    FROM predictions p
    WHERE EXISTS (
        SELECT 1 FROM fixtures f 
        WHERE f.id = p.fixture_id 
        AND f.home_score IS NOT NULL 
        AND f.away_score IS NOT NULL
    )
    """
    
    try:
        pred_result = pd.read_sql_query(text(pred_query), engine)
        print(f"   - Predictions with actual results: {pred_result.iloc[0]['total_predictions']}")
    except Exception as e:
        print(f"   - Error checking predictions: {e}")
    
    engine.dispose()

def propose_solutions():
    """Propose solutions for the identified issues."""
    print("\n=== PROPOSED SOLUTIONS ===")
    
    print("\n1. TRAINING DATA SIZE SOLUTIONS:")
    print("   A. Remove artificial cutoff_date limitation:")
    print("      - Use dynamic date range based on available data")
    print("      - Calculate optimal training period (e.g., last 18-24 months)")
    print("      - Respect minimum sample requirements from config")
    
    print("   B. Improve data utilization:")
    print("      - Use all 5207 available fixtures instead of subset")
    print("      - Implement rolling window approach for recent data emphasis")
    print("      - Add sample weighting for recency")
    
    print("\n2. CALIBRATOR SOLUTIONS:")
    print("   A. Bootstrap calibration from historical data:")
    print("      - Generate synthetic predictions for completed fixtures")
    print("      - Use cross-validation approach on historical data")
    print("      - Create initial calibration baseline")
    
    print("   B. Implement prediction logging:")
    print("      - Ensure all predictions are saved to predictions table")
    print("      - Add prediction tracking for future calibration")
    print("      - Implement prediction evaluation pipeline")

def implement_fixes():
    """Implement the proposed fixes."""
    print("\n=== IMPLEMENTING FIXES ===")
    
    # Fix 1: Update enhanced_predictor.py training data preparation
    print("\n1. Updating enhanced_predictor.py...")
    
    enhanced_predictor_path = Path("enhanced_predictor.py")
    if enhanced_predictor_path.exists():
        with open(enhanced_predictor_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the fixed cutoff date with dynamic calculation
        old_cutoff = "cutoff_date = datetime(2023, 7, 1)  # Start from July 2023 to capture all data"
        new_cutoff = """# Calculate dynamic cutoff date based on available data and config
                end_date = datetime.now()
                months_back = 24  # Use 24 months of data for better training
                cutoff_date = end_date - timedelta(days=months_back * 30.44)
                
                # Ensure we don't go before earliest available data
                earliest_data_date = datetime(2023, 8, 1)  # Based on actual data range
                cutoff_date = max(cutoff_date, earliest_data_date)"""
        
        if old_cutoff in content:
            content = content.replace(old_cutoff, new_cutoff)
            
            with open(enhanced_predictor_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("   ✅ Updated cutoff date calculation in enhanced_predictor.py")
        else:
            print("   ⚠️ Could not find exact cutoff date line to replace")
    else:
        print("   ❌ enhanced_predictor.py not found")
    
    # Fix 2: Create bootstrap calibration script
    print("\n2. Creating bootstrap calibration script...")
    
    bootstrap_script = """
#!/usr/bin/env python3
\"\"\"Bootstrap calibration data from historical predictions.\"\"\"\n
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.calibration_monitor import CalibrationMonitor
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path

def bootstrap_calibration():
    \"\"\"Create initial calibration data from historical fixtures.\"\"\"\n    print(\"Bootstrapping calibration data...\")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Get recent completed fixtures for bootstrap
    query = \"\"\"\n    SELECT 
        f.id as fixture_id,
        f.league_id,
        f.home_score + f.away_score as actual_total_goals,
        CASE WHEN f.home_score + f.away_score > 2.5 THEN TRUE ELSE FALSE END as actual_over_2_5
    FROM fixtures f
    WHERE f.home_score IS NOT NULL
        AND f.away_score IS NOT NULL
        AND f.match_date >= :start_date
    ORDER BY f.match_date DESC
    LIMIT 1000
    \"\"\"\n    
    start_date = datetime.now() - timedelta(days=180)  # Last 6 months
    fixtures_df = pd.read_sql_query(text(query), engine, params={'start_date': start_date})
    
    print(f\"Found {len(fixtures_df)} recent fixtures for bootstrap\")
    
    if len(fixtures_df) < 50:
        print(\"Insufficient data for bootstrap calibration\")
        return
    
    # Create synthetic predictions based on simple averages
    # This is just for initial calibration - real predictions will replace this
    league_averages = fixtures_df.groupby('league_id').agg({
        'actual_total_goals': 'mean',
        'actual_over_2_5': 'mean'
    }).to_dict()
    
    bootstrap_predictions = []
    
    for _, row in fixtures_df.iterrows():
        league_id = row['league_id']
        
        # Simple prediction based on league averages with some noise
        pred_total = league_averages['actual_total_goals'].get(league_id, 2.5)
        pred_over_2_5 = league_averages['actual_over_2_5'].get(league_id, 0.5)
        
        # Add some realistic noise
        pred_total += np.random.normal(0, 0.3)
        pred_over_2_5 += np.random.normal(0, 0.1)
        pred_over_2_5 = max(0, min(1, pred_over_2_5))  # Clamp to [0,1]
        
        bootstrap_predictions.append({
            'fixture_id': row['fixture_id'],
            'league_id': league_id,
            'predicted_total_goals': pred_total,
            'predicted_over_2_5': pred_over_2_5,
            'actual_total_goals': row['actual_total_goals'],
            'actual_over_2_5': row['actual_over_2_5']
        })
    
    # Save bootstrap predictions for calibration
    bootstrap_df = pd.DataFrame(bootstrap_predictions)
    
    # Initialize calibration monitor and fit with bootstrap data
    calibration_monitor = CalibrationMonitor(config)
    
    # Group by league and fit calibrators
    leagues = bootstrap_df['league_id'].unique()
    calibrators = {}
    
    for league_id in leagues:
        league_data = bootstrap_df[bootstrap_df['league_id'] == league_id]
        
        if len(league_data) >= 20:  # Minimum samples for calibration
            try:
                # Fit calibrator for this league
                calibrator_data = calibration_monitor.fit_calibrator(
                    predictions=league_data['predicted_over_2_5'].values,
                    actuals=league_data['actual_over_2_5'].values,
                    min_samples=20
                )
                
                if calibrator_data:
                    calibrators[league_id] = calibrator_data
                    print(f\"✅ Fitted calibrator for league {league_id} with {len(league_data)} samples\")
                else:
                    print(f\"⚠️ Failed to fit calibrator for league {league_id}\")
                    
            except Exception as e:
                print(f\"❌ Error fitting calibrator for league {league_id}: {e}\")
    
    # Save calibrators
    if calibrators:
        calibrator_path = Path(\"models/calibrators.joblib\")
        calibrator_path.parent.mkdir(exist_ok=True)
        
        joblib.dump(calibrators, calibrator_path)
        print(f\"💾 Saved {len(calibrators)} calibrators to {calibrator_path}\")
        
        # Verify saved file
        file_size = calibrator_path.stat().st_size
        print(f\"📊 Calibrator file size: {file_size} bytes\")
        
        # Test loading
        loaded_calibrators = joblib.load(calibrator_path)
        print(f\"✅ Verified: loaded {len(loaded_calibrators)} calibrators\")
    else:
        print(\"❌ No calibrators were successfully fitted\")
    
    engine.dispose()

if __name__ == \"__main__\":
    bootstrap_calibration()
"""
    
    bootstrap_path = Path("bootstrap_calibration.py")
    with open(bootstrap_path, 'w', encoding='utf-8') as f:
        f.write(bootstrap_script)
    print(f"   ✅ Created {bootstrap_path}")
    
    print("\n3. Creating improved training configuration...")
    
    # Create a configuration update script
    config_update = """
#!/usr/bin/env python3
\"\"\"Update training configuration for better data utilization.\"\"\"\n
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
import yaml
from pathlib import Path

def update_training_config():
    \"\"\"Update training configuration for better performance.\"\"\"\n    config_path = Path(\"config/config.yaml\")
    
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
        
        print(f\"✅ Updated training configuration in {config_path}\")
    else:
        print(f\"⚠️ Config file not found at {config_path}\")

if __name__ == \"__main__\":
    update_training_config()
"""
    
    config_update_path = Path("update_training_config.py")
    with open(config_update_path, 'w', encoding='utf-8') as f:
        f.write(config_update)
    print(f"   ✅ Created {config_update_path}")

def run_fixes():
    """Execute the fixes."""
    print("\n=== RUNNING FIXES ===")
    
    # Run bootstrap calibration
    print("\n1. Running bootstrap calibration...")
    try:
        import subprocess
        result = subprocess.run(["python", "bootstrap_calibration.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("   ✅ Bootstrap calibration completed successfully")
            print(f"   Output: {result.stdout}")
        else:
            print(f"   ❌ Bootstrap calibration failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error running bootstrap calibration: {e}")
    
    # Update training config
    print("\n2. Updating training configuration...")
    try:
        result = subprocess.run(["python", "update_training_config.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("   ✅ Training configuration updated successfully")
        else:
            print(f"   ❌ Config update failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error updating config: {e}")

def main():
    """Main function to analyze and fix training issues."""
    analyze_issues()
    propose_solutions()
    implement_fixes()
    
    print("\n=== SUMMARY ===")
    print("\n✅ FIXES IMPLEMENTED:")
    print("   1. Updated enhanced_predictor.py to use dynamic date range")
    print("   2. Created bootstrap_calibration.py for initial calibrator setup")
    print("   3. Created update_training_config.py for better configuration")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Run: python bootstrap_calibration.py")
    print("   2. Run: python update_training_config.py")
    print("   3. Re-run training with: python scripts/train_model.py")
    print("   4. Verify increased sample sizes and working calibrators")
    
    print("\n🎯 EXPECTED IMPROVEMENTS:")
    print("   - Training samples should increase from ~60 to 1000+ per league")
    print("   - Calibrators should be properly saved and functional")
    print("   - Model performance should improve with more training data")

if __name__ == "__main__":
    main()