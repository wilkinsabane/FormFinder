
#!/usr/bin/env python3
"""Bootstrap calibration data from historical predictions."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from formfinder.config import load_config, get_config
from calibration_monitor import CalibrationMonitor
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

def bootstrap_calibration():
    """Create initial calibration data from historical fixtures."""
    print("Bootstrapping calibration data...")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Get recent completed fixtures for bootstrap
    query = """
    SELECT 
        f.id as fixture_id,
        f.league_id,
        f.home_score + f.away_score as actual_total_goals,
        CASE WHEN f.home_score + f.away_score > 2.5 THEN TRUE ELSE FALSE END as actual_over_2_5
    FROM fixtures f
    WHERE f.home_score IS NOT NULL
        AND f.away_score IS NOT NULL
        AND f.match_date >= :start_date
    ORDER BY f.match_date DESC
    LIMIT 2000
    """
    
    start_date = datetime.now() - timedelta(days=365)  # Last year
    fixtures_df = pd.read_sql_query(text(query), engine, params={'start_date': start_date})
    
    print(f"Found {len(fixtures_df)} recent fixtures for bootstrap")
    
    if len(fixtures_df) < 100:
        print("Insufficient data for bootstrap calibration")
        return
    
    # Create synthetic predictions based on league averages
    league_stats = fixtures_df.groupby('league_id').agg({
        'actual_total_goals': ['mean', 'std'],
        'actual_over_2_5': 'mean'
    }).round(4)
    
    # Flatten column names
    league_stats.columns = ['_'.join(col).strip() for col in league_stats.columns]
    league_stats = league_stats.reset_index()
    
    print(f"\nLeague statistics for {len(league_stats)} leagues:")
    for _, row in league_stats.head(10).iterrows():
        print(f"  League {row['league_id']}: avg_goals={row['actual_total_goals_mean']:.2f}, over_2.5_rate={row['actual_over_2_5_mean']:.2f}")
    
    # Create bootstrap predictions with realistic noise
    bootstrap_predictions = []
    
    for _, fixture in fixtures_df.iterrows():
        league_id = fixture['league_id']
        
        # Get league stats
        league_row = league_stats[league_stats['league_id'] == league_id]
        if len(league_row) == 0:
            continue
            
        league_row = league_row.iloc[0]
        
        # Create realistic predictions with noise
        base_over_2_5_prob = league_row['actual_over_2_5_mean']
        
        # Add realistic noise based on uncertainty
        noise_factor = 0.15  # 15% noise
        predicted_over_2_5 = base_over_2_5_prob + np.random.normal(0, noise_factor)
        predicted_over_2_5 = max(0.05, min(0.95, predicted_over_2_5))  # Clamp to realistic range
        
        bootstrap_predictions.append({
            'fixture_id': fixture['fixture_id'],
            'league_id': league_id,
            'over_2_5_probability': predicted_over_2_5,
            'actual_over_2_5': fixture['actual_over_2_5']
        })
    
    bootstrap_df = pd.DataFrame(bootstrap_predictions)
    print(f"\nCreated {len(bootstrap_df)} bootstrap predictions")
    
    # Create calibrators manually using the same approach as CalibrationMonitor
    calibrators = {}
    leagues = bootstrap_df['league_id'].unique()
    
    print(f"\nFitting calibrators for {len(leagues)} leagues...")
    
    for league_id in leagues:
        league_data = bootstrap_df[bootstrap_df['league_id'] == league_id]
        
        if len(league_data) >= 30:  # Minimum samples for calibration
            try:
                # Prepare data
                X = league_data['over_2_5_probability'].values
                y = league_data['actual_over_2_5'].values
                
                # Calculate pre-calibration metrics
                def calculate_ece(y_true, y_prob, n_bins=10):
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    ece = 0
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = y_true[in_bin].mean()
                            avg_confidence_in_bin = y_prob[in_bin].mean()
                            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    return ece
                
                pre_ece = calculate_ece(y, X)
                pre_brier = brier_score_loss(y, X)
                
                # Fit isotonic regression calibrator
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(X, y)
                
                # Calculate post-calibration metrics
                post_probs = calibrator.predict(X)
                post_ece = calculate_ece(y, post_probs)
                post_brier = brier_score_loss(y, post_probs)
                
                # Store calibrator
                calibrators[league_id] = {
                    'calibrator': calibrator,
                    'fitted_date': datetime.now(),
                    'n_samples': len(league_data),
                    'pre_calibration_ece': pre_ece,
                    'post_calibration_ece': post_ece,
                    'pre_calibration_brier': pre_brier,
                    'post_calibration_brier': post_brier,
                    'improvement': pre_ece - post_ece
                }
                
                print(f"  ✅ League {league_id}: {len(league_data)} samples, ECE: {pre_ece:.4f} → {post_ece:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error fitting calibrator for league {league_id}: {e}")
    
    # Save calibrators
    if calibrators:
        calibrator_path = Path("models/calibrators.joblib")
        calibrator_path.parent.mkdir(exist_ok=True)
        
        joblib.dump(calibrators, calibrator_path)
        print(f"\n💾 Saved {len(calibrators)} calibrators to {calibrator_path}")
        
        # Verify saved file
        file_size = calibrator_path.stat().st_size
        print(f"📊 Calibrator file size: {file_size:,} bytes")
        
        # Test loading
        try:
            loaded_calibrators = joblib.load(calibrator_path)
            print(f"✅ Verified: loaded {len(loaded_calibrators)} calibrators")
            
            # Show sample calibrator info
            for league_id, cal_data in list(loaded_calibrators.items())[:3]:
                print(f"  League {league_id}: {cal_data['n_samples']} samples, improvement: {cal_data['improvement']:.4f}")
                
        except Exception as e:
            print(f"❌ Error verifying calibrators: {e}")
    else:
        print("❌ No calibrators were successfully fitted")
    
    engine.dispose()
    print("\n🎉 Bootstrap calibration completed!")

if __name__ == "__main__":
    bootstrap_calibration()
