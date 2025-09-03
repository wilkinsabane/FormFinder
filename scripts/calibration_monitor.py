"""League-specific probability calibration with monitoring for goal predictions."""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add parent directory to path to import formfinder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formfinder.config import load_config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationMonitor:
    """Monitor and calibrate predictions by league with drift detection."""
    
    def __init__(self, config=None):
        if config is None:
            load_config()
            config = get_config()
        self.config = config
        self.engine = create_engine(config.get_database_url())
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        self.calibrators = {}
        self.calibration_history = []
        
    def get_historical_predictions(self, league_id: int, days_back: int = 90) -> pd.DataFrame:
        """Get historical predictions with actual outcomes for calibration."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        query = """
        SELECT 
            p.predicted_total_goals,
            p.over_2_5_probability,
            p.confidence_score,
            f.home_score,
            f.away_score,
            f.match_date,
            f.league_id
        FROM predictions p
        JOIN fixtures f ON p.fixture_id = f.id
        WHERE f.league_id = :league_id 
            AND f.match_date >= :cutoff_date
            AND f.home_score IS NOT NULL
            AND f.away_score IS NOT NULL
        ORDER BY f.match_date DESC
        """
        
        df = pd.read_sql_query(text(query), self.engine, params={
            'league_id': league_id,
            'cutoff_date': cutoff_date.isoformat()
        })
            
        if len(df) == 0:
            return df
            
        # Calculate actual outcomes
        df['actual_total_goals'] = df['home_score'] + df['away_score']
        df['actual_over_2_5'] = (df['actual_total_goals'] > 2.5).astype(int)
        
        return df
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
        """Calculate Expected Calibration Error (ECE) and reliability diagram data."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'count': in_bin.sum(),
                    'proportion': prop_in_bin
                })
        
        return {
            'ece': ece,
            'brier_score': brier_score_loss(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'bin_data': bin_data
        }
    
    def fit_calibrator(self, league_id: int, min_samples: int = 50) -> bool:
        """Fit isotonic regression calibrator for a specific league."""
        df = self.get_historical_predictions(league_id)
        
        if len(df) < min_samples:
            logger.warning(f"Insufficient data for league {league_id}: {len(df)} samples")
            return False
            
        # Calibrate over 2.5 probabilities
        X = df['over_2_5_probability'].values.reshape(-1, 1)
        y = df['actual_over_2_5'].values
        
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X.ravel(), y)
        
        # Calculate pre and post calibration metrics
        pre_metrics = self.calculate_calibration_metrics(y, X.ravel())
        post_probs = calibrator.predict(X.ravel())
        post_metrics = self.calculate_calibration_metrics(y, post_probs)
        
        self.calibrators[league_id] = {
            'calibrator': calibrator,
            'fitted_date': datetime.now(),
            'n_samples': len(df),
            'pre_calibration_ece': pre_metrics['ece'],
            'post_calibration_ece': post_metrics['ece'],
            'improvement': pre_metrics['ece'] - post_metrics['ece']
        }
        
        logger.info(f"League {league_id}: ECE improved from {pre_metrics['ece']:.4f} to {post_metrics['ece']:.4f}")
        return True
    
    def calibrate_prediction(self, league_id: int, probability: float) -> float:
        """Apply calibration to a probability for a specific league."""
        if league_id not in self.calibrators:
            logger.warning(f"No calibrator found for league {league_id}")
            return probability
            
        calibrator = self.calibrators[league_id]['calibrator']
        return float(calibrator.predict([probability])[0])
    
    def monitor_drift(self, league_id: int) -> Dict:
        """Monitor calibration drift for a league over time."""
        df = self.get_historical_predictions(league_id, days_back=30)
        
        if len(df) < 20:
            return {'status': 'insufficient_data', 'samples': len(df)}
            
        # Calculate recent calibration metrics
        y_true = df['actual_over_2_5'].values
        y_prob = df['over_2_5_probability'].values
        
        metrics = self.calculate_calibration_metrics(y_true, y_prob)
        
        # Check if recalibration is needed
        needs_recalibration = (
            metrics['ece'] > 0.05 or  # ECE threshold
            metrics['brier_score'] > 0.3 or  # Brier score threshold
            len(df) % 100 == 0  # Periodic recalibration
        )
        
        return {
            'status': 'needs_recalibration' if needs_recalibration else 'good',
            'ece': metrics['ece'],
            'brier_score': metrics['brier_score'],
            'log_loss': metrics['log_loss'],
            'samples': len(df),
            'needs_recalibration': needs_recalibration
        }
    
    def save_calibrators(self, filepath: str = "models/calibrators.joblib"):
        """Save all calibrators to disk."""
        joblib.dump(self.calibrators, filepath)
        logger.info(f"Saved {len(self.calibrators)} calibrators to {filepath}")
    
    def load_calibrators(self, filepath: str = "models/calibrators.joblib"):
        """Load calibrators from disk."""
        try:
            self.calibrators = joblib.load(filepath)
            logger.info(f"Loaded {len(self.calibrators)} calibrators from {filepath}")
        except FileNotFoundError:
            logger.warning(f"No calibrators found at {filepath}")
    
    def run_monthly_calibration(self):
        """Run calibration for all active leagues."""
        # Load existing calibrators first
        self.load_calibrators()
        
        # Get all leagues with recent predictions
        query = """
        SELECT DISTINCT f.league_id, COUNT(*) as prediction_count
        FROM predictions p
        JOIN fixtures f ON p.fixture_id = f.id
        WHERE f.match_date >= CURRENT_DATE - INTERVAL '90 days'
            AND f.home_score IS NOT NULL
        GROUP BY f.league_id
        HAVING COUNT(*) >= 20
        """
        
        leagues_df = pd.read_sql_query(text(query), self.engine)
        
        results = []
        
        # If no recent data available, report existing calibrators
        if len(leagues_df) == 0:
            logger.info(f"No recent predictions with results found. Using existing {len(self.calibrators)} calibrators.")
            for league_id in self.calibrators.keys():
                results.append({
                    'league_id': league_id,
                    'calibrated': True,
                    'drift_status': 'existing',
                    'source': 'bootstrap'
                })
        else:
            # Process leagues with recent data
            for _, row in leagues_df.iterrows():
                league_id = row['league_id']
                success = self.fit_calibrator(league_id)
                
                if success:
                    drift_status = self.monitor_drift(league_id)
                    results.append({
                        'league_id': league_id,
                        'calibrated': True,
                        'drift_status': drift_status['status'],
                        'ece': drift_status.get('ece', None)
                    })
                else:
                    results.append({
                        'league_id': league_id,
                        'calibrated': False,
                        'reason': 'insufficient_data'
                    })
        
        self.save_calibrators()
        return results

def main():
    """Run calibration monitoring."""
    monitor = CalibrationMonitor()
    monitor.load_calibrators()
    
    results = monitor.run_monthly_calibration()
    
    print("\nCalibration Results:")
    for result in results:
        print(f"League {result['league_id']}: {result}")

if __name__ == "__main__":
    main()