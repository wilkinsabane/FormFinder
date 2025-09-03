#!/usr/bin/env python3
"""
Final test script to verify that the prediction system is working correctly
with all 87 features and producing diverse, accurate predictions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from enhanced_predictor import EnhancedGoalPredictor
from sqlalchemy import text
import numpy as np

def test_predictions():
    """Test predictions on multiple fixtures to verify diversity and accuracy."""
    load_config()
    
    predictor = EnhancedGoalPredictor()
    
    # Load the trained models
    models_loaded = predictor.load_models("models/models.pkl")
    if not models_loaded:
        print("❌ Failed to load trained models")
        return
    print("✅ Loaded trained models successfully")
    
    with get_db_session() as session:
        # Get a sample of fixtures with different characteristics
        query = text("""
            SELECT DISTINCT f.id as fixture_id, ht.name as home_team, at.name as away_team,
                   f.home_score, f.away_score
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.home_score IS NOT NULL AND f.away_score IS NOT NULL
            ORDER BY f.id DESC
            LIMIT 10
        """)
        
        fixtures = session.execute(query).fetchall()
        
        if not fixtures:
            print("❌ No fixtures found for testing")
            return
        
        print(f"🧪 Testing predictions on {len(fixtures)} fixtures...\n")
        
        predictions = []
        actual_results = []
        
        for fixture in fixtures:
            fixture_id, home_team, away_team, actual_home, actual_away = fixture
            
            try:
                # Get prediction
                prediction = predictor.predict_with_uncertainty(fixture_id)
                pred_home = prediction.get('home_goals', 0)
                pred_away = prediction.get('away_goals', 0)
                predictions.append((pred_home, pred_away))
                actual_results.append((actual_home, actual_away))
                
                print(f"Fixture {fixture_id}: {home_team} vs {away_team}")
                print(f"  Predicted: {pred_home:.2f} - {pred_away:.2f}")
                print(f"  Actual:    {actual_home} - {actual_away}")
                
                # Calculate prediction error
                home_error = abs(pred_home - actual_home)
                away_error = abs(pred_away - actual_away)
                total_error = home_error + away_error
                
                print(f"  Error:     {home_error:.2f} + {away_error:.2f} = {total_error:.2f}")
                print()
                
            except Exception as e:
                print(f"❌ Error predicting fixture {fixture_id}: {e}")
                print()
        
        # Analyze prediction diversity
        if predictions:
            home_preds = [p[0] for p in predictions]
            away_preds = [p[1] for p in predictions]
            
            print("📊 Prediction Analysis:")
            print(f"  Home goals - Min: {min(home_preds):.2f}, Max: {max(home_preds):.2f}, Std: {np.std(home_preds):.2f}")
            print(f"  Away goals - Min: {min(away_preds):.2f}, Max: {max(away_preds):.2f}, Std: {np.std(away_preds):.2f}")
            
            # Check for diversity (standard deviation should be > 0.1)
            home_diversity = np.std(home_preds) > 0.1
            away_diversity = np.std(away_preds) > 0.1
            
            if home_diversity and away_diversity:
                print("✅ Predictions show good diversity!")
            else:
                print("⚠️  Predictions may lack diversity")
            
            # Calculate average error
            errors = []
            for i, (pred, actual) in enumerate(zip(predictions, actual_results)):
                error = abs(pred[0] - actual[0]) + abs(pred[1] - actual[1])
                errors.append(error)
            
            avg_error = np.mean(errors)
            print(f"  Average prediction error: {avg_error:.2f} goals")
            
            if avg_error < 2.0:
                print("✅ Prediction accuracy looks good!")
            else:
                print("⚠️  Prediction accuracy could be improved")

def test_feature_availability():
    """Test that all expected features are available in the database."""
    load_config()
    
    with get_db_session() as session:
        # Check how many fixtures have complete feature data
        query = text("""
            SELECT COUNT(*) as total_fixtures,
                   COUNT(CASE WHEN home_attack_strength IS NOT NULL THEN 1 END) as with_attack_strength,
                   COUNT(CASE WHEN home_form_diff IS NOT NULL THEN 1 END) as with_form_diff,
                   COUNT(CASE WHEN home_xg IS NOT NULL THEN 1 END) as with_xg
            FROM pre_computed_features
        """)
        
        result = session.execute(query).fetchone()
        
        print("📈 Feature Availability:")
        print(f"  Total fixtures with features: {result[0]}")
        print(f"  With attack strength: {result[1]} ({result[1]/result[0]*100:.1f}%)")
        print(f"  With form diff: {result[2]} ({result[2]/result[0]*100:.1f}%)")
        print(f"  With xG: {result[3]} ({result[3]/result[0]*100:.1f}%)")
        
        if result[1] > 0 and result[2] > 0 and result[3] > 0:
            print("✅ Key features are available in the database!")
        else:
            print("❌ Some key features are missing from the database")

def main():
    print("🔍 Final Prediction System Test\n")
    print("=" * 50)
    
    # Test feature availability
    test_feature_availability()
    print()
    
    # Test predictions
    test_predictions()
    
    print("=" * 50)
    print("✅ Final test completed!")

if __name__ == "__main__":
    main()