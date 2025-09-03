#!/usr/bin/env python3
"""
Test script to debug model loading issues.
"""

import os
import sys
import joblib
import glob
import json
from datetime import datetime

def test_model_loading():
    """Test loading the trained models to identify issues."""
    print("=== Model Loading Test ===")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found")
        return False
    
    # Find latest models by timestamp
    regressor_files = sorted(glob.glob(f"{models_dir}/goal_regressor_*.joblib"))
    classifier_files = sorted(glob.glob(f"{models_dir}/over25_classifier_*.joblib"))
    scaler_files = sorted(glob.glob(f"{models_dir}/feature_scaler_*.joblib"))
    metadata_files = sorted(glob.glob(f"{models_dir}/metadata_*.json"))
    
    print(f"Found {len(regressor_files)} regressor files")
    print(f"Found {len(classifier_files)} classifier files")
    print(f"Found {len(scaler_files)} scaler files")
    print(f"Found {len(metadata_files)} metadata files")
    
    if not all([regressor_files, classifier_files, scaler_files, metadata_files]):
        print("❌ Missing model files")
        return False
    
    # Test loading each type of model
    try:
        print(f"\nTesting regressor: {regressor_files[-1]}")
        regressor = joblib.load(regressor_files[-1])
        print(f"✅ Regressor loaded successfully: {type(regressor)}")
    except Exception as e:
        print(f"❌ Failed to load regressor: {e}")
        return False
    
    try:
        print(f"\nTesting classifier: {classifier_files[-1]}")
        classifier = joblib.load(classifier_files[-1])
        print(f"✅ Classifier loaded successfully: {type(classifier)}")
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return False
    
    try:
        print(f"\nTesting scaler: {scaler_files[-1]}")
        scaler = joblib.load(scaler_files[-1])
        print(f"✅ Scaler loaded successfully: {type(scaler)}")
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
        return False
    
    try:
        print(f"\nTesting metadata: {metadata_files[-1]}")
        with open(metadata_files[-1], 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded successfully")
        print(f"  - Features: {len(metadata.get('feature_names', []))}")
        print(f"  - Training date: {metadata.get('training_date', 'Unknown')}")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False
    
    print("\n🎉 All models loaded successfully!")
    return True

if __name__ == "__main__":
    # Print Python and library versions
    print(f"Python version: {sys.version}")
    
    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not available")
    
    try:
        import joblib
        print(f"Joblib version: {joblib.__version__}")
    except ImportError:
        print("❌ Joblib not available")
    
    print()
    
    success = test_model_loading()
    sys.exit(0 if success else 1)