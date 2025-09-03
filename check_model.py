import joblib
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from scripts.database_feature_engine import DatabaseFeatureEngine
from formfinder.database import get_db_session

# Load configuration
load_config()

# Load the model
model_data = joblib.load('models/enhanced_predictor.joblib')

print("Model keys:", list(model_data.keys()))

# Check scaler info
if 'scalers' in model_data and 'global' in model_data['scalers']:
    scaler = model_data['scalers']['global']
    print(f"Scaler expects {scaler.n_features_in_} features")
    if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
        print(f"Scaler feature names: {list(scaler.feature_names_in_)}")
    else:
        print("No feature names saved in scaler")
        
        # Get the feature columns from DatabaseFeatureEngine
        print("\nGetting feature columns from DatabaseFeatureEngine...")
        with get_db_session() as session:
            engine = DatabaseFeatureEngine(session)
            feature_cols = engine.feature_columns
            print(f"DatabaseFeatureEngine has {len(feature_cols)} features:")
            for i, col in enumerate(feature_cols, 1):
                print(f"{i:2d}. {col}")
else:
    print("No global scaler found in model")

# Check if there's any metadata about features
if 'model_configs' in model_data:
    configs = model_data['model_configs']
    print('Model configs keys:', list(configs.keys()))
    if 'feature_columns' in configs:
        print('Feature columns in configs:', configs['feature_columns'])