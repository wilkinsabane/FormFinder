#!/usr/bin/env python3
"""
Script to inspect the structure of models.pkl file
"""

import joblib
import sys

def inspect_models():
    try:
        # Load the models.pkl file
        model_data = joblib.load("models/models.pkl")
        
        print("📋 Models.pkl file structure:")
        print(f"Type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"Keys: {list(model_data.keys())}")
            
            for key, value in model_data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, '__len__') and not isinstance(value, str):
                    try:
                        print(f"    Length: {len(value)}")
                    except:
                        pass
        else:
            print(f"Content: {model_data}")
            
    except Exception as e:
        print(f"❌ Error loading models.pkl: {e}")
        return False
    
    return True

if __name__ == "__main__":
    inspect_models()