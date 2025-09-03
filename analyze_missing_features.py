import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from formfinder.database import get_db_session
from joblib import load
from sqlalchemy import text
import pandas as pd

def analyze_missing_features():
    """Analyze all missing features to understand what they represent."""
    load_config()
    
    # Load the expected features from metadata
    import glob
    import json
    
    # Find the latest metadata file
    metadata_files = glob.glob('models/metadata_*.json')
    if not metadata_files:
        raise FileNotFoundError("No metadata files found")
    
    latest_metadata = max(metadata_files)
    print(f"Loading feature list from: {latest_metadata}")
    
    with open(latest_metadata, 'r') as f:
        metadata = json.load(f)
    
    expected_features = metadata['feature_columns']
    print(f"Total expected features: {len(expected_features)}")
    
    # Get available features from database
    with get_db_session() as db:
        result = db.execute(text("SELECT * FROM pre_computed_features LIMIT 1"))
        row = result.fetchone()
        if row:
            available_columns = list(row._mapping.keys())
        else:
            available_columns = []
    
    # Remove non-feature columns
    non_feature_cols = ['id', 'fixture_id', 'created_at', 'updated_at']
    available_features = [col for col in available_columns if col not in non_feature_cols]
    
    print(f"Available features in database: {len(available_features)}")
    
    # Find missing features
    missing_features = [f for f in expected_features if f not in available_features]
    matching_features = [f for f in expected_features if f in available_features]
    
    print(f"\nMissing features ({len(missing_features)}):")
    print("=" * 50)
    
    # Categorize missing features by type
    categories = {
        'markov': [],
        'position': [],
        'league': [],
        'momentum': [],
        'xg': [],
        'trend': [],
        'form': [],
        'strength': [],
        'other': []
    }
    
    for feature in missing_features:
        feature_lower = feature.lower()
        if 'markov' in feature_lower:
            categories['markov'].append(feature)
        elif 'position' in feature_lower:
            categories['position'].append(feature)
        elif 'league' in feature_lower:
            categories['league'].append(feature)
        elif 'momentum' in feature_lower:
            categories['momentum'].append(feature)
        elif 'xg' in feature_lower:
            categories['xg'].append(feature)
        elif 'trend' in feature_lower:
            categories['trend'].append(feature)
        elif 'form' in feature_lower:
            categories['form'].append(feature)
        elif 'strength' in feature_lower:
            categories['strength'].append(feature)
        else:
            categories['other'].append(feature)
    
    # Print categorized missing features
    for category, features in categories.items():
        if features:
            print(f"\n{category.upper()} Features ({len(features)}):")
            for i, feature in enumerate(features, 1):
                print(f"  {i:2d}. {feature}")
    
    print(f"\n\nMatching features ({len(matching_features)}):")
    print("=" * 50)
    for i, feature in enumerate(matching_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # Save detailed analysis to file
    with open('missing_features_analysis.txt', 'w') as f:
        f.write(f"MISSING FEATURES ANALYSIS\n")
        f.write(f"========================\n\n")
        f.write(f"Total expected features: {len(expected_features)}\n")
        f.write(f"Available features: {len(available_features)}\n")
        f.write(f"Missing features: {len(missing_features)}\n")
        f.write(f"Match percentage: {len(matching_features)/len(expected_features)*100:.1f}%\n\n")
        
        f.write("CATEGORIZED MISSING FEATURES:\n")
        f.write("============================\n\n")
        
        for category, features in categories.items():
            if features:
                f.write(f"{category.upper()} Features ({len(features)}):\n")
                for feature in features:
                    f.write(f"  - {feature}\n")
                f.write("\n")
        
        f.write("\nALL MISSING FEATURES (alphabetical):\n")
        f.write("===================================\n")
        for feature in sorted(missing_features):
            f.write(f"  - {feature}\n")
        
        f.write("\n\nMATCHING FEATURES:\n")
        f.write("=================\n")
        for feature in sorted(matching_features):
            f.write(f"  - {feature}\n")
    
    print(f"\nDetailed analysis saved to 'missing_features_analysis.txt'")
    
    db.close()

if __name__ == "__main__":
    analyze_missing_features()