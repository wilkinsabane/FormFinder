#!/usr/bin/env python3
"""Analyze the feature discrepancy between enhanced_predictor and database storage."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
import pandas as pd

def get_precomputed_feature_columns():
    """Get actual feature columns from pre_computed_features table."""
    load_config()
    with get_db_session() as db:
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'pre_computed_features'
            AND column_name NOT IN (
                'id', 'fixture_id', 'home_team_id', 'away_team_id', 'match_date', 'league_id',
                'total_goals', 'over_2_5', 'home_score', 'away_score', 'match_result',
                'features_computed_at', 'data_quality_score', 'computation_source', 'h2h_last_updated'
            )
            ORDER BY column_name
        """))
        return [row[0] for row in result.fetchall()]

def get_enhanced_predictor_features():
    """List features that enhanced_predictor.py creates."""
    # Based on the code analysis
    derived_features = [
        'home_attack_strength', 'home_defense_strength', 'away_attack_strength', 'away_defense_strength',
        'home_form_diff', 'away_form_diff', 'home_advantage', 'defensive_home_advantage',
        'h2h_total_goals', 'h2h_competitiveness', 'home_xg', 'away_xg'
    ]
    
    raw_features = [
        'home_avg_goals_for', 'home_avg_goals_against', 'away_avg_goals_for', 'away_avg_goals_against',
        'league_avg_goals'
    ]
    
    metadata_features = ['fixture_id', 'league_id']
    
    # Note: Markov features are conditional and variable in number
    
    return {
        'derived': derived_features,
        'raw': raw_features, 
        'metadata': metadata_features,
        'total_base': len(derived_features) + len(raw_features) + len(metadata_features)
    }

def analyze_discrepancy():
    """Analyze the feature count discrepancy."""
    print("=== FEATURE DISCREPANCY ANALYSIS ===")
    print()
    
    # Get pre-computed features from database
    db_features = get_precomputed_feature_columns()
    print(f"Pre-computed features in database: {len(db_features)}")
    print("Database features:")
    for i, feat in enumerate(db_features, 1):
        print(f"  {i:2d}. {feat}")
    print()
    
    # Get enhanced predictor features
    ep_features = get_enhanced_predictor_features()
    print(f"Enhanced predictor base features: {ep_features['total_base']}")
    print("Enhanced predictor features:")
    print("  Derived features:")
    for i, feat in enumerate(ep_features['derived'], 1):
        print(f"    {i:2d}. {feat}")
    print("  Raw features:")
    for i, feat in enumerate(ep_features['raw'], 1):
        print(f"    {i:2d}. {feat}")
    print("  Metadata features:")
    for i, feat in enumerate(ep_features['metadata'], 1):
        print(f"    {i:2d}. {feat}")
    print()
    
    # Find missing features
    all_ep_features = ep_features['derived'] + ep_features['raw'] + ep_features['metadata']
    missing_in_db = [f for f in all_ep_features if f not in db_features and f not in ['fixture_id', 'league_id']]
    extra_in_db = [f for f in db_features if f not in all_ep_features]
    
    print("=== DISCREPANCY ANALYSIS ===")
    print(f"Features in enhanced_predictor but NOT in database: {len(missing_in_db)}")
    for feat in missing_in_db:
        print(f"  - {feat}")
    print()
    
    print(f"Features in database but NOT in enhanced_predictor: {len(extra_in_db)}")
    for feat in extra_in_db:
        print(f"  + {feat}")
    print()
    
    print("=== CONCLUSION ===")
    print(f"Enhanced predictor extracts: ~{ep_features['total_base']} base features + variable Markov features")
    print(f"Database stores: {len(db_features)} features")
    print(f"Training likely uses: {len(db_features)} features (from database)")
    print()
    print("The discrepancy occurs because:")
    print("1. Enhanced predictor creates derived features on-the-fly")
    print("2. Database only stores raw/basic features")
    print("3. Training uses database features, not enhanced predictor features")
    print("4. Markov features (if enabled) add additional variable count")

if __name__ == "__main__":
    analyze_discrepancy()