#!/usr/bin/env python3
"""
Debug script to test what features the enhanced predictor actually returns
and identify missing mappings in the unified feature populator.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from enhanced_predictor import EnhancedGoalPredictor
from formfinder.database import get_db_session
from sqlalchemy import text
import pandas as pd
import json
from typing import Dict, Any

def test_enhanced_features():
    """Test what features the enhanced predictor returns for a sample fixture."""
    
    with get_db_session() as session:
        # Get a sample fixture with completed results
        query = text("""
        SELECT id, home_team_id, away_team_id, match_date, league_id, 
               home_score, away_score
        FROM fixtures 
        WHERE home_score IS NOT NULL 
          AND away_score IS NOT NULL
          AND match_date >= '2024-01-01'
        ORDER BY match_date DESC
        LIMIT 5
        """)
        
        fixtures_df = pd.read_sql_query(query, session.bind)
        
        if len(fixtures_df) == 0:
            print("No completed fixtures found")
            return
        
        print(f"Found {len(fixtures_df)} completed fixtures to test")
        print("\nFixtures:")
        for _, row in fixtures_df.iterrows():
            print(f"  Fixture {row['id']}: {row['match_date']} - Score: {row['home_score']}-{row['away_score']}")
        
        # Initialize enhanced predictor
        print("\nInitializing enhanced predictor...")
        try:
            predictor = EnhancedGoalPredictor()
            print("Enhanced predictor initialized successfully")
        except Exception as e:
            print(f"Failed to initialize enhanced predictor: {e}")
            return
        
        # Test feature extraction for each fixture
        for _, row in fixtures_df.iterrows():
            fixture_id = row['id']
            print(f"\n{'='*60}")
            print(f"Testing fixture {fixture_id}")
            print(f"Match: {row['match_date']} - Score: {row['home_score']}-{row['away_score']}")
            print(f"{'='*60}")
            
            try:
                features = predictor.extract_enhanced_features(fixture_id)
                
                if features is None:
                    print("❌ No features returned")
                    continue
                
                print(f"✅ Extracted {len(features)} features")
                
                # Check for the specific features we're looking for
                target_features = [
                    'home_attack_strength', 'home_defense_strength',
                    'away_attack_strength', 'away_defense_strength',
                    'home_form_diff', 'away_form_diff',
                    'home_team_form_score', 'away_team_form_score',
                    'home_team_strength', 'away_team_strength',
                    'home_xg', 'away_xg'
                ]
                
                print("\nTarget features status:")
                for feature in target_features:
                    if feature in features:
                        value = features[feature]
                        print(f"  ✅ {feature}: {value}")
                    else:
                        print(f"  ❌ {feature}: MISSING")
                
                # Show all available features
                print(f"\nAll {len(features)} available features:")
                for key, value in sorted(features.items()):
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
                # Check for features that might be missing from unified script mapping
                unified_mapped_features = {
                    'home_xg', 'away_xg', 'home_team_strength', 'away_team_strength',
                    'home_team_momentum', 'away_team_momentum', 'home_team_sentiment',
                    'away_team_sentiment'
                }
                
                missing_from_unified = set(features.keys()) - unified_mapped_features - {'fixture_id', 'league_id'}
                if missing_from_unified:
                    print(f"\n⚠️ Features NOT mapped in unified script ({len(missing_from_unified)}):")
                    for feature in sorted(missing_from_unified):
                        value = features[feature]
                        if isinstance(value, (int, float)):
                            print(f"  {feature}: {value:.4f}")
                        else:
                            print(f"  {feature}: {value}")
                
                break  # Only test first fixture for detailed output
                
            except Exception as e:
                print(f"❌ Error extracting features: {e}")
                import traceback
                traceback.print_exc()

def check_database_columns():
    """Check what columns exist in the pre_computed_features table."""
    
    with get_db_session() as session:
        # Get table schema
        query = text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'pre_computed_features'
        ORDER BY ordinal_position
        """)
        
        try:
            columns_df = pd.read_sql_query(query, session.bind)
            
            print(f"\nPre-computed features table has {len(columns_df)} columns:")
            
            # Check for the specific columns we're interested in
            target_columns = [
                'home_attack_strength', 'home_defense_strength',
                'away_attack_strength', 'away_defense_strength', 
                'home_form_diff', 'away_form_diff',
                'home_team_form_score', 'away_team_form_score'
            ]
            
            existing_columns = set(columns_df['column_name'].tolist())
            
            print("\nTarget columns status:")
            for col in target_columns:
                if col in existing_columns:
                    print(f"  ✅ {col}: EXISTS")
                else:
                    print(f"  ❌ {col}: MISSING")
            
            print(f"\nAll columns ({len(columns_df)}):")
            for _, row in columns_df.iterrows():
                nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  {row['column_name']}: {row['data_type']} {nullable}")
                
        except Exception as e:
            print(f"Error checking database columns: {e}")

def main():
    """Main function to test enhanced features."""
    print("=== Enhanced Features Debug Test ===")
    print("Testing EnhancedGoalPredictor.extract_enhanced_features()\n")
    
    # Load configuration
    load_config()
    
    print("1. Checking database columns...")
    check_database_columns()
    
    print("\n2. Testing enhanced feature extraction...")
    test_enhanced_features()

if __name__ == "__main__":
    main()