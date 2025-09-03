#!/usr/bin/env python3
"""Debug script to identify VARCHAR(20) truncation issues."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from enhanced_predictor import EnhancedGoalPredictor
from sqlalchemy import text
import json

def debug_varchar_fields():
    """Debug VARCHAR(20) fields that might cause truncation."""
    print("=== VARCHAR(20) Truncation Debug ===")
    
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Get fixture 160 specifically (the one causing the error)
        result = session.execute(text("""
            SELECT f.id, f.home_team_id, f.away_team_id, f.match_date, f.league_id,
                   f.home_score, f.away_score
            FROM fixtures f
            WHERE f.id = 160
        """))
        
        fixture = result.fetchone()
        if not fixture:
            print("Fixture 160 not found")
            return
            
        print(f"Testing fixture {fixture.id}: {fixture.home_team_id} vs {fixture.away_team_id}")
        
        # Initialize enhanced predictor
        try:
            enhanced_predictor = EnhancedGoalPredictor()
            print("✅ Enhanced predictor initialized")
        except Exception as e:
            print(f"❌ Failed to initialize enhanced predictor: {e}")
            return
            
        # Extract enhanced features
        try:
            enhanced_features = enhanced_predictor.extract_enhanced_features(fixture.id)
            
            print(f"✅ Enhanced features extracted")
            
            # Check VARCHAR(20) fields specifically
            varchar_fields = {
                'home_team_current_state': enhanced_features.get('markov_home_current_state', 'average'),
                'away_team_current_state': enhanced_features.get('markov_away_current_state', 'average'),
                'home_team_expected_next_state': enhanced_features.get('markov_home_expected_next_state', 'average'),
                'away_team_expected_next_state': enhanced_features.get('markov_away_expected_next_state', 'average')
            }
            
            print("\n=== VARCHAR(20) Fields Analysis ===")
            for field_name, value in varchar_fields.items():
                value_str = str(value)
                length = len(value_str)
                status = "✅ OK" if length <= 20 else f"❌ TOO LONG ({length} chars)"
                print(f"{field_name}: '{value_str}' [{length} chars] {status}")
                
            # Check for any other string fields that might be problematic
            print("\n=== All String Fields Analysis ===")
            string_fields = {k: v for k, v in enhanced_features.items() if isinstance(v, str)}
            
            for field_name, value in string_fields.items():
                length = len(value)
                if length > 20:
                    print(f"⚠️  {field_name}: '{value}' [{length} chars] - POTENTIAL ISSUE")
                elif length > 15:
                    print(f"🔶 {field_name}: '{value}' [{length} chars] - CLOSE TO LIMIT")
                    
        except Exception as e:
            print(f"❌ Failed to extract enhanced features: {e}")
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    debug_varchar_fields()