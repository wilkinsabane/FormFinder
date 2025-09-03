#!/usr/bin/env python3
"""
Analyze Unused Database Columns

This script identifies database columns that are not used for feature extraction,
training, or prediction and can potentially be removed.

Author: FormFinder2 Team
Created: 2025-01-03
Purpose: Database optimization and cleanup
"""

import logging
import sys
from pathlib import Path
from typing import Set, Dict, List, Any
import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session
from formfinder.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnusedColumnAnalyzer:
    """Analyzes database columns to identify unused ones."""
    
    def __init__(self):
        # Load configuration first
        load_config()
        self.db_session = None
        
    def get_actual_database_columns(self) -> Set[str]:
        """Get actual columns from pre_computed_features table."""
        try:
            with get_db_session() as session:
                query = text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'pre_computed_features'
                    ORDER BY ordinal_position
                """)
                result = session.execute(query)
                actual_columns = {row[0] for row in result}
                logger.info(f"Actual database has {len(actual_columns)} columns in pre_computed_features")
                return actual_columns
        except Exception as e:
            logger.error(f"Could not get actual database columns: {e}")
            return set()
    
    def get_used_columns_from_code_analysis(self) -> Set[str]:
        """Get columns used based on code analysis."""
        # Based on analysis of DatabaseFeatureEngine, TrainingEngine, and other scripts
        used_columns = {
            # Metadata columns (always needed)
            'id', 'fixture_id', 'home_team_id', 'away_team_id', 'league_id', 'match_date',
            'created_at', 'updated_at', 'features_computed_at', 'data_quality_score', 'computation_source',
            
            # Core team form features (used in DatabaseFeatureEngine)
            'home_avg_goals_scored', 'home_avg_goals_conceded', 'home_avg_goals_scored_home',
            'home_avg_goals_conceded_home', 'home_wins_last_5', 'home_draws_last_5',
            'home_losses_last_5', 'home_goals_for_last_5', 'home_goals_against_last_5',
            'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_avg_goals_scored_away',
            'away_avg_goals_conceded_away', 'away_wins_last_5', 'away_draws_last_5',
            'away_losses_last_5', 'away_goals_for_last_5', 'away_goals_against_last_5',
            
            # H2H features (used in DatabaseFeatureEngine)
            'h2h_overall_games', 'h2h_avg_total_goals', 'h2h_overall_home_goals',
            'h2h_overall_away_goals', 'h2h_home_advantage', 'h2h_team1_wins',
            'h2h_team2_wins', 'h2h_draws',
            
            # Weather features (used in DatabaseFeatureEngine)
            'weather_temp_c', 'weather_humidity', 'weather_wind_speed', 
            'weather_precipitation', 'weather_condition',
            
            # Preview features
            'excitement_rating',
            
            # xG features (used in enhanced predictor)
            'home_xg', 'away_xg',
            
            # Team strength features (used in training)
            'home_team_strength', 'away_team_strength', 'home_team_momentum', 'away_team_momentum',
            
            # Sentiment features (used in enhanced predictor)
            'home_team_sentiment', 'away_team_sentiment',
        }
        
        logger.info(f"Code analysis identifies {len(used_columns)} used columns")
        return used_columns
    
    def get_potentially_unused_columns(self) -> Set[str]:
        """Get columns that are potentially unused based on code analysis."""
        # These columns exist in the database model but are not used in current training/prediction
        potentially_unused = {
            # Duplicate weather column
            'weather_temp_f',  # We use weather_temp_c
            
            # Form string columns (we use numeric form features instead)
            'home_form_last_5_games', 'away_form_last_5_games',
            
            # H2H timestamp (metadata, not used for prediction)
            'h2h_last_updated',
            
            # Detailed Markov features (not used in current training)
            'home_team_markov_momentum', 'away_team_markov_momentum',
            'home_team_state_stability', 'away_team_state_stability',
            'home_team_transition_entropy', 'away_team_transition_entropy',
            'home_team_performance_volatility', 'away_team_performance_volatility',
            'home_team_current_state', 'away_team_current_state',
            'home_team_state_duration', 'away_team_state_duration',
            'home_team_expected_next_state', 'away_team_expected_next_state',
            'home_team_state_confidence', 'away_team_state_confidence',
            'markov_match_prediction_confidence', 'markov_outcome_probabilities',
        }
        
        logger.info(f"Identified {len(potentially_unused)} potentially unused columns")
        return potentially_unused
    
    def analyze_column_usage(self) -> Dict[str, Any]:
        """Analyze which columns are used vs unused."""
        logger.info("Starting column usage analysis...")
        
        # Get column sets
        actual_columns = self.get_actual_database_columns()
        used_columns = self.get_used_columns_from_code_analysis()
        potentially_unused = self.get_potentially_unused_columns()
        
        # Find columns that exist in database
        existing_used_columns = actual_columns & used_columns
        existing_unused_columns = actual_columns & potentially_unused
        unknown_columns = actual_columns - used_columns - potentially_unused
        
        # Categorize columns
        metadata_columns = {
            'id', 'fixture_id', 'home_team_id', 'away_team_id', 'league_id',
            'match_date', 'created_at', 'updated_at', 'features_computed_at',
            'data_quality_score', 'computation_source', 'h2h_last_updated'
        }
        
        feature_columns = existing_used_columns - metadata_columns
        unused_feature_columns = existing_unused_columns - metadata_columns
        unused_metadata_columns = existing_unused_columns & metadata_columns
        unknown_feature_columns = unknown_columns - metadata_columns
        unknown_metadata_columns = unknown_columns & metadata_columns
        
        analysis = {
            'total_columns': len(actual_columns),
            'used_columns': len(existing_used_columns),
            'unused_columns': len(existing_unused_columns),
            'unknown_columns': len(unknown_columns),
            'used_feature_columns': len(feature_columns),
            'unused_feature_columns': len(unused_feature_columns),
            'unused_metadata_columns': len(unused_metadata_columns),
            'column_details': {
                'all_columns': sorted(actual_columns),
                'used_columns': sorted(existing_used_columns),
                'unused_columns': sorted(existing_unused_columns),
                'unknown_columns': sorted(unknown_columns),
                'used_feature_columns': sorted(feature_columns),
                'unused_feature_columns': sorted(unused_feature_columns),
                'unused_metadata_columns': sorted(unused_metadata_columns),
                'unknown_feature_columns': sorted(unknown_feature_columns),
                'unknown_metadata_columns': sorted(unknown_metadata_columns),
                'metadata_columns': sorted(metadata_columns & actual_columns)
            }
        }
        
        return analysis
    
    def generate_removal_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for column removal."""
        recommendations = []
        
        unused_feature_cols = analysis['column_details']['unused_feature_columns']
        unused_metadata_cols = analysis['column_details']['unused_metadata_columns']
        unknown_cols = analysis['column_details']['unknown_columns']
        
        if unused_feature_cols:
            recommendations.append(
                f"✅ SAFE TO REMOVE - {len(unused_feature_cols)} unused feature columns:"
            )
            for col in unused_feature_cols:
                recommendations.append(f"  - {col}")
        
        if unused_metadata_cols:
            recommendations.append(
                f"\n⚠️  REVIEW NEEDED - {len(unused_metadata_cols)} unused metadata columns:"
            )
            for col in unused_metadata_cols:
                recommendations.append(f"  - {col}")
        
        if unknown_cols:
            recommendations.append(
                f"\n❓ INVESTIGATE - {len(unknown_cols)} unknown columns (not in analysis):"
            )
            for col in unknown_cols:
                recommendations.append(f"  - {col}")
        
        # Safety warnings
        if unused_feature_cols or unused_metadata_cols or unknown_cols:
            recommendations.extend([
                "\n🛡️  SAFETY CHECKLIST:",
                "1. ✅ Backup database before removing columns",
                "2. ✅ Verify columns are not used in other scripts/tools",
                "3. ✅ Check if columns are needed for future features",
                "4. ✅ Test training and prediction after removal",
                "5. ✅ Consider archiving data instead of deleting",
                "6. ✅ Update SQLAlchemy models after removal"
            ])
        
        return recommendations
    
    def generate_sql_removal_script(self, analysis: Dict[str, Any]) -> str:
        """Generate SQL script to remove unused columns."""
        unused_cols = analysis['column_details']['unused_columns']
        
        if not unused_cols:
            return "-- No columns to remove\n"
        
        sql_script = [
            "-- SQL Script to Remove Unused Columns from pre_computed_features",
            "-- Generated by analyze_unused_columns.py",
            "-- WARNING: BACKUP DATABASE BEFORE RUNNING!",
            "",
            "BEGIN;",
            ""
        ]
        
        for col in sorted(unused_cols):
            sql_script.append(f"ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS {col};")
        
        sql_script.extend([
            "",
            "-- Verify the changes",
            "SELECT column_name FROM information_schema.columns ",
            "WHERE table_name = 'pre_computed_features' ",
            "ORDER BY ordinal_position;",
            "",
            "-- Uncomment the next line to commit changes",
            "-- COMMIT;",
            "ROLLBACK; -- Remove this line when ready to commit"
        ])
        
        return "\n".join(sql_script)
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print detailed analysis report."""
        print("\n" + "="*80)
        print("DATABASE COLUMN USAGE ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total columns in pre_computed_features: {analysis['total_columns']}")
        print(f"  Used columns: {analysis['used_columns']} ({analysis['used_columns']/analysis['total_columns']*100:.1f}%)")
        print(f"  Unused columns: {analysis['unused_columns']} ({analysis['unused_columns']/analysis['total_columns']*100:.1f}%)")
        print(f"  Unknown columns: {analysis['unknown_columns']} ({analysis['unknown_columns']/analysis['total_columns']*100:.1f}%)")
        
        print(f"\n🔧 FEATURE COLUMNS:")
        print(f"  Used feature columns: {analysis['used_feature_columns']}")
        print(f"  Unused feature columns: {analysis['unused_feature_columns']}")
        
        print(f"\n📋 METADATA COLUMNS:")
        print(f"  Used metadata columns: {len(analysis['column_details']['metadata_columns']) - analysis['unused_metadata_columns']}")
        print(f"  Unused metadata columns: {analysis['unused_metadata_columns']}")
        
        if analysis['unused_columns'] > 0:
            print(f"\n❌ UNUSED COLUMNS ({analysis['unused_columns']}):")
            for col in analysis['column_details']['unused_columns']:
                print(f"  - {col}")
        
        if analysis['unknown_columns'] > 0:
            print(f"\n❓ UNKNOWN COLUMNS ({analysis['unknown_columns']}):")
            for col in analysis['column_details']['unknown_columns']:
                print(f"  - {col}")
        
        recommendations = self.generate_removal_recommendations(analysis)
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(rec)
        else:
            print(f"\n✅ All columns are being used - no removal needed!")
    
    def close(self):
        """Close database session."""
        # No need to close session as we use context manager
        pass


def main():
    """Main analysis function."""
    analyzer = None
    try:
        analyzer = UnusedColumnAnalyzer()
        analysis = analyzer.analyze_column_usage()
        analyzer.print_analysis_report(analysis)
        
        # Save detailed analysis to file
        output_file = project_root / "unused_columns_analysis.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n📄 Detailed analysis saved to: {output_file}")
        
        # Generate SQL removal script
        sql_script = analyzer.generate_sql_removal_script(analysis)
        sql_file = project_root / "remove_unused_columns.sql"
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(sql_script)
        print(f"📄 SQL removal script saved to: {sql_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        if analyzer:
            analyzer.close()


if __name__ == "__main__":
    main()