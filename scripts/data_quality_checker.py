#!/usr/bin/env python3
"""
Data Quality Checker

Ensures data quality and feature integrity for the FormFinder system.
This component is part of the Enhanced Data Collection Layer as specified in the PRD.

Usage:
    python scripts/data_quality_checker.py [--output-format json|text]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from formfinder.database import get_db_session
from formfinder.config import load_config
from sqlalchemy import text
import numpy as np
import pandas as pd


class DataQualityChecker:
    """Ensures data quality and feature integrity."""
    
    def __init__(self, db_session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds from PRD
        self.MIN_FEATURE_COMPLETENESS = 0.9
        self.MAX_MISSING_H2H_RATE = 0.2
        self.MAX_COMPUTATION_FAILURE_RATE = 0.1
        
        # Feature value ranges for validation
        self.FEATURE_RANGES = {
            'home_avg_goals_scored': (0.0, 8.0),
            'home_avg_goals_conceded': (0.0, 8.0),
            'away_avg_goals_scored': (0.0, 8.0),
            'away_avg_goals_conceded': (0.0, 8.0),
            'h2h_total_matches': (0, 50),
            'h2h_avg_total_goals': (0.0, 10.0),
            'excitement_rating': (0.0, 10.0),
            'temperature': (-20.0, 50.0),
            'humidity': (0.0, 100.0),
            'wind_speed': (0.0, 50.0),
            'total_goals': (0, 15),
            'data_quality_score': (0.0, 1.0)
        }
    
    def validate_feature_distributions(self) -> Dict[str, Any]:
        """Validate that feature distributions are reasonable."""
        self.logger.info("Validating feature distributions...")
        
        try:
            # Get all pre-computed features
            query = text("""
                SELECT home_avg_goals_scored, home_avg_goals_conceded,
                       away_avg_goals_scored, away_avg_goals_conceded,
                       h2h_total_matches, h2h_avg_total_goals,
                       excitement_rating, temperature, humidity, wind_speed,
                       total_goals, data_quality_score
                FROM pre_computed_features
                WHERE data_quality_score >= :min_quality
            """)
            
            result = self.db_session.execute(query, {
                'min_quality': self.MIN_FEATURE_COMPLETENESS
            })
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(result.fetchall(), columns=[
                'home_avg_goals_scored', 'home_avg_goals_conceded',
                'away_avg_goals_scored', 'away_avg_goals_conceded',
                'h2h_total_matches', 'h2h_avg_total_goals',
                'excitement_rating', 'temperature', 'humidity', 'wind_speed',
                'total_goals', 'data_quality_score'
            ])
            
            if df.empty:
                return {
                    'status': 'warning',
                    'message': 'No high-quality features found for validation',
                    'total_records': 0
                }
            
            validation_results = {
                'status': 'pass',
                'total_records': len(df),
                'feature_statistics': {},
                'range_violations': {},
                'distribution_warnings': []
            }
            
            # Validate each feature
            for feature in df.columns:
                if feature in self.FEATURE_RANGES:
                    min_val, max_val = self.FEATURE_RANGES[feature]
                    
                    # Calculate statistics
                    feature_stats = {
                        'mean': float(df[feature].mean()),
                        'std': float(df[feature].std()),
                        'min': float(df[feature].min()),
                        'max': float(df[feature].max()),
                        'null_count': int(df[feature].isnull().sum()),
                        'null_percentage': float(df[feature].isnull().sum() / len(df) * 100)
                    }
                    
                    validation_results['feature_statistics'][feature] = feature_stats
                    
                    # Check for range violations
                    violations = df[(df[feature] < min_val) | (df[feature] > max_val)][feature]
                    if len(violations) > 0:
                        validation_results['range_violations'][feature] = {
                            'count': len(violations),
                            'percentage': len(violations) / len(df) * 100,
                            'values': violations.tolist()[:10]  # First 10 violations
                        }
                        validation_results['status'] = 'warning'
                    
                    # Check for distribution anomalies
                    if feature_stats['std'] == 0:
                        validation_results['distribution_warnings'].append(
                            f"{feature}: No variation (std=0)"
                        )
                    elif feature_stats['null_percentage'] > 10:
                        validation_results['distribution_warnings'].append(
                            f"{feature}: High null percentage ({feature_stats['null_percentage']:.1f}%)"
                        )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating feature distributions: {e}")
            return {
                'status': 'error',
                'message': f"Validation failed: {str(e)}"
            }
    
    def check_data_consistency(self) -> Dict[str, Any]:
        """Check for data consistency issues."""
        self.logger.info("Checking data consistency...")
        
        try:
            consistency_results = {
                'status': 'pass',
                'checks': {},
                'issues': []
            }
            
            # Check 1: Fixture data consistency
            fixture_check = self._check_fixture_consistency()
            consistency_results['checks']['fixture_consistency'] = fixture_check
            if fixture_check['status'] != 'pass':
                consistency_results['status'] = 'warning'
                consistency_results['issues'].extend(fixture_check.get('issues', []))
            
            # Check 2: Feature computation completeness
            completeness_check = self._check_feature_completeness()
            consistency_results['checks']['feature_completeness'] = completeness_check
            if completeness_check['status'] != 'pass':
                consistency_results['status'] = 'warning'
                consistency_results['issues'].extend(completeness_check.get('issues', []))
            
            # Check 3: H2H cache consistency
            h2h_check = self._check_h2h_consistency()
            consistency_results['checks']['h2h_consistency'] = h2h_check
            if h2h_check['status'] != 'pass':
                consistency_results['status'] = 'warning'
                consistency_results['issues'].extend(h2h_check.get('issues', []))
            
            # Check 4: Computation log integrity
            log_check = self._check_computation_log_integrity()
            consistency_results['checks']['computation_log'] = log_check
            if log_check['status'] != 'pass':
                consistency_results['status'] = 'warning'
                consistency_results['issues'].extend(log_check.get('issues', []))
            
            return consistency_results
            
        except Exception as e:
            self.logger.error(f"Error checking data consistency: {e}")
            return {
                'status': 'error',
                'message': f"Consistency check failed: {str(e)}"
            }
    
    def identify_outliers(self) -> Dict[str, Any]:
        """Identify potential data outliers."""
        self.logger.info("Identifying data outliers...")
        
        try:
            # Get feature data for outlier detection
            query = text("""
                SELECT fixture_id, home_avg_goals_scored, home_avg_goals_conceded,
                       away_avg_goals_scored, away_avg_goals_conceded,
                       h2h_avg_total_goals, excitement_rating,
                       total_goals, data_quality_score
                FROM pre_computed_features
                WHERE data_quality_score >= :min_quality
            """)
            
            result = self.db_session.execute(query, {
                'min_quality': self.MIN_FEATURE_COMPLETENESS
            })
            
            df = pd.DataFrame(result.fetchall(), columns=[
                'fixture_id', 'home_avg_goals_scored', 'home_avg_goals_conceded',
                'away_avg_goals_scored', 'away_avg_goals_conceded',
                'h2h_avg_total_goals', 'excitement_rating',
                'total_goals', 'data_quality_score'
            ])
            
            if df.empty:
                return {
                    'status': 'warning',
                    'message': 'No data available for outlier detection'
                }
            
            outlier_results = {
                'status': 'pass',
                'total_records': len(df),
                'outliers_by_feature': {},
                'extreme_outliers': []
            }
            
            # Detect outliers using IQR method
            numeric_features = ['home_avg_goals_scored', 'home_avg_goals_conceded',
                              'away_avg_goals_scored', 'away_avg_goals_conceded',
                              'h2h_avg_total_goals', 'excitement_rating', 'total_goals']
            
            for feature in numeric_features:
                if feature in df.columns:
                    Q1 = df[feature].quantile(0.25)
                    Q3 = df[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
                    
                    if len(outliers) > 0:
                        outlier_results['outliers_by_feature'][feature] = {
                            'count': len(outliers),
                            'percentage': len(outliers) / len(df) * 100,
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'fixture_ids': outliers['fixture_id'].tolist()[:10]
                        }
                        
                        # Check for extreme outliers (beyond 3*IQR)
                        extreme_lower = Q1 - 3 * IQR
                        extreme_upper = Q3 + 3 * IQR
                        extreme_outliers = outliers[
                            (outliers[feature] < extreme_lower) | 
                            (outliers[feature] > extreme_upper)
                        ]
                        
                        if len(extreme_outliers) > 0:
                            outlier_results['extreme_outliers'].extend([
                                {
                                    'fixture_id': int(row['fixture_id']),
                                    'feature': feature,
                                    'value': float(row[feature]),
                                    'expected_range': f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                                }
                                for _, row in extreme_outliers.iterrows()
                            ])
            
            if outlier_results['extreme_outliers']:
                outlier_results['status'] = 'warning'
            
            return outlier_results
            
        except Exception as e:
            self.logger.error(f"Error identifying outliers: {e}")
            return {
                'status': 'error',
                'message': f"Outlier detection failed: {str(e)}"
            }
    
    def _check_fixture_consistency(self) -> Dict[str, Any]:
        """Check fixture data consistency."""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as total_fixtures,
                    COUNT(CASE WHEN home_score IS NULL OR away_score IS NULL THEN 1 END) as missing_scores,
                    COUNT(CASE WHEN total_goals != (home_score + away_score) THEN 1 END) as score_mismatch
                FROM pre_computed_features
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            """)
            
            result = self.db_session.execute(query).fetchone()
            
            issues = []
            if result.missing_scores > 0:
                issues.append(f"{result.missing_scores} fixtures have missing scores")
            if result.score_mismatch > 0:
                issues.append(f"{result.score_mismatch} fixtures have score/total_goals mismatch")
            
            return {
                'status': 'pass' if not issues else 'warning',
                'total_fixtures': result.total_fixtures,
                'missing_scores': result.missing_scores,
                'score_mismatches': result.score_mismatch,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Fixture consistency check failed: {str(e)}"
            }
    
    def _check_feature_completeness(self) -> Dict[str, Any]:
        """Check feature computation completeness."""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as total_features,
                    COUNT(CASE WHEN data_quality_score >= :min_quality THEN 1 END) as high_quality,
                    AVG(data_quality_score) as avg_quality_score
                FROM pre_computed_features
            """)
            
            result = self.db_session.execute(query, {
                'min_quality': self.MIN_FEATURE_COMPLETENESS
            }).fetchone()
            
            completeness_rate = result.high_quality / result.total_features if result.total_features > 0 else 0
            
            issues = []
            if completeness_rate < self.MIN_FEATURE_COMPLETENESS:
                issues.append(
                    f"Feature completeness rate ({completeness_rate:.2%}) below threshold "
                    f"({self.MIN_FEATURE_COMPLETENESS:.2%})"
                )
            
            return {
                'status': 'pass' if not issues else 'warning',
                'total_features': result.total_features,
                'high_quality_features': result.high_quality,
                'completeness_rate': completeness_rate,
                'avg_quality_score': float(result.avg_quality_score) if result.avg_quality_score else 0.0,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Feature completeness check failed: {str(e)}"
            }
    
    def _check_h2h_consistency(self) -> Dict[str, Any]:
        """Check H2H cache consistency."""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as total_h2h_records,
                    COUNT(CASE WHEN last_fetched_at < NOW() - INTERVAL '24 hours' THEN 1 END) as stale_records,
                    COUNT(CASE WHEN overall_games_played = 0 THEN 1 END) as empty_h2h
                FROM h2h_cache
            """)
            
            result = self.db_session.execute(query).fetchone()
            
            issues = []
            if result.stale_records > 0:
                issues.append(f"{result.stale_records} H2H records are stale (>24h old)")
            if result.empty_h2h > result.total_h2h_records * self.MAX_MISSING_H2H_RATE:
                issues.append(f"High rate of empty H2H records: {result.empty_h2h}")
            
            return {
                'status': 'pass' if not issues else 'warning',
                'total_h2h_records': result.total_h2h_records,
                'stale_records': result.stale_records,
                'empty_h2h_records': result.empty_h2h,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"H2H consistency check failed: {str(e)}"
            }
    
    def _check_computation_log_integrity(self) -> Dict[str, Any]:
        """Check computation log integrity."""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as total_computations,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_computations,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as recent_computations
                FROM feature_computation_log
                WHERE created_at >= NOW() - INTERVAL '7 days'
            """)
            
            result = self.db_session.execute(query).fetchone()
            
            failure_rate = result.failed_computations / result.total_computations if result.total_computations > 0 else 0
            
            issues = []
            if failure_rate > self.MAX_COMPUTATION_FAILURE_RATE:
                issues.append(
                    f"Computation failure rate ({failure_rate:.2%}) above threshold "
                    f"({self.MAX_COMPUTATION_FAILURE_RATE:.2%})"
                )
            
            return {
                'status': 'pass' if not issues else 'warning',
                'total_computations': result.total_computations,
                'failed_computations': result.failed_computations,
                'recent_computations': result.recent_computations,
                'failure_rate': failure_rate,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Computation log check failed: {str(e)}"
            }
    
    def run_full_quality_check(self) -> Dict[str, Any]:
        """Run all quality checks and return comprehensive report."""
        self.logger.info("Running full data quality check...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'pass',
            'checks': {
                'feature_distributions': self.validate_feature_distributions(),
                'data_consistency': self.check_data_consistency(),
                'outlier_detection': self.identify_outliers()
            },
            'summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'warnings': 0
            }
        }
        
        # Aggregate status and count issues
        for check_name, check_result in report['checks'].items():
            if check_result['status'] == 'error':
                report['overall_status'] = 'error'
                report['summary']['critical_issues'] += 1
            elif check_result['status'] == 'warning':
                if report['overall_status'] != 'error':
                    report['overall_status'] = 'warning'
                report['summary']['warnings'] += 1
            
            # Count specific issues
            if 'issues' in check_result:
                report['summary']['total_issues'] += len(check_result['issues'])
        
        self.logger.info(f"Quality check complete. Status: {report['overall_status']}")
        return report


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """Main function to run data quality checks."""
    parser = argparse.ArgumentParser(description='Check data quality for FormFinder')
    parser.add_argument('--output-format', choices=['json', 'text'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--save-report', type=str,
                       help='Save report to file')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        load_config()
        
        # Use database session context manager
        with get_db_session() as db_session:
            # Create quality checker and run checks (without creating tables)
            checker = DataQualityChecker(db_session)
            report = checker.run_full_quality_check()
        
            # Output results
            if args.output_format == 'json':
                output = json.dumps(report, indent=2, default=str)
            else:
                output = format_text_report(report)
            
            print(output)
            
            # Save report if requested
            if args.save_report:
                with open(args.save_report, 'w') as f:
                    f.write(output)
                logger.info(f"Report saved to {args.save_report}")
            
            # Exit with appropriate code
            if report['overall_status'] == 'error':
                sys.exit(1)
            elif report['overall_status'] == 'warning':
                sys.exit(2)
            else:
                sys.exit(0)
            
    except Exception as e:
        logger.error(f"Data quality check failed: {e}")
        sys.exit(1)


def format_text_report(report: Dict[str, Any]) -> str:
    """Format report as human-readable text."""
    lines = [
        "=" * 60,
        "FormFinder Data Quality Report",
        "=" * 60,
        f"Timestamp: {report['timestamp']}",
        f"Overall Status: {report['overall_status'].upper()}",
        "",
        "Summary:",
        f"  Total Issues: {report['summary']['total_issues']}",
        f"  Critical Issues: {report['summary']['critical_issues']}",
        f"  Warnings: {report['summary']['warnings']}",
        ""
    ]
    
    for check_name, check_result in report['checks'].items():
        lines.extend([
            f"{check_name.replace('_', ' ').title()}:",
            f"  Status: {check_result['status'].upper()}"
        ])
        
        if 'issues' in check_result and check_result['issues']:
            lines.append("  Issues:")
            for issue in check_result['issues']:
                lines.append(f"    - {issue}")
        
        if check_name == 'feature_distributions' and 'feature_statistics' in check_result:
            lines.append("  Feature Statistics:")
            for feature, stats in check_result['feature_statistics'].items():
                lines.append(f"    {feature}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == '__main__':
    main()