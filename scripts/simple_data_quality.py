#!/usr/bin/env python3
"""
Simplified Data Quality Checker

A working version that avoids database table creation issues.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def check_data_quality():
    """Run comprehensive data quality checks."""
    try:
        # Load configuration
        load_config()
        config = get_config()
        
        # Create direct database connection
        engine = create_engine(config.get_database_url())
        Session = sessionmaker(bind=engine)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'pass',
            'checks': {},
            'summary': {
                'total_issues': 0,
                'warnings': 0,
                'errors': 0
            }
        }
        
        with Session() as session:
            # 1. Check pre-computed features quality
            print("🔍 Checking pre-computed features...")
            features_query = text("""
                SELECT COUNT(*) as total_features,
                       COUNT(CASE WHEN data_quality_score >= 0.9 THEN 1 END) as high_quality_features,
                       AVG(data_quality_score) as avg_quality_score,
                       MIN(data_quality_score) as min_quality_score,
                       MAX(data_quality_score) as max_quality_score
                FROM pre_computed_features
            """)
            
            features_result = session.execute(features_query).fetchone()
            completeness_rate = features_result.high_quality_features / features_result.total_features if features_result.total_features > 0 else 0
            
            features_check = {
                'status': 'pass' if completeness_rate >= 0.9 else 'warning',
                'total_features': features_result.total_features,
                'high_quality_features': features_result.high_quality_features,
                'completeness_rate': completeness_rate,
                'avg_quality_score': float(features_result.avg_quality_score) if features_result.avg_quality_score else 0.0,
                'min_quality_score': float(features_result.min_quality_score) if features_result.min_quality_score else 0.0,
                'max_quality_score': float(features_result.max_quality_score) if features_result.max_quality_score else 0.0,
                'issues': [] if completeness_rate >= 0.9 else [f"Low feature completeness: {completeness_rate:.1%}"]
            }
            report['checks']['feature_quality'] = features_check
            
            # 2. Check fixtures data consistency
            print("⚽ Checking fixtures data...")
            fixtures_query = text("""
                SELECT COUNT(*) as total_fixtures,
                       COUNT(CASE WHEN status = 'finished' THEN 1 END) as finished_fixtures,
                       COUNT(CASE WHEN status = 'finished' AND (home_score IS NULL OR away_score IS NULL) THEN 1 END) as missing_scores,
                       COUNT(CASE WHEN match_date > NOW() THEN 1 END) as upcoming_fixtures
                FROM fixtures
            """)
            
            fixtures_result = session.execute(fixtures_query).fetchone()
            missing_score_rate = fixtures_result.missing_scores / fixtures_result.finished_fixtures if fixtures_result.finished_fixtures > 0 else 0
            
            fixtures_check = {
                'status': 'pass' if missing_score_rate < 0.05 else 'warning',
                'total_fixtures': fixtures_result.total_fixtures,
                'finished_fixtures': fixtures_result.finished_fixtures,
                'upcoming_fixtures': fixtures_result.upcoming_fixtures,
                'missing_scores': fixtures_result.missing_scores,
                'missing_score_rate': missing_score_rate,
                'issues': [] if missing_score_rate < 0.05 else [f"High missing score rate: {missing_score_rate:.1%}"]
            }
            report['checks']['fixtures_consistency'] = fixtures_check
            
            # 3. Check H2H cache status
            print("🔄 Checking H2H cache...")
            try:
                h2h_query = text("""
                    SELECT COUNT(*) as total_h2h_records,
                           COUNT(CASE WHEN last_fetched_at < NOW() - INTERVAL '7 days' THEN 1 END) as stale_records,
                           COUNT(CASE WHEN overall_games_played = 0 THEN 1 END) as empty_h2h,
                           AVG(overall_games_played) as avg_games_played
                    FROM h2h_cache
                """)
                
                h2h_result = session.execute(h2h_query).fetchone()
                stale_rate = h2h_result.stale_records / h2h_result.total_h2h_records if h2h_result.total_h2h_records > 0 else 0
                empty_rate = h2h_result.empty_h2h / h2h_result.total_h2h_records if h2h_result.total_h2h_records > 0 else 0
                
                h2h_issues = []
                if stale_rate > 0.2:
                    h2h_issues.append(f"High stale record rate: {stale_rate:.1%}")
                if empty_rate > 0.1:
                    h2h_issues.append(f"High empty H2H rate: {empty_rate:.1%}")
                
                h2h_check = {
                    'status': 'pass' if not h2h_issues else 'warning',
                    'total_h2h_records': h2h_result.total_h2h_records,
                    'stale_records': h2h_result.stale_records,
                    'empty_h2h_records': h2h_result.empty_h2h,
                    'avg_games_played': float(h2h_result.avg_games_played) if h2h_result.avg_games_played else 0.0,
                    'stale_rate': stale_rate,
                    'empty_rate': empty_rate,
                    'issues': h2h_issues
                }
                report['checks']['h2h_cache'] = h2h_check
                
            except Exception as e:
                report['checks']['h2h_cache'] = {
                    'status': 'error',
                    'message': f"H2H cache check failed: {str(e)}"
                }
            
            # 4. Check computation log
            print("📝 Checking computation log...")
            try:
                log_query = text("""
                    SELECT COUNT(*) as total_computations,
                           COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_computations,
                           COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_computations,
                           AVG(computation_time_ms) as avg_computation_time,
                           COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as recent_computations
                    FROM feature_computation_log
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                """)
                
                log_result = session.execute(log_query).fetchone()
                failure_rate = log_result.failed_computations / log_result.total_computations if log_result.total_computations > 0 else 0
                
                log_issues = []
                if failure_rate > 0.1:
                    log_issues.append(f"High computation failure rate: {failure_rate:.1%}")
                if log_result.recent_computations == 0:
                    log_issues.append("No recent computations in last 24 hours")
                
                log_check = {
                    'status': 'pass' if not log_issues else 'warning',
                    'total_computations': log_result.total_computations,
                    'successful_computations': log_result.successful_computations,
                    'failed_computations': log_result.failed_computations,
                    'recent_computations': log_result.recent_computations,
                    'failure_rate': failure_rate,
                    'avg_computation_time_ms': float(log_result.avg_computation_time) if log_result.avg_computation_time else 0.0,
                    'issues': log_issues
                }
                report['checks']['computation_log'] = log_check
                
            except Exception as e:
                report['checks']['computation_log'] = {
                    'status': 'error',
                    'message': f"Computation log check failed: {str(e)}"
                }
            
            # Calculate overall status and summary
            for check_name, check_result in report['checks'].items():
                if check_result['status'] == 'error':
                    report['overall_status'] = 'error'
                    report['summary']['errors'] += 1
                elif check_result['status'] == 'warning':
                    if report['overall_status'] != 'error':
                        report['overall_status'] = 'warning'
                    report['summary']['warnings'] += 1
                
                # Count issues
                if 'issues' in check_result:
                    report['summary']['total_issues'] += len(check_result['issues'])
            
            return report
            
    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'error',
            'error': str(e),
            'checks': {},
            'summary': {'total_issues': 1, 'warnings': 0, 'errors': 1}
        }

def format_text_report(report):
    """Format report as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 FormFinder Data Quality Report")
    lines.append("=" * 60)
    lines.append(f"Timestamp: {report['timestamp']}")
    lines.append(f"Overall Status: {report['overall_status'].upper()}")
    lines.append("")
    
    # Summary
    lines.append("📋 Summary:")
    lines.append(f"  Total Issues: {report['summary']['total_issues']}")
    lines.append(f"  Warnings: {report['summary']['warnings']}")
    lines.append(f"  Errors: {report['summary']['errors']}")
    lines.append("")
    
    # Individual checks
    for check_name, check_result in report['checks'].items():
        status_emoji = "✅" if check_result['status'] == 'pass' else "⚠️" if check_result['status'] == 'warning' else "❌"
        lines.append(f"{status_emoji} {check_name.replace('_', ' ').title()}: {check_result['status'].upper()}")
        
        if 'message' in check_result:
            lines.append(f"  Error: {check_result['message']}")
        elif 'issues' in check_result and check_result['issues']:
            lines.append("  Issues:")
            for issue in check_result['issues']:
                lines.append(f"    - {issue}")
        
        # Add key metrics
        if check_name == 'feature_quality':
            lines.append(f"  Features: {check_result.get('total_features', 0)} total, {check_result.get('high_quality_features', 0)} high-quality")
            lines.append(f"  Completeness: {check_result.get('completeness_rate', 0):.1%}")
            lines.append(f"  Avg Quality Score: {check_result.get('avg_quality_score', 0):.3f}")
        elif check_name == 'fixtures_consistency':
            lines.append(f"  Fixtures: {check_result.get('total_fixtures', 0)} total, {check_result.get('finished_fixtures', 0)} finished")
            lines.append(f"  Missing Scores: {check_result.get('missing_scores', 0)} ({check_result.get('missing_score_rate', 0):.1%})")
        elif check_name == 'h2h_cache':
            lines.append(f"  H2H Records: {check_result.get('total_h2h_records', 0)} total")
            lines.append(f"  Stale Records: {check_result.get('stale_records', 0)} ({check_result.get('stale_rate', 0):.1%})")
            lines.append(f"  Empty Records: {check_result.get('empty_h2h_records', 0)} ({check_result.get('empty_rate', 0):.1%})")
        elif check_name == 'computation_log':
            lines.append(f"  Computations: {check_result.get('total_computations', 0)} total, {check_result.get('successful_computations', 0)} successful")
            lines.append(f"  Failure Rate: {check_result.get('failure_rate', 0):.1%}")
            lines.append(f"  Recent: {check_result.get('recent_computations', 0)} in last 24h")
        
        lines.append("")
    
    return "\n".join(lines)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run simplified data quality checks')
    parser.add_argument('--output-format', choices=['json', 'text'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--save-report', type=str,
                       help='Save report to file')
    
    args = parser.parse_args()
    
    print("🚀 Starting data quality checks...")
    report = check_data_quality()
    
    # Format output
    if args.output_format == 'json':
        output = json.dumps(report, indent=2, default=str)
    else:
        output = format_text_report(report)
    
    # Save or print
    if args.save_report:
        with open(args.save_report, 'w') as f:
            f.write(output)
        print(f"📄 Report saved to {args.save_report}")
    else:
        print(output)
    
    # Exit with appropriate code
    if report['overall_status'] == 'error':
        sys.exit(1)
    elif report['overall_status'] == 'warning':
        sys.exit(0)  # Warnings are acceptable
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()