#!/usr/bin/env python3
"""
Script to check for null values in pre_computed_features table
"""

import sqlite3
from formfinder.config import load_config

def check_null_values():
    """Check for null values in pre_computed_features table"""
    load_config()
    
    conn = sqlite3.connect('formfinder.db')
    cursor = conn.cursor()
    
    # Get total row count
    cursor.execute('SELECT COUNT(*) as total_rows FROM pre_computed_features')
    total = cursor.fetchone()[0]
    print(f'Total rows in pre_computed_features: {total}')
    
    if total == 0:
        print('No data in pre_computed_features table')
        conn.close()
        return
    
    # Get column names
    cursor.execute('PRAGMA table_info(pre_computed_features)')
    columns = [row[1] for row in cursor.fetchall()]
    
    print(f'\nChecking {len(columns)} columns for null values...')
    
    null_counts = {}
    for col in columns:
        cursor.execute(f'SELECT COUNT(*) FROM pre_computed_features WHERE "{col}" IS NULL')
        null_counts[col] = cursor.fetchone()[0]
    
    print('\nColumns with null values:')
    has_nulls = False
    for col, count in null_counts.items():
        if count > 0:
            has_nulls = True
            percentage = (count / total) * 100
            print(f'  {col}: {count}/{total} ({percentage:.1f}% null)')
    
    if not has_nulls:
        print('  No null values found in any column!')
    
    # Check specific columns mentioned by user
    target_columns = [
        'home_attack_strength', 'home_defense_strength', 
        'away_attack_strength', 'away_defense_strength',
        'home_form_diff', 'away_form_diff',
        'home_team_form_score', 'away_team_form_score'
    ]
    
    print('\nSpecific columns of interest:')
    for col in target_columns:
        if col in null_counts:
            count = null_counts[col]
            percentage = (count / total) * 100
            print(f'  {col}: {count}/{total} ({percentage:.1f}% null)')
        else:
            print(f'  {col}: Column not found in table')
    
    conn.close()

if __name__ == '__main__':
    check_null_values()