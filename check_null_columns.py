#!/usr/bin/env python3
"""
Script to check null values in specific pre_computed_features columns
"""

import sqlite3
from formfinder.database import get_db_session
from formfinder.config import load_config

def check_null_columns():
    """Check null values in specific columns of pre_computed_features table"""
    # Load configuration
    load_config()
    
    with get_db_session() as session:
        # Get raw connection for direct SQL queries
        conn = session.bind.raw_connection()
        cursor = conn.cursor()
        
        # Get total row count
        cursor.execute('SELECT COUNT(*) as total_rows FROM pre_computed_features')
        total = cursor.fetchone()[0]
        print(f'Total rows: {total}')
        print('\nNull value analysis:')
        print('-' * 60)
        
        # Columns to check
        columns = [
            'home_xg', 'away_xg',
            'home_team_strength', 'away_team_strength',
            'home_team_momentum', 'away_team_momentum',
            'home_form_last_5_games',
            'home_team_sentiment', 'away_team_sentiment',
            'home_team_markov_momentum', 'away_team_markov_momentum',
            'home_team_state_stability', 'away_team_state_stability',
            'home_team_transition_entropy', 'away_team_transition_entropy',
            'home_team_performance_volatility', 'away_team_performance_volatility',
            'home_team_current_state', 'away_team_current_state',
            'home_team_state_duration', 'away_team_state_duration',
            'home_team_expected_next_state', 'away_team_expected_next_state',
            'home_team_state_confidence', 'away_team_state_confidence',
            'markov_match_prediction_confidence', 'markov_outcome_probabilities'
        ]
        
        for col in columns:
            cursor.execute(f'SELECT COUNT(*) FROM pre_computed_features WHERE {col} IS NULL')
            null_count = cursor.fetchone()[0]
            null_rate = (null_count / total) * 100 if total > 0 else 0
            print(f'{col:35}: {null_count:4}/{total} ({null_rate:5.1f}% null)')
        
        # Sample some records to see actual values
        print('\nSample records (first 3):')
        print('-' * 60)
        cursor.execute('SELECT fixture_id, home_xg, away_xg, home_team_strength, home_team_sentiment, home_team_markov_momentum FROM pre_computed_features LIMIT 3')
        for row in cursor.fetchall():
            print(f'Fixture {row[0]}: xG={row[1]},{row[2]} | Strength={row[3]} | Sentiment={row[4]} | Markov={row[5]}')
        
        cursor.close()

if __name__ == '__main__':
    check_null_columns()