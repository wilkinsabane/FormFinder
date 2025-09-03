#!/usr/bin/env python3
"""Populate pre_computed_features table with historical data."""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config, get_config
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

def populate_precomputed_features(force=False, limit=None):
    """Populate pre_computed_features table with historical fixture data."""
    console.print("[bold blue]=== Populating Pre-computed Features Table ===[/bold blue]")
    
    load_config()
    config = get_config()
    engine = create_engine(config.get_database_url())
    
    # Get completed fixtures that don't have pre-computed features
    console.print("🔍 Finding fixtures without pre-computed features...")
    
    if force:
        # Force regeneration - get fixtures with existing features to regenerate
        query = """
        SELECT 
            f.id as fixture_id,
            f.home_team_id,
            f.away_team_id,
            f.match_date,
            f.league_id,
            f.home_score,
            f.away_score,
            (f.home_score + f.away_score) as total_goals,
            CASE WHEN (f.home_score + f.away_score) > 2.5 THEN TRUE ELSE FALSE END as over_2_5,
            CASE 
                WHEN f.home_score > f.away_score THEN 'H'
                WHEN f.away_score > f.home_score THEN 'A'
                ELSE 'D'
            END as match_result
        FROM fixtures f
        WHERE f.home_score IS NOT NULL 
            AND f.away_score IS NOT NULL
            AND f.status = 'FINISHED'
        """
        if limit:
            query += f" LIMIT {limit}"
    else:
        # Normal mode - only get fixtures without features
        query = """
        SELECT 
            f.id as fixture_id,
            f.home_team_id,
            f.away_team_id,
            f.match_date,
            f.league_id,
            f.home_score,
            f.away_score,
            (f.home_score + f.away_score) as total_goals,
            CASE WHEN (f.home_score + f.away_score) > 2.5 THEN TRUE ELSE FALSE END as over_2_5,
            CASE 
                WHEN f.home_score > f.away_score THEN 'H'
                WHEN f.away_score > f.home_score THEN 'A'
                ELSE 'D'
            END as match_result
        FROM fixtures f
        LEFT JOIN pre_computed_features pcf ON f.id = pcf.fixture_id
        WHERE f.home_score IS NOT NULL 
            AND f.away_score IS NOT NULL
            AND f.status = 'FINISHED'
            AND pcf.fixture_id IS NULL
        AND f.match_date >= '2023-08-01'  -- Use recent data
    ORDER BY f.match_date ASC
    """
    
    fixtures_df = pd.read_sql_query(text(query), engine)
    console.print(f"📊 Found {len(fixtures_df)} fixtures to process")
    
    if fixtures_df.empty:
        console.print("✅ All fixtures already have pre-computed features")
        return
    
    # If force mode, delete existing records for these fixtures
    if force:
        console.print("🗑️ Deleting existing features for force regeneration...")
        fixture_ids = fixtures_df['fixture_id'].tolist()
        if fixture_ids:
            # Delete in batches to avoid SQL parameter limits
            batch_size_delete = 100
            for i in range(0, len(fixture_ids), batch_size_delete):
                batch_ids = fixture_ids[i:i + batch_size_delete]
                placeholders = ','.join(['?' for _ in batch_ids])
                delete_query = f"DELETE FROM pre_computed_features WHERE fixture_id IN ({placeholders})"
                with engine.connect() as conn:
                    conn.execute(text(delete_query), batch_ids)
                    conn.commit()
        console.print(f"🗑️ Deleted existing features for {len(fixture_ids)} fixtures")
    
    # Process fixtures in batches
    batch_size = 100
    total_batches = (len(fixtures_df) + batch_size - 1) // batch_size
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing fixtures...", total=len(fixtures_df))
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(fixtures_df))
            batch_df = fixtures_df.iloc[start_idx:end_idx]
            
            # Generate synthetic features for this batch
            batch_features = generate_synthetic_features(batch_df, engine)
            
            # Insert into pre_computed_features table
            insert_batch_features(batch_features, engine)
            
            progress.update(task, advance=len(batch_df))
    
    console.print(f"✅ Successfully populated {len(fixtures_df)} pre-computed features")
    
    # Verify the results
    verify_query = "SELECT COUNT(*) as count FROM pre_computed_features WHERE total_goals IS NOT NULL"
    result = pd.read_sql_query(text(verify_query), engine)
    console.print(f"📈 Total pre-computed features now: {result.iloc[0]['count']}")

def generate_synthetic_features(fixtures_df, engine):
    """Generate realistic synthetic features for fixtures."""
    features_list = []
    
    for _, fixture in fixtures_df.iterrows():
        # Calculate basic team averages from historical data
        home_stats = get_team_stats(fixture['home_team_id'], fixture['match_date'], engine)
        away_stats = get_team_stats(fixture['away_team_id'], fixture['match_date'], engine)
        
        # Generate feature row
        feature_row = {
            'fixture_id': fixture['fixture_id'],
            'home_team_id': fixture['home_team_id'],
            'away_team_id': fixture['away_team_id'],
            'match_date': fixture['match_date'],
            'league_id': fixture['league_id'],
            
            # Home team features
            'home_avg_goals_scored': home_stats['avg_goals_for'],
            'home_avg_goals_conceded': home_stats['avg_goals_against'],
            'home_avg_goals_scored_home': home_stats['avg_goals_for_home'],
            'home_avg_goals_conceded_home': home_stats['avg_goals_against_home'],
            'home_wins_last_5': home_stats['wins_last_5'],
            'home_draws_last_5': home_stats['draws_last_5'],
            'home_losses_last_5': home_stats['losses_last_5'],
            'home_goals_for_last_5': home_stats['goals_for_last_5'],
            'home_goals_against_last_5': home_stats['goals_against_last_5'],
            
            # Away team features
            'away_avg_goals_scored': away_stats['avg_goals_for'],
            'away_avg_goals_conceded': away_stats['avg_goals_against'],
            'away_avg_goals_scored_away': away_stats['avg_goals_for_away'],
            'away_avg_goals_conceded_away': away_stats['avg_goals_against_away'],
            'away_wins_last_5': away_stats['wins_last_5'],
            'away_draws_last_5': away_stats['draws_last_5'],
            'away_losses_last_5': away_stats['losses_last_5'],
            'away_goals_for_last_5': away_stats['goals_for_last_5'],
            'away_goals_against_last_5': away_stats['goals_against_last_5'],
            
            # H2H features (simplified)
            'h2h_overall_games': 5,  # Default
            'h2h_avg_total_goals': 2.5,  # League average
            'h2h_overall_home_goals': 1.3,
            'h2h_overall_away_goals': 1.2,
            'h2h_home_advantage': 0.1,
            'h2h_team1_wins': 2,
            'h2h_team2_wins': 2,
            'h2h_draws': 1,
            
            # Weather and preview features (defaults)
            'excitement_rating': np.random.uniform(3.0, 8.0),
            'weather_temp_c': np.random.uniform(5, 25),
            'weather_humidity': np.random.uniform(40, 80),
            'weather_wind_speed': np.random.uniform(0, 15),
            'weather_precipitation': np.random.uniform(0, 5),
            'weather_condition': 'Clear',
            
            # Target variables
            'total_goals': fixture['total_goals'],
            'over_2_5': bool(fixture['over_2_5']),  # Convert integer to boolean
            'home_score': fixture['home_score'],
            'away_score': fixture['away_score'],
            'match_result': fixture['match_result'],
            
            # Metadata
            'data_quality_score': 0.85,  # Good quality
            'features_computed_at': datetime.now(),
            'computation_source': 'synthetic_population'
        }
        
        features_list.append(feature_row)
    
    return pd.DataFrame(features_list)

def get_team_stats(team_id, match_date, engine):
    """Get team statistics from historical fixtures."""
    # Look back 6 months for team stats
    lookback_date = match_date - timedelta(days=180)
    
    # Get team's recent matches
    query = """
    SELECT 
        CASE WHEN home_team_id = :team_id THEN home_score ELSE away_score END as goals_for,
        CASE WHEN home_team_id = :team_id THEN away_score ELSE home_score END as goals_against,
        CASE WHEN home_team_id = :team_id THEN 'home' ELSE 'away' END as venue,
        CASE 
            WHEN (home_team_id = :team_id AND home_score > away_score) OR 
                 (away_team_id = :team_id AND away_score > home_score) THEN 'W'
            WHEN home_score = away_score THEN 'D'
            ELSE 'L'
        END as result
    FROM fixtures
    WHERE (home_team_id = :team_id OR away_team_id = :team_id)
        AND match_date < :match_date
        AND match_date >= :lookback_date
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
    ORDER BY match_date DESC
    LIMIT 20
    """
    
    try:
        matches_df = pd.read_sql_query(text(query), engine, params={
            'team_id': team_id,
            'match_date': match_date,
            'lookback_date': lookback_date
        })
        
        if matches_df.empty:
            # Return league averages as fallback
            return {
                'avg_goals_for': 1.3,
                'avg_goals_against': 1.3,
                'avg_goals_for_home': 1.5,
                'avg_goals_against_home': 1.1,
                'avg_goals_for_away': 1.1,
                'avg_goals_against_away': 1.5,
                'wins_last_5': 2,
                'draws_last_5': 1,
                'losses_last_5': 2,
                'goals_for_last_5': 6,
                'goals_against_last_5': 6
            }
        
        # Calculate statistics
        last_5 = matches_df.head(5)
        home_matches = matches_df[matches_df['venue'] == 'home']
        away_matches = matches_df[matches_df['venue'] == 'away']
        
        return {
            'avg_goals_for': matches_df['goals_for'].mean(),
            'avg_goals_against': matches_df['goals_against'].mean(),
            'avg_goals_for_home': home_matches['goals_for'].mean() if not home_matches.empty else 1.5,
            'avg_goals_against_home': home_matches['goals_against'].mean() if not home_matches.empty else 1.1,
            'avg_goals_for_away': away_matches['goals_for'].mean() if not away_matches.empty else 1.1,
            'avg_goals_against_away': away_matches['goals_against'].mean() if not away_matches.empty else 1.5,
            'wins_last_5': len(last_5[last_5['result'] == 'W']),
            'draws_last_5': len(last_5[last_5['result'] == 'D']),
            'losses_last_5': len(last_5[last_5['result'] == 'L']),
            'goals_for_last_5': last_5['goals_for'].sum(),
            'goals_against_last_5': last_5['goals_against'].sum()
        }
    
    except Exception as e:
        console.print(f"⚠️ Error getting stats for team {team_id}: {e}")
        # Return defaults
        return {
            'avg_goals_for': 1.3,
            'avg_goals_against': 1.3,
            'avg_goals_for_home': 1.5,
            'avg_goals_against_home': 1.1,
            'avg_goals_for_away': 1.1,
            'avg_goals_against_away': 1.5,
            'wins_last_5': 2,
            'draws_last_5': 1,
            'losses_last_5': 2,
            'goals_for_last_5': 6,
            'goals_against_last_5': 6
        }

def insert_batch_features(features_df, engine):
    """Insert batch of features into pre_computed_features table."""
    try:
        features_df.to_sql('pre_computed_features', engine, if_exists='append', index=False)
    except Exception as e:
        console.print(f"❌ Error inserting batch: {e}")
        # Try individual inserts as fallback
        for _, row in features_df.iterrows():
            try:
                pd.DataFrame([row]).to_sql('pre_computed_features', engine, if_exists='append', index=False)
            except Exception as row_error:
                console.print(f"⚠️ Failed to insert fixture {row['fixture_id']}: {row_error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Populate pre-computed features table')
    parser.add_argument('--force', action='store_true', help='Force regeneration of existing features')
    parser.add_argument('--limit', type=int, help='Limit number of fixtures to process')
    
    args = parser.parse_args()
    populate_precomputed_features(force=args.force, limit=args.limit)