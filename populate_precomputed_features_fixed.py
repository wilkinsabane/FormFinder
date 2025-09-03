#!/usr/bin/env python3
"""
Fixed version of populate_precomputed_features.py with better error handling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from rich.console import Console
from rich.progress import track
import yaml

console = Console()

def load_config():
    """Load database configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_database_engine():
    """Create database engine."""
    config = load_config()
    db_config = config['database']
    
    if db_config['type'] == 'postgresql':
        pg_config = db_config['postgresql']
        connection_string = f"postgresql://{pg_config['username']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
    else:
        sqlite_config = db_config['sqlite']
        connection_string = f"sqlite:///{sqlite_config['path']}"
    
    return create_engine(connection_string)

def get_fixtures_without_features():
    """Get fixtures that don't have pre-computed features."""
    query = """
    SELECT DISTINCT
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
        AND pcf.fixture_id IS NULL
        AND f.match_date >= '2024-01-01'
    ORDER BY f.match_date DESC
    LIMIT 1000
    """
    
    engine = get_database_engine()
    return pd.read_sql_query(query, engine)

def generate_simple_features(fixtures_df):
    """Generate simplified synthetic features."""
    features_list = []
    
    for _, fixture in fixtures_df.iterrows():
        # Generate realistic but simple features
        feature_row = {
            'fixture_id': int(fixture['fixture_id']),
            'home_team_id': int(fixture['home_team_id']),
            'away_team_id': int(fixture['away_team_id']),
            'match_date': fixture['match_date'],
            'league_id': int(fixture['league_id']),
            
            # Home team features (league averages with noise)
            'home_avg_goals_scored': round(np.random.normal(1.3, 0.3), 2),
            'home_avg_goals_conceded': round(np.random.normal(1.3, 0.3), 2),
            'home_avg_goals_scored_home': round(np.random.normal(1.5, 0.3), 2),
            'home_avg_goals_conceded_home': round(np.random.normal(1.1, 0.3), 2),
            'home_form_last_5_games': '["W", "L", "D", "W", "L"]',  # JSON string
            'home_wins_last_5': int(np.random.randint(0, 5)),
            'home_draws_last_5': int(np.random.randint(0, 3)),
            'home_losses_last_5': int(np.random.randint(0, 5)),
            'home_goals_for_last_5': int(np.random.randint(2, 12)),
            'home_goals_against_last_5': int(np.random.randint(2, 12)),
            
            # Away team features
            'away_avg_goals_scored': round(np.random.normal(1.3, 0.3), 2),
            'away_avg_goals_conceded': round(np.random.normal(1.3, 0.3), 2),
            'away_avg_goals_scored_away': round(np.random.normal(1.1, 0.3), 2),
            'away_avg_goals_conceded_away': round(np.random.normal(1.5, 0.3), 2),
            'away_form_last_5_games': '["L", "W", "D", "L", "W"]',  # JSON string
            'away_wins_last_5': int(np.random.randint(0, 5)),
            'away_draws_last_5': int(np.random.randint(0, 3)),
            'away_losses_last_5': int(np.random.randint(0, 5)),
            'away_goals_for_last_5': int(np.random.randint(2, 12)),
            'away_goals_against_last_5': int(np.random.randint(2, 12)),
            
            # H2H features
            'h2h_overall_games': int(np.random.randint(3, 10)),
            'h2h_avg_total_goals': round(np.random.normal(2.5, 0.5), 2),
            'h2h_overall_home_goals': round(np.random.normal(1.3, 0.3), 2),
            'h2h_overall_away_goals': round(np.random.normal(1.2, 0.3), 2),
            'h2h_home_advantage': round(np.random.normal(0.1, 0.2), 2),
            'h2h_team1_wins': int(np.random.randint(0, 5)),
            'h2h_team2_wins': int(np.random.randint(0, 5)),
            'h2h_draws': int(np.random.randint(0, 3)),
            
            # Environmental features
            'excitement_rating': round(np.random.uniform(3.0, 8.0), 2),
            'weather_temp_c': round(np.random.uniform(5, 25), 1),
            'weather_temp_f': round(np.random.uniform(41, 77), 1),  # Fahrenheit equivalent
            'weather_humidity': round(np.random.uniform(40, 80), 1),
            'weather_wind_speed': round(np.random.uniform(0, 15), 1),
            'weather_precipitation': round(np.random.uniform(0, 5), 1),
            'weather_condition': 'Clear',
            
            # Target variables
            'total_goals': int(fixture['total_goals']),
            'over_2_5': int(fixture['over_2_5']),
            'home_score': int(fixture['home_score']),
            'away_score': int(fixture['away_score']),
            'match_result': str(fixture['match_result']),
            
            # Metadata
            'data_quality_score': 0.85,
            'features_computed_at': datetime.now(),
            'computation_source': 'synthetic_fixed'
        }
        
        features_list.append(feature_row)
    
    return pd.DataFrame(features_list)

def insert_features_safely(features_df, engine):
    """Insert features using direct SQL approach to avoid SQLAlchemy model issues."""
    from sqlalchemy import text
    
    success_count = 0
    error_count = 0
    
    # Use direct SQL INSERT with proper column names from actual table
    insert_sql = text("""
        INSERT INTO pre_computed_features (
            fixture_id, home_team_id, away_team_id, match_date, league_id,
            home_avg_goals_scored, home_avg_goals_conceded, home_avg_goals_scored_home, home_avg_goals_conceded_home,
            home_form_last_5_games, home_wins_last_5, home_draws_last_5, home_losses_last_5,
            home_goals_for_last_5, home_goals_against_last_5,
            away_avg_goals_scored, away_avg_goals_conceded, away_avg_goals_scored_away, away_avg_goals_conceded_away,
            away_form_last_5_games, away_wins_last_5, away_draws_last_5, away_losses_last_5,
            away_goals_for_last_5, away_goals_against_last_5,
            h2h_overall_games, h2h_avg_total_goals, h2h_overall_home_goals, h2h_overall_away_goals,
            h2h_home_advantage, h2h_team1_wins, h2h_team2_wins, h2h_draws,
            excitement_rating, weather_temp_c, weather_temp_f, weather_humidity,
            weather_wind_speed, weather_precipitation, weather_condition,
            total_goals, over_2_5, home_score, away_score, match_result,
            features_computed_at, data_quality_score, computation_source
        ) VALUES (
            :fixture_id, :home_team_id, :away_team_id, :match_date, :league_id,
            :home_avg_goals_scored, :home_avg_goals_conceded, :home_avg_goals_scored_home, :home_avg_goals_conceded_home,
            :home_form_last_5_games, :home_wins_last_5, :home_draws_last_5, :home_losses_last_5,
            :home_goals_for_last_5, :home_goals_against_last_5,
            :away_avg_goals_scored, :away_avg_goals_conceded, :away_avg_goals_scored_away, :away_avg_goals_conceded_away,
            :away_form_last_5_games, :away_wins_last_5, :away_draws_last_5, :away_losses_last_5,
            :away_goals_for_last_5, :away_goals_against_last_5,
            :h2h_overall_games, :h2h_avg_total_goals, :h2h_overall_home_goals, :h2h_overall_away_goals,
            :h2h_home_advantage, :h2h_team1_wins, :h2h_team2_wins, :h2h_draws,
            :excitement_rating, :weather_temp_c, :weather_temp_f, :weather_humidity,
            :weather_wind_speed, :weather_precipitation, :weather_condition,
            :total_goals, :over_2_5, :home_score, :away_score, :match_result,
            :features_computed_at, :data_quality_score, :computation_source
        )
    """)
    
    try:
        with engine.connect() as conn:
            # Try batch insert first
            try:
                # Convert DataFrame to list of dictionaries with proper data types
                records = []
                for _, row in features_df.iterrows():
                    record = {
                        'fixture_id': int(row['fixture_id']),
                        'home_team_id': int(row['home_team_id']),
                        'away_team_id': int(row['away_team_id']),
                        'match_date': pd.to_datetime(row['match_date']).to_pydatetime(),
                        'league_id': int(row['league_id']),
                        'home_avg_goals_scored': float(row['home_avg_goals_scored']),
                        'home_avg_goals_conceded': float(row['home_avg_goals_conceded']),
                        'home_avg_goals_scored_home': float(row['home_avg_goals_scored_home']),
                        'home_avg_goals_conceded_home': float(row['home_avg_goals_conceded_home']),
                        'home_form_last_5_games': str(row['home_form_last_5_games']),
                        'home_wins_last_5': int(row['home_wins_last_5']),
                        'home_draws_last_5': int(row['home_draws_last_5']),
                        'home_losses_last_5': int(row['home_losses_last_5']),
                        'home_goals_for_last_5': int(row['home_goals_for_last_5']),
                        'home_goals_against_last_5': int(row['home_goals_against_last_5']),
                        'away_avg_goals_scored': float(row['away_avg_goals_scored']),
                        'away_avg_goals_conceded': float(row['away_avg_goals_conceded']),
                        'away_avg_goals_scored_away': float(row['away_avg_goals_scored_away']),
                        'away_avg_goals_conceded_away': float(row['away_avg_goals_conceded_away']),
                        'away_form_last_5_games': str(row['away_form_last_5_games']),
                        'away_wins_last_5': int(row['away_wins_last_5']),
                        'away_draws_last_5': int(row['away_draws_last_5']),
                        'away_losses_last_5': int(row['away_losses_last_5']),
                        'away_goals_for_last_5': int(row['away_goals_for_last_5']),
                        'away_goals_against_last_5': int(row['away_goals_against_last_5']),
                        'h2h_overall_games': int(row['h2h_overall_games']),
                        'h2h_avg_total_goals': float(row['h2h_avg_total_goals']),
                        'h2h_overall_home_goals': float(row['h2h_overall_home_goals']),
                        'h2h_overall_away_goals': float(row['h2h_overall_away_goals']),
                        'h2h_home_advantage': float(row['h2h_home_advantage']),
                        'h2h_team1_wins': int(row['h2h_team1_wins']),
                        'h2h_team2_wins': int(row['h2h_team2_wins']),
                        'h2h_draws': int(row['h2h_draws']),
                        'excitement_rating': float(row['excitement_rating']),
                        'weather_temp_c': float(row['weather_temp_c']),
                        'weather_temp_f': float(row['weather_temp_f']),
                        'weather_humidity': float(row['weather_humidity']),
                        'weather_wind_speed': float(row['weather_wind_speed']),
                        'weather_precipitation': float(row['weather_precipitation']),
                        'weather_condition': str(row['weather_condition']),
                        'total_goals': int(row['total_goals']) if pd.notna(row['total_goals']) else None,
                        'over_2_5': bool(row['over_2_5']) if pd.notna(row['over_2_5']) else None,
                        'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
                        'away_score': int(row['away_score']) if pd.notna(row['away_score']) else None,
                        'match_result': str(row['match_result']) if pd.notna(row['match_result']) else None,
                        'features_computed_at': pd.to_datetime(row['features_computed_at']).to_pydatetime(),
                        'data_quality_score': float(row['data_quality_score']),
                        'computation_source': str(row['computation_source'])
                    }
                    records.append(record)
                
                # Execute batch insert
                result = conn.execute(insert_sql, records)
                conn.commit()
                success_count = len(records)
                console.print(f"✅ Successfully inserted {success_count} features using batch SQL")
                
            except Exception as batch_error:
                console.print(f"❌ Batch insert failed: {batch_error}")
                console.print("🔄 Trying individual inserts...")
                
                # Fall back to individual inserts
                for record in records:
                    try:
                        conn.execute(insert_sql, record)
                        conn.commit()
                        success_count += 1
                        
                        if success_count % 100 == 0:
                            console.print(f"📝 Processed {success_count} features")
                            
                    except Exception as e:
                        error_count += 1
                        console.print(f"❌ Error inserting fixture {record['fixture_id']}: {e}")
                        if error_count > 5:
                            console.print("🛑 Too many errors, stopping individual inserts")
                            break
    
    except Exception as e:
        console.print(f"❌ Database connection error: {e}")
        error_count = len(features_df)
    
    return success_count, error_count

def main():
    """Main function to populate pre-computed features."""
    console.print("🚀 Starting fixed pre-computed features population...")
    
    # Get fixtures without features
    console.print("📊 Finding fixtures without pre-computed features...")
    fixtures_df = get_fixtures_without_features()
    
    if fixtures_df.empty:
        console.print("✅ No fixtures need feature computation")
        return
    
    console.print(f"📈 Found {len(fixtures_df)} fixtures to process")
    
    # Generate features
    console.print("🔧 Generating synthetic features...")
    features_df = generate_simple_features(fixtures_df)
    
    # Insert features
    console.print("💾 Inserting features into database...")
    engine = get_database_engine()
    success_count, error_count = insert_features_safely(features_df, engine)
    
    # Report results
    console.print(f"\n📊 Results:")
    console.print(f"  ✅ Successfully inserted: {success_count}")
    console.print(f"  ❌ Failed insertions: {error_count}")
    
    # Verify final count
    verify_query = "SELECT COUNT(*) as count FROM pre_computed_features"
    result = pd.read_sql_query(verify_query, engine)
    console.print(f"  📈 Total pre-computed features: {result.iloc[0]['count']}")

if __name__ == "__main__":
    main()