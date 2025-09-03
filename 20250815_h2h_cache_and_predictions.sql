-- Migration: Add H2H cache table and update predictions table for goal-based predictions
-- Created: 2025-08-15

-- Create h2h_cache table for storing Head-to-Head statistics
CREATE TABLE IF NOT EXISTS h2h_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    h2h_data TEXT NOT NULL,
    preview_data TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    UNIQUE(home_team_id, away_team_id)
);

-- Create index for faster H2H lookups
CREATE INDEX IF NOT EXISTS idx_h2h_teams ON h2h_cache(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_h2h_expires ON h2h_cache(expires_at);

-- Create predictions table if it doesn't exist
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_result TEXT,
    confidence REAL,
    predicted_total_goals REAL,
    over_2_5_probability REAL,
    under_2_5_probability REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add new columns for goal-based predictions if they don't exist
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we'll handle this in Python

-- Create index for goal-based prediction queries
CREATE INDEX IF NOT EXISTS idx_predictions_goals ON predictions(predicted_total_goals, over_2_5_probability);