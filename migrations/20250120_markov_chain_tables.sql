-- Migration: Add Markov Chain Tables for Enhanced Prediction
-- Date: 2025-01-20
-- Description: Creates new tables for Markov chain analysis including team performance states,
--              transition matrices, and computed features for improved prediction accuracy.

-- Create TeamPerformanceState table
CREATE TABLE IF NOT EXISTS team_performance_states (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    league_id INTEGER NOT NULL REFERENCES leagues(league_pk) ON DELETE CASCADE,
    fixture_id INTEGER REFERENCES fixtures(id) ON DELETE SET NULL,
    state_date TIMESTAMP NOT NULL,
    
    -- Performance state classification
    performance_state VARCHAR(20) NOT NULL CHECK (performance_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
    state_score FLOAT NOT NULL,
    
    -- Metrics used for state calculation
    goals_scored FLOAT,
    goals_conceded FLOAT,
    goal_difference FLOAT,
    win_rate FLOAT,
    points_per_game FLOAT,
    form_streak VARCHAR(10),
    
    -- Context information
    matches_considered INTEGER DEFAULT 5,
    home_away_context VARCHAR(10) CHECK (home_away_context IN ('home', 'away', 'overall')),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for TeamPerformanceState
CREATE INDEX IF NOT EXISTS ix_performance_state_team ON team_performance_states(team_id);
CREATE INDEX IF NOT EXISTS ix_performance_state_league ON team_performance_states(league_id);
CREATE INDEX IF NOT EXISTS ix_performance_state_date ON team_performance_states(state_date);
CREATE INDEX IF NOT EXISTS ix_performance_state_fixture ON team_performance_states(fixture_id);
CREATE INDEX IF NOT EXISTS ix_performance_state_context ON team_performance_states(home_away_context);

-- Create MarkovTransitionMatrix table
CREATE TABLE IF NOT EXISTS markov_transition_matrices (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    league_id INTEGER NOT NULL REFERENCES leagues(league_pk) ON DELETE CASCADE,
    
    -- Transition definition
    from_state VARCHAR(20) NOT NULL CHECK (from_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
    to_state VARCHAR(20) NOT NULL CHECK (to_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
    
    -- Transition statistics
    transition_count INTEGER DEFAULT 0,
    transition_probability FLOAT DEFAULT 0.0,
    smoothed_probability FLOAT DEFAULT 0.0,
    
    -- Context and metadata
    home_away_context VARCHAR(10) DEFAULT 'overall' CHECK (home_away_context IN ('home', 'away', 'overall')),
    calculation_date TIMESTAMP NOT NULL,
    data_window_start TIMESTAMP,
    data_window_end TIMESTAMP,
    total_transitions INTEGER DEFAULT 0,
    
    -- Smoothing parameters
    smoothing_alpha FLOAT DEFAULT 1.0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique transition matrices per team/league/context
    CONSTRAINT uq_transition_matrix UNIQUE (team_id, league_id, from_state, to_state, home_away_context)
);

-- Create indexes for MarkovTransitionMatrix
CREATE INDEX IF NOT EXISTS ix_transition_team ON markov_transition_matrices(team_id);
CREATE INDEX IF NOT EXISTS ix_transition_league ON markov_transition_matrices(league_id);
CREATE INDEX IF NOT EXISTS ix_transition_states ON markov_transition_matrices(from_state, to_state);
CREATE INDEX IF NOT EXISTS ix_transition_context ON markov_transition_matrices(home_away_context);
CREATE INDEX IF NOT EXISTS ix_transition_date ON markov_transition_matrices(calculation_date);

-- Create MarkovFeatures table
CREATE TABLE IF NOT EXISTS markov_features (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    league_id INTEGER NOT NULL REFERENCES leagues(league_pk) ON DELETE CASCADE,
    fixture_id INTEGER REFERENCES fixtures(id) ON DELETE SET NULL,
    feature_date TIMESTAMP NOT NULL,
    
    -- Current state information
    current_state VARCHAR(20) NOT NULL CHECK (current_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
    state_duration INTEGER DEFAULT 1,
    
    -- Momentum and trend features
    momentum_score FLOAT DEFAULT 0.0,
    trend_direction VARCHAR(10) CHECK (trend_direction IN ('improving', 'declining', 'stable')),
    
    -- Stability and volatility features
    state_stability FLOAT DEFAULT 0.0,
    transition_entropy FLOAT DEFAULT 0.0,
    performance_volatility FLOAT DEFAULT 0.0,
    
    -- Prediction features
    expected_next_state VARCHAR(20) CHECK (expected_next_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
    next_state_probability FLOAT DEFAULT 0.0,
    state_confidence FLOAT DEFAULT 0.0,
    
    -- Advanced features
    mean_return_time FLOAT,
    steady_state_probability FLOAT,
    absorption_probability FLOAT,
    
    -- Context and metadata
    home_away_context VARCHAR(10) DEFAULT 'overall' CHECK (home_away_context IN ('home', 'away', 'overall')),
    lookback_window INTEGER DEFAULT 10,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for MarkovFeatures
CREATE INDEX IF NOT EXISTS ix_markov_features_team ON markov_features(team_id);
CREATE INDEX IF NOT EXISTS ix_markov_features_league ON markov_features(league_id);
CREATE INDEX IF NOT EXISTS ix_markov_features_fixture ON markov_features(fixture_id);
CREATE INDEX IF NOT EXISTS ix_markov_features_date ON markov_features(feature_date);
CREATE INDEX IF NOT EXISTS ix_markov_features_state ON markov_features(current_state);
CREATE INDEX IF NOT EXISTS ix_markov_features_context ON markov_features(home_away_context);

-- Add Markov features to pre_computed_features table
ALTER TABLE pre_computed_features 
ADD COLUMN IF NOT EXISTS home_team_markov_momentum FLOAT,
ADD COLUMN IF NOT EXISTS away_team_markov_momentum FLOAT,
ADD COLUMN IF NOT EXISTS home_team_state_stability FLOAT,
ADD COLUMN IF NOT EXISTS away_team_state_stability FLOAT,
ADD COLUMN IF NOT EXISTS home_team_transition_entropy FLOAT,
ADD COLUMN IF NOT EXISTS away_team_transition_entropy FLOAT,
ADD COLUMN IF NOT EXISTS home_team_performance_volatility FLOAT,
ADD COLUMN IF NOT EXISTS away_team_performance_volatility FLOAT,
ADD COLUMN IF NOT EXISTS home_team_current_state VARCHAR(20),
ADD COLUMN IF NOT EXISTS away_team_current_state VARCHAR(20),
ADD COLUMN IF NOT EXISTS home_team_state_duration INTEGER,
ADD COLUMN IF NOT EXISTS away_team_state_duration INTEGER,
ADD COLUMN IF NOT EXISTS home_team_expected_next_state VARCHAR(20),
ADD COLUMN IF NOT EXISTS away_team_expected_next_state VARCHAR(20),
ADD COLUMN IF NOT EXISTS home_team_state_confidence FLOAT,
ADD COLUMN IF NOT EXISTS away_team_state_confidence FLOAT,
ADD COLUMN IF NOT EXISTS markov_match_prediction_confidence FLOAT,
ADD COLUMN IF NOT EXISTS markov_outcome_probabilities TEXT;

-- Add check constraints for new columns
ALTER TABLE pre_computed_features 
ADD CONSTRAINT IF NOT EXISTS chk_home_team_current_state 
    CHECK (home_team_current_state IS NULL OR home_team_current_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
ADD CONSTRAINT IF NOT EXISTS chk_away_team_current_state 
    CHECK (away_team_current_state IS NULL OR away_team_current_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
ADD CONSTRAINT IF NOT EXISTS chk_home_team_expected_next_state 
    CHECK (home_team_expected_next_state IS NULL OR home_team_expected_next_state IN ('excellent', 'good', 'average', 'poor', 'terrible')),
ADD CONSTRAINT IF NOT EXISTS chk_away_team_expected_next_state 
    CHECK (away_team_expected_next_state IS NULL OR away_team_expected_next_state IN ('excellent', 'good', 'average', 'poor', 'terrible'));

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to new tables
CREATE TRIGGER IF NOT EXISTS update_team_performance_states_updated_at 
    BEFORE UPDATE ON team_performance_states 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER IF NOT EXISTS update_markov_transition_matrices_updated_at 
    BEFORE UPDATE ON markov_transition_matrices 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER IF NOT EXISTS update_markov_features_updated_at 
    BEFORE UPDATE ON markov_features 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial configuration for Markov chain processing
INSERT INTO processing_status (process_name, status, details) 
VALUES 
    ('markov_state_classification', 'idle', 'Markov chain state classification process'),
    ('markov_transition_calculation', 'idle', 'Markov chain transition matrix calculation process'),
    ('markov_feature_generation', 'idle', 'Markov chain feature generation process')
ON CONFLICT (process_name) DO NOTHING;

-- Create view for easy access to team current states
CREATE OR REPLACE VIEW team_current_markov_states AS
SELECT DISTINCT ON (tps.team_id, tps.league_id, tps.home_away_context)
    tps.team_id,
    tps.league_id,
    t.name as team_name,
    l.name as league_name,
    tps.performance_state,
    tps.state_score,
    tps.state_date,
    tps.home_away_context,
    tps.form_streak,
    tps.matches_considered
FROM team_performance_states tps
JOIN teams t ON tps.team_id = t.id
JOIN leagues l ON tps.league_id = l.league_pk
ORDER BY tps.team_id, tps.league_id, tps.home_away_context, tps.state_date DESC;

-- Create view for transition matrix summary
CREATE OR REPLACE VIEW markov_transition_summary AS
SELECT 
    mtm.team_id,
    mtm.league_id,
    t.name as team_name,
    l.name as league_name,
    mtm.home_away_context,
    mtm.from_state,
    mtm.to_state,
    mtm.transition_count,
    mtm.transition_probability,
    mtm.smoothed_probability,
    mtm.calculation_date
FROM markov_transition_matrices mtm
JOIN teams t ON mtm.team_id = t.id
JOIN leagues l ON mtm.league_id = l.league_pk
ORDER BY mtm.team_id, mtm.league_id, mtm.home_away_context, mtm.from_state, mtm.to_state;

-- Migration completed successfully
INSERT INTO processing_status (process_name, last_run, status, details) 
VALUES ('markov_migration_20250120', CURRENT_TIMESTAMP, 'success', 'Markov chain tables created successfully')
ON CONFLICT (process_name) DO UPDATE SET 
    last_run = CURRENT_TIMESTAMP,
    status = 'success',
    details = 'Markov chain tables created successfully',
    updated_at = CURRENT_TIMESTAMP;

COMMIT;