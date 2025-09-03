-- Data Collection and Training Separation Migration
-- This migration implements the PRD for separating data collection from training
-- Created: 2025-01-01
-- Purpose: Add pre-computed features tables and enhanced caching

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Pre-computed Features Table
-- This table stores all pre-computed features for training to eliminate API calls during training
CREATE TABLE IF NOT EXISTS pre_computed_features (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    match_date TIMESTAMP NOT NULL,
    league_id INTEGER NOT NULL,
    
    -- Home team form features
    home_avg_goals_scored DECIMAL(4,2) DEFAULT 0.0,
    home_avg_goals_conceded DECIMAL(4,2) DEFAULT 0.0,
    home_avg_goals_scored_home DECIMAL(4,2) DEFAULT 0.0,
    home_avg_goals_conceded_home DECIMAL(4,2) DEFAULT 0.0,
    home_form_last_5_games TEXT, -- JSON array of last 5 results
    home_wins_last_5 INTEGER DEFAULT 0,
    home_draws_last_5 INTEGER DEFAULT 0,
    home_losses_last_5 INTEGER DEFAULT 0,
    home_goals_for_last_5 INTEGER DEFAULT 0,
    home_goals_against_last_5 INTEGER DEFAULT 0,
    
    -- Away team form features
    away_avg_goals_scored DECIMAL(4,2) DEFAULT 0.0,
    away_avg_goals_conceded DECIMAL(4,2) DEFAULT 0.0,
    away_avg_goals_scored_away DECIMAL(4,2) DEFAULT 0.0,
    away_avg_goals_conceded_away DECIMAL(4,2) DEFAULT 0.0,
    away_form_last_5_games TEXT, -- JSON array of last 5 results
    away_wins_last_5 INTEGER DEFAULT 0,
    away_draws_last_5 INTEGER DEFAULT 0,
    away_losses_last_5 INTEGER DEFAULT 0,
    away_goals_for_last_5 INTEGER DEFAULT 0,
    away_goals_against_last_5 INTEGER DEFAULT 0,
    
    -- H2H features
    h2h_overall_games INTEGER DEFAULT 0,
    h2h_avg_total_goals DECIMAL(4,2) DEFAULT 0.0,
    h2h_overall_home_goals DECIMAL(4,2) DEFAULT 0.0,
    h2h_overall_away_goals DECIMAL(4,2) DEFAULT 0.0,
    h2h_home_advantage DECIMAL(4,2) DEFAULT 0.0,
    h2h_team1_wins INTEGER DEFAULT 0,
    h2h_team2_wins INTEGER DEFAULT 0,
    h2h_draws INTEGER DEFAULT 0,
    h2h_last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Preview and weather features
    excitement_rating DECIMAL(4,2) DEFAULT 0.0,
    weather_temp_c DECIMAL(4,1) DEFAULT 21.0,
    weather_temp_f DECIMAL(4,1) DEFAULT 69.8,
    weather_humidity DECIMAL(4,1) DEFAULT 50.0,
    weather_wind_speed DECIMAL(4,1) DEFAULT 5.0,
    weather_precipitation DECIMAL(4,1) DEFAULT 0.0,
    weather_condition VARCHAR(50) DEFAULT 'Clear',
    
    -- Target variables (for completed matches)
    total_goals INTEGER,
    over_2_5 BOOLEAN,
    home_score INTEGER,
    away_score INTEGER,
    match_result VARCHAR(10), -- 'H', 'D', 'A'
    
    -- Metadata
    features_computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    computation_source VARCHAR(20) DEFAULT 'api', -- 'api', 'cache', 'fallback'
    
    CONSTRAINT unique_fixture_features UNIQUE(fixture_id)
);

-- Indexes for pre_computed_features
CREATE INDEX IF NOT EXISTS idx_pcf_match_date ON pre_computed_features (match_date);
CREATE INDEX IF NOT EXISTS idx_pcf_teams ON pre_computed_features (home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_pcf_league_date ON pre_computed_features (league_id, match_date);
CREATE INDEX IF NOT EXISTS idx_pcf_computed_at ON pre_computed_features (features_computed_at);
CREATE INDEX IF NOT EXISTS idx_pcf_quality ON pre_computed_features (data_quality_score);

-- 2. Enhanced H2H Cache Table
-- This enhances the existing h2h_cache with additional statistics and better caching
CREATE TABLE IF NOT EXISTS h2h_cache_enhanced (
    id SERIAL PRIMARY KEY,
    team1_id INTEGER NOT NULL,
    team2_id INTEGER NOT NULL,
    competition_id INTEGER,
    
    -- Overall statistics
    total_games INTEGER DEFAULT 0,
    team1_wins INTEGER DEFAULT 0,
    team2_wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    
    -- Goal statistics
    avg_total_goals DECIMAL(4,2) DEFAULT 0.0,
    avg_team1_goals DECIMAL(4,2) DEFAULT 0.0,
    avg_team2_goals DECIMAL(4,2) DEFAULT 0.0,
    total_goals_scored INTEGER DEFAULT 0,
    
    -- Home/Away breakdown
    team1_home_games INTEGER DEFAULT 0,
    team1_home_wins INTEGER DEFAULT 0,
    team1_home_draws INTEGER DEFAULT 0,
    team1_home_losses INTEGER DEFAULT 0,
    team1_home_goals_for INTEGER DEFAULT 0,
    team1_home_goals_against INTEGER DEFAULT 0,
    
    team2_home_games INTEGER DEFAULT 0,
    team2_home_wins INTEGER DEFAULT 0,
    team2_home_draws INTEGER DEFAULT 0,
    team2_home_losses INTEGER DEFAULT 0,
    team2_home_goals_for INTEGER DEFAULT 0,
    team2_home_goals_against INTEGER DEFAULT 0,
    
    -- Recent form (last 5 meetings)
    recent_games_data TEXT, -- JSON array of recent matches
    recent_avg_goals DECIMAL(4,2) DEFAULT 0.0,
    
    -- Cache metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cache_expires_at TIMESTAMP,
    data_source VARCHAR(20) DEFAULT 'api',
    etag TEXT,
    api_calls_used INTEGER DEFAULT 1,
    
    CONSTRAINT unique_teams_comp_enhanced UNIQUE(team1_id, team2_id, COALESCE(competition_id, 0))
);

-- Indexes for h2h_cache_enhanced
CREATE INDEX IF NOT EXISTS idx_h2h_enhanced_teams_comp ON h2h_cache_enhanced (team1_id, team2_id, competition_id);
CREATE INDEX IF NOT EXISTS idx_h2h_enhanced_cache_expiry ON h2h_cache_enhanced (cache_expires_at);
CREATE INDEX IF NOT EXISTS idx_h2h_enhanced_last_updated ON h2h_cache_enhanced (last_updated);

-- 3. Feature Computation Log Table
-- This tracks the success/failure of feature computations for monitoring
CREATE TABLE IF NOT EXISTS feature_computation_log (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL,
    computation_type VARCHAR(50) NOT NULL, -- 'form', 'h2h', 'preview', 'weather', 'all'
    status VARCHAR(20) NOT NULL, -- 'success', 'failed', 'skipped', 'partial'
    error_message TEXT,
    computation_time_ms INTEGER,
    api_calls_used INTEGER DEFAULT 0,
    features_computed INTEGER DEFAULT 0,
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional metadata
    retry_count INTEGER DEFAULT 0,
    last_retry_at TIMESTAMP,
    computation_source VARCHAR(20) DEFAULT 'api' -- 'api', 'cache', 'fallback'
);

-- Indexes for feature_computation_log
CREATE INDEX IF NOT EXISTS idx_fcl_fixture_type ON feature_computation_log (fixture_id, computation_type);
CREATE INDEX IF NOT EXISTS idx_fcl_status_date ON feature_computation_log (status, created_at);
CREATE INDEX IF NOT EXISTS idx_fcl_created_at ON feature_computation_log (created_at);
CREATE INDEX IF NOT EXISTS idx_fcl_computation_type ON feature_computation_log (computation_type);

-- 4. API Usage Tracking Table
-- This helps monitor and manage API quota usage
CREATE TABLE IF NOT EXISTS api_usage_log (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(100) NOT NULL,
    request_type VARCHAR(20) NOT NULL, -- 'h2h', 'preview', 'weather', 'fixtures'
    team1_id INTEGER,
    team2_id INTEGER,
    league_id INTEGER,
    fixture_id INTEGER,
    response_status INTEGER, -- HTTP status code
    response_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    api_quota_used INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Rate limiting info
    rate_limit_remaining INTEGER,
    rate_limit_reset_at TIMESTAMP
);

-- Indexes for api_usage_log
CREATE INDEX IF NOT EXISTS idx_aul_created_at ON api_usage_log (created_at);
CREATE INDEX IF NOT EXISTS idx_aul_endpoint ON api_usage_log (endpoint);
CREATE INDEX IF NOT EXISTS idx_aul_request_type ON api_usage_log (request_type);
CREATE INDEX IF NOT EXISTS idx_aul_cache_hit ON api_usage_log (cache_hit);

-- 5. Data Quality Metrics Table
-- This tracks data quality metrics for monitoring
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'feature_completeness', 'api_success_rate', 'cache_hit_rate'
    metric_value DECIMAL(5,4) NOT NULL,
    total_records INTEGER DEFAULT 0,
    successful_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_date_type UNIQUE(metric_date, metric_type)
);

-- Indexes for data_quality_metrics
CREATE INDEX IF NOT EXISTS idx_dqm_date_type ON data_quality_metrics (metric_date, metric_type);
CREATE INDEX IF NOT EXISTS idx_dqm_created_at ON data_quality_metrics (created_at);

-- 6. Feature Computation Queue Table
-- This manages the queue of features that need to be computed
CREATE TABLE IF NOT EXISTS feature_computation_queue (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL,
    priority INTEGER DEFAULT 5, -- 1 (highest) to 10 (lowest)
    computation_types TEXT[], -- Array of computation types needed
    scheduled_for TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_attempt_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for feature_computation_queue
CREATE INDEX IF NOT EXISTS idx_fcq_status_priority ON feature_computation_queue (status, priority, scheduled_for);
CREATE INDEX IF NOT EXISTS idx_fcq_fixture_id ON feature_computation_queue (fixture_id);
CREATE INDEX IF NOT EXISTS idx_fcq_scheduled_for ON feature_computation_queue (scheduled_for);

-- 7. Add foreign key constraints where possible
-- Note: We'll add these as soft constraints since the referenced tables might not have the exact structure

-- Add comments for documentation
COMMENT ON TABLE pre_computed_features IS 'Stores all pre-computed features for training to eliminate API dependency during model training';
COMMENT ON TABLE h2h_cache_enhanced IS 'Enhanced head-to-head statistics cache with comprehensive team performance data';
COMMENT ON TABLE feature_computation_log IS 'Tracks the success and failure of feature computation operations for monitoring';
COMMENT ON TABLE api_usage_log IS 'Monitors API usage for quota management and performance optimization';
COMMENT ON TABLE data_quality_metrics IS 'Tracks data quality metrics for system health monitoring';
COMMENT ON TABLE feature_computation_queue IS 'Manages the queue of features that need to be computed';

-- 8. Create views for common queries

-- View for training data with complete features
CREATE OR REPLACE VIEW training_ready_features AS
SELECT 
    pcf.*,
    f.status as fixture_status,
    f.match_date as fixture_date
FROM pre_computed_features pcf
JOIN fixtures f ON pcf.fixture_id = f.id
WHERE pcf.data_quality_score >= 0.8
    AND pcf.total_goals IS NOT NULL
    AND f.status = 'FINISHED';

-- View for feature computation statistics
CREATE OR REPLACE VIEW feature_computation_stats AS
SELECT 
    DATE(created_at) as computation_date,
    computation_type,
    COUNT(*) as total_computations,
    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_computations,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_computations,
    AVG(computation_time_ms) as avg_computation_time_ms,
    SUM(api_calls_used) as total_api_calls
FROM feature_computation_log
GROUP BY DATE(created_at), computation_type
ORDER BY computation_date DESC, computation_type;

-- View for daily API usage summary
CREATE OR REPLACE VIEW daily_api_usage AS
SELECT 
    DATE(created_at) as usage_date,
    request_type,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN cache_hit = true THEN 1 END) as cache_hits,
    COUNT(CASE WHEN cache_hit = false THEN 1 END) as api_calls,
    AVG(response_time_ms) as avg_response_time_ms,
    SUM(api_quota_used) as total_quota_used
FROM api_usage_log
GROUP BY DATE(created_at), request_type
ORDER BY usage_date DESC, request_type;

-- 9. Insert initial data quality tracking records
INSERT INTO processing_status (process_name, status, details) VALUES
    ('feature_precomputation', 'pending', 'Pre-computation of features for training'),
    ('h2h_cache_refresh', 'pending', 'Refresh of H2H statistics cache'),
    ('data_quality_check', 'pending', 'Daily data quality validation'),
    ('api_quota_monitor', 'pending', 'API quota usage monitoring')
ON CONFLICT (process_name) DO NOTHING;

-- 10. Create triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply the trigger to relevant tables
CREATE TRIGGER update_feature_computation_queue_updated_at 
    BEFORE UPDATE ON feature_computation_queue 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO formfinder_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO formfinder_app;

COMMIT;