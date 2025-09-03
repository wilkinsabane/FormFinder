-- Migration script to create workflow and job execution tracking tables
-- for the FormFinder Data Collection and Training Separation architecture

-- Create workflow_executions table for tracking orchestrator runs
CREATE TABLE IF NOT EXISTS workflow_executions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    workflow_status VARCHAR(20) NOT NULL CHECK (workflow_status IN ('completed', 'partial', 'failed')),
    total_tasks INTEGER NOT NULL DEFAULT 0,
    completed_tasks INTEGER NOT NULL DEFAULT 0,
    failed_tasks INTEGER NOT NULL DEFAULT 0,
    duration FLOAT NOT NULL DEFAULT 0.0,
    details JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create index for efficient querying
CREATE INDEX IF NOT EXISTS idx_workflow_executions_timestamp ON workflow_executions(timestamp);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(workflow_status);

-- Create job_executions table for tracking scheduled job runs
CREATE TABLE IF NOT EXISTS job_executions (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(100) NOT NULL,
    job_name VARCHAR(200) NOT NULL,
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN (
        'data_collection', 'feature_computation', 'model_training', 
        'data_quality_check', 'full_workflow', 'maintenance'
    )),
    status VARCHAR(20) NOT NULL CHECK (status IN (
        'scheduled', 'running', 'completed', 'failed', 'skipped'
    )),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    duration FLOAT NOT NULL DEFAULT 0.0,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_job_executions_job_id ON job_executions(job_id);
CREATE INDEX IF NOT EXISTS idx_job_executions_start_time ON job_executions(start_time);
CREATE INDEX IF NOT EXISTS idx_job_executions_status ON job_executions(status);
CREATE INDEX IF NOT EXISTS idx_job_executions_job_type ON job_executions(job_type);

-- Create h2h_cache table if it doesn't exist (for H2H manager)
CREATE TABLE IF NOT EXISTS h2h_cache (
    id SERIAL PRIMARY KEY,
    team1_id INTEGER NOT NULL,
    team2_id INTEGER NOT NULL,
    competition_id INTEGER NOT NULL,
    total_matches INTEGER NOT NULL DEFAULT 0,
    team1_wins INTEGER NOT NULL DEFAULT 0,
    team2_wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    team1_goals INTEGER NOT NULL DEFAULT 0,
    team2_goals INTEGER NOT NULL DEFAULT 0,
    avg_total_goals DECIMAL(4,2) NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create unique constraint and indexes for h2h_cache
CREATE UNIQUE INDEX IF NOT EXISTS idx_h2h_cache_teams_competition 
    ON h2h_cache(LEAST(team1_id, team2_id), GREATEST(team1_id, team2_id), competition_id);
CREATE INDEX IF NOT EXISTS idx_h2h_cache_updated_at ON h2h_cache(updated_at);

-- Create feature_computation_log table for tracking feature computation status
CREATE TABLE IF NOT EXISTS feature_computation_log (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL,
    computation_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'computing', 'completed', 'failed')),
    features_computed INTEGER NOT NULL DEFAULT 0,
    total_features INTEGER NOT NULL DEFAULT 0,
    quality_score DECIMAL(5,4),
    missing_features_pct DECIMAL(5,2),
    computation_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for feature_computation_log
CREATE INDEX IF NOT EXISTS idx_feature_computation_log_fixture_id ON feature_computation_log(fixture_id);
CREATE INDEX IF NOT EXISTS idx_feature_computation_log_status ON feature_computation_log(status);
CREATE INDEX IF NOT EXISTS idx_feature_computation_log_computation_date ON feature_computation_log(computation_date);

-- Create system_metrics table for monitoring system performance
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metric_unit VARCHAR(20),
    category VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for system_metrics
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_category ON system_metrics(category);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Create data_quality_reports table for tracking data quality over time
CREATE TABLE IF NOT EXISTS data_quality_reports (
    id SERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    total_fixtures INTEGER NOT NULL DEFAULT 0,
    fixtures_with_features INTEGER NOT NULL DEFAULT 0,
    avg_quality_score DECIMAL(5,4),
    avg_missing_features_pct DECIMAL(5,2),
    api_calls_made INTEGER NOT NULL DEFAULT 0,
    api_quota_used_pct DECIMAL(5,2),
    computation_failures INTEGER NOT NULL DEFAULT 0,
    details JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create unique constraint and index for data_quality_reports
CREATE UNIQUE INDEX IF NOT EXISTS idx_data_quality_reports_date ON data_quality_reports(report_date);
CREATE INDEX IF NOT EXISTS idx_data_quality_reports_created_at ON data_quality_reports(created_at);

-- Add comments for documentation
COMMENT ON TABLE workflow_executions IS 'Tracks orchestrator workflow execution history';
COMMENT ON TABLE job_executions IS 'Tracks scheduled job execution history';
COMMENT ON TABLE h2h_cache IS 'Caches head-to-head statistics between teams';
COMMENT ON TABLE feature_computation_log IS 'Tracks feature computation status for individual fixtures';
COMMENT ON TABLE system_metrics IS 'Stores system performance and monitoring metrics';
COMMENT ON TABLE data_quality_reports IS 'Daily data quality summary reports';

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, category, metadata) VALUES
('training_pipeline_setup', 1, 'boolean', 'system', '{"description": "Training pipeline setup completed"}'),
('database_schema_version', 2.0, 'version', 'system', '{"description": "Database schema version for workflow tracking"}'),
('feature_computation_enabled', 1, 'boolean', 'features', '{"description": "Feature computation system enabled"}');

COMMIT;