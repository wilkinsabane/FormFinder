-- FormFinder Database Schema
-- PostgreSQL schema for sentiment analysis and form analysis

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    team_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    short_code VARCHAR(10),
    country VARCHAR(100),
    logo_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Leagues table
CREATE TABLE IF NOT EXISTS leagues (
    id SERIAL PRIMARY KEY,
    league_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    season VARCHAR(20),
    logo_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matches table
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    match_id INTEGER UNIQUE NOT NULL,
    league_id INTEGER REFERENCES leagues(league_id),
    home_team_id INTEGER REFERENCES teams(team_id),
    away_team_id INTEGER REFERENCES teams(team_id),
    match_date TIMESTAMP NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment sources table
CREATE TABLE IF NOT EXISTS sentiment_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'twitter', 'reddit', 'news', etc.
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw sentiment data table
CREATE TABLE IF NOT EXISTS sentiment_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id INTEGER REFERENCES teams(team_id),
    source_id INTEGER REFERENCES sentiment_sources(id),
    content TEXT NOT NULL,
    sentiment_score DECIMAL(3,2), -- -1.00 to 1.00
    confidence_score DECIMAL(3,2), -- 0.00 to 1.00
    sentiment_label VARCHAR(20), -- 'positive', 'negative', 'neutral'
    author VARCHAR(255),
    post_date TIMESTAMP,
    url TEXT,
    raw_data JSONB,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregated sentiment scores table
CREATE TABLE IF NOT EXISTS team_sentiment_scores (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    date DATE NOT NULL,
    avg_sentiment_score DECIMAL(3,2),
    sentiment_volume INTEGER DEFAULT 0,
    positive_ratio DECIMAL(3,2),
    negative_ratio DECIMAL(3,2),
    neutral_ratio DECIMAL(3,2),
    data_points INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, date)
);

-- Form analysis table
CREATE TABLE IF NOT EXISTS team_form_analysis (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    league_id INTEGER REFERENCES leagues(league_id),
    analysis_date DATE NOT NULL,
    matches_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    win_rate DECIMAL(3,2),
    form_score DECIMAL(3,2), -- -1.00 to 1.00 based on recent performance
    streak VARCHAR(10), -- 'W', 'D', 'L' sequence
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, league_id, analysis_date)
);

-- Combined sentiment and form analysis table
CREATE TABLE IF NOT EXISTS combined_analysis (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    league_id INTEGER REFERENCES leagues(league_id),
    analysis_date DATE NOT NULL,
    sentiment_score DECIMAL(3,2),
    form_score DECIMAL(3,2),
    combined_score DECIMAL(3,2), -- weighted combination
    sentiment_weight DECIMAL(3,2) DEFAULT 0.3,
    form_weight DECIMAL(3,2) DEFAULT 0.7,
    prediction_confidence DECIMAL(3,2),
    trend_direction VARCHAR(20), -- 'improving', 'declining', 'stable'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, league_id, analysis_date)
);

-- Alert thresholds table
CREATE TABLE IF NOT EXISTS alert_thresholds (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    threshold_type VARCHAR(50) NOT NULL, -- 'sentiment', 'form', 'combined'
    min_value DECIMAL(3,2),
    max_value DECIMAL(3,2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notification logs table
CREATE TABLE IF NOT EXISTS notification_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id INTEGER REFERENCES teams(team_id),
    threshold_id INTEGER REFERENCES alert_thresholds(id),
    notification_type VARCHAR(50), -- 'email', 'sms'
    message TEXT,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'sent' -- 'sent', 'failed', 'pending'
);

-- Processing status tracking
CREATE TABLE IF NOT EXISTS processing_status (
    id SERIAL PRIMARY KEY,
    process_name VARCHAR(100) UNIQUE NOT NULL,
    last_run TIMESTAMP,
    status VARCHAR(20), -- 'success', 'failed', 'running'
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sentiment_data_team_date ON sentiment_data(team_id, post_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_source ON sentiment_data(source_id);
CREATE INDEX IF NOT EXISTS idx_team_sentiment_team_date ON team_sentiment_scores(team_id, date);
CREATE INDEX IF NOT EXISTS idx_team_form_team_league_date ON team_form_analysis(team_id, league_id, analysis_date);
CREATE INDEX IF NOT EXISTS idx_combined_analysis_team_league_date ON combined_analysis(team_id, league_id, analysis_date);
CREATE INDEX IF NOT EXISTS idx_matches_league_date ON matches(league_id, match_date);

-- Insert default sentiment sources
INSERT INTO sentiment_sources (source_name, source_type) VALUES
    ('Twitter', 'twitter'),
    ('Reddit', 'reddit'),
    ('News', 'news'),
    ('Sports Blogs', 'blog')
ON CONFLICT (source_name) DO NOTHING;

-- Insert processing status records
INSERT INTO processing_status (process_name, status) VALUES
    ('sentiment_collection', 'pending'),
    ('form_analysis', 'pending'),
    ('combined_analysis', 'pending'),
    ('alert_processing', 'pending')
ON CONFLICT (process_name) DO NOTHING;