-- 1) H2H cache table (24h TTL policy handled in code)
CREATE TABLE IF NOT EXISTS h2h_cache (
  team1_id INTEGER NOT NULL,
  team2_id INTEGER NOT NULL,
  competition_id INTEGER,
  -- overall
  overall_games_played INTEGER NOT NULL,
  overall_team1_wins INTEGER NOT NULL,
  overall_team2_wins INTEGER NOT NULL,
  overall_draws INTEGER NOT NULL,
  overall_team1_scored INTEGER NOT NULL,
  overall_team2_scored INTEGER NOT NULL,
  -- team1 at home
  team1_games_played_at_home INTEGER,
  team1_wins_at_home INTEGER,
  team1_losses_at_home INTEGER,
  team1_draws_at_home INTEGER,
  team1_scored_at_home INTEGER,
  team1_conceded_at_home INTEGER,
  -- team2 at home
  team2_games_played_at_home INTEGER,
  team2_wins_at_home INTEGER,
  team2_losses_at_home INTEGER,
  team2_draws_at_home INTEGER,
  team2_scored_at_home INTEGER,
  team2_conceded_at_home INTEGER,
  -- computed convenience
  avg_total_goals FLOAT NOT NULL,
  -- caching metadata
  etag TEXT,
  last_fetched_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (team1_id, team2_id, COALESCE(competition_id, 0))
);
CREATE INDEX IF NOT EXISTS h2h_cache_last_fetched_idx ON h2h_cache (last_fetched_at DESC);

-- 2) Predictions table alterations for goal-based outputs
ALTER TABLE predictions
  DROP COLUMN IF EXISTS home_win_probability,
  DROP COLUMN IF EXISTS draw_probability,
  DROP COLUMN IF EXISTS away_win_probability,
  ADD COLUMN IF NOT EXISTS predicted_total_goals FLOAT,
  ADD COLUMN IF NOT EXISTS over_2_5_probability FLOAT,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;