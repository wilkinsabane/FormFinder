-- Remove specified H2H columns from pre_computed_features table
-- WARNING: BACKUP DATABASE BEFORE RUNNING!

BEGIN;

-- Drop the specified H2H columns
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_overall_games;
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_avg_total_goals;
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_overall_home_goals;
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_overall_away_goals;
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_home_advantage;
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_team1_wins;
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_team2_wins;
ALTER TABLE pre_computed_features DROP COLUMN IF EXISTS h2h_draws;

-- Verify the columns have been removed
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'pre_computed_features' 
AND column_name IN (
    'h2h_overall_games',
    'h2h_avg_total_goals', 
    'h2h_overall_home_goals',
    'h2h_overall_away_goals',
    'h2h_home_advantage',
    'h2h_team1_wins',
    'h2h_team2_wins',
    'h2h_draws'
);

-- If the above query returns no rows, the columns have been successfully removed
-- Uncomment the next line to commit the changes
-- COMMIT;

-- By default, rollback to be safe
ROLLBACK;