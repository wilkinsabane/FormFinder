-- Clear existing data
TRUNCATE TABLE pre_computed_features;

-- Insert sample data with only essential columns
INSERT INTO pre_computed_features (
    fixture_id, league_id, match_date, home_team_id, away_team_id,
    total_goals, home_score, away_score, match_result, over_2_5,
    data_quality_score
) VALUES
(1001, 181, '2024-01-15', 1, 2, 3, 2, 1, 'H', true, 1.0),
(1002, 181, '2024-01-16', 3, 4, 2, 1, 1, 'D', false, 1.0),
(1003, 181, '2024-01-17', 5, 6, 4, 3, 1, 'H', true, 1.0),
(1004, 181, '2024-01-18', 7, 8, 1, 0, 1, 'A', false, 1.0),
(1005, 181, '2024-01-19', 9, 10, 5, 3, 2, 'H', true, 1.0),
(1006, 203, '2024-01-20', 11, 12, 3, 2, 1, 'H', true, 1.0),
(1007, 203, '2024-01-21', 13, 14, 1, 0, 1, 'A', false, 1.0),
(1008, 203, '2024-01-22', 15, 16, 4, 3, 1, 'H', true, 1.0),
(1009, 203, '2024-01-23', 17, 18, 2, 1, 1, 'D', false, 1.0),
(1010, 203, '2024-01-24', 19, 20, 6, 4, 2, 'H', true, 1.0);

-- Generate additional random sample data
INSERT INTO pre_computed_features (
    fixture_id, league_id, match_date, home_team_id, away_team_id,
    total_goals, home_score, away_score, match_result, over_2_5,
    data_quality_score
)
SELECT 
    1100 + generate_series as fixture_id,
    (ARRAY[181, 203, 204, 205])[floor(random() * 4 + 1)] as league_id,
    '2024-01-01'::date + (random() * 365)::int as match_date,
    floor(random() * 20 + 1)::int as home_team_id,
    floor(random() * 20 + 21)::int as away_team_id,
    floor(random() * 6 + 0)::int as total_goals,
    floor(random() * 4 + 0)::int as home_score,
    floor(random() * 4 + 0)::int as away_score,
    (ARRAY['H', 'A', 'D'])[floor(random() * 3 + 1)] as match_result,
    random() > 0.5 as over_2_5,
    1.0 as data_quality_score
FROM generate_series(1, 50);