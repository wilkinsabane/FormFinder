import pandas as pd
from datetime import datetime

# Read the predictions CSV
df = pd.read_csv('predictions/latest_predictions.csv')

print('Match dates in predictions:')
print(df['match_date'].value_counts().head(10))

print('\nToday matches:')
today_matches = df[df['match_date'].str.startswith('2025-09-03')]
print(f'Found {len(today_matches)} matches for today')

if len(today_matches) > 0:
    print('Sample today matches:')
    print(today_matches[['home_team', 'away_team', 'over_2_5_probability']].head())
    print('\nConfidence scores for today:')
    print(today_matches['over_2_5_probability'].describe())
else:
    print('\nNo matches found for today. Checking earliest dates:')
    print(df['match_date'].min())
    print('Latest dates:')
    print(df['match_date'].max())