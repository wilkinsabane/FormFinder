A comprehensive plan on how to best integrate sentiment analysis.

Your current system is well-structured, which makes adding a new feature like this much more manageable. You were right to suggest performing this analysis within the prediction script before the final predictions are made. The ideal place is within the `process_league` and `save_predictions_to_db` methods.

Here is a step-by-step guide to achieve this, transforming your predictor from being purely quantitative (win rates) to a hybrid model that also considers qualitative data (news sentiment).

### **High-Level Plan**

1.  **Fetch News Data:** We'll need a way to get relevant news articles for an upcoming match. The best approach is using a News API for reliability.
2.  **Perform Sentiment Analysis:** Once we have the text, we'll analyze it to get a sentiment score (e.g., from -1.0 for very negative to +1.0 for very positive).
3.  **Update Database Schema:** We'll add columns to the `predictions` table to store the sentiment scores.
4.  **Integrate into the Script:** We will modify `standalone_predictor_outputter.py` to orchestrate the fetching, analysis, and storage.
5.  **Adjust Prediction Logic:** The final and most important step is to combine the existing `win_rate` (form score) with the new sentiment score to produce a more nuanced prediction.

-----

### **Step 1: Choose Your Tools**

First, you'll need to install a couple of new libraries.

  * **For Fetching News:** `requests` is the standard for making HTTP requests to an API.
  * **For Sentiment Analysis:**
      * **Easy Option:** `textblob`. It's very simple to use and provides a polarity score out of the box. No API key needed.
      * **Advanced Option:** Hugging Face's `transformers` library. This is more powerful but also more complex to set up.

For this guide, we'll use `requests` and `textblob` for their simplicity and effectiveness.

```bash
pip install requests textblob
# TextBlob may need to download its corpora on first use
python -m textblob.download_corpora
```

You will also need a **News API Key**. A great free option is [NewsAPI.org](https://newsapi.org/). Register and get a free developer key.

### **Step 2: Update Your Database Model**

You need to add fields to your `Prediction` model to store the sentiment data. Open your `formfinder/database.py` file and modify the `Prediction` class.

**File: `formfinder/database.py` (Suggested Changes)**

```python
# In your Prediction class
class Prediction(Base):
    # ... other columns
    home_team_form_score = Column(Float)
    away_team_form_score = Column(Float)
    
    # --- ADD THESE NEW COLUMNS ---
    home_team_sentiment = Column(Float, nullable=True)
    away_team_sentiment = Column(Float, nullable=True)
    sentiment_articles_analyzed = Column(Integer, nullable=True)
    # --- END OF NEW COLUMNS ---
    
    confidence_score = Column(Float)
    algorithm_version = Column(String)
    features_used = Column(JSON) # or Text
    prediction_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
```

After changing the model, you will need to migrate your database schema. If you're using a tool like Alembic, create a new migration. If you're in development, you might just drop and recreate the tables.

### **Step 3: Update Configuration**

Add a new section to your configuration for sentiment analysis. This keeps your API key and other settings out of the code.

**File: `sdata_init_config.json` (or your default config)**

```json
{
  "output_directory": "data/predictions",
  "database": { ... },
  "file_processing": { ... },
  "sentiment_analysis": {
    "enabled": true,
    "news_api_key": "YOUR_NEWS_API_KEY_HERE",
    "prediction_weight_form": 0.7,
    "prediction_weight_sentiment": 0.3
  }
}
```

We'll also update the `DEFAULT_CONFIG` in the script to reflect this.

### **Step 4: Refactor `standalone_predictor_outputter.py`**

This is where we'll put everything together. I will provide a refactored version of your script. The key changes are:

1.  **New Imports:** Add `requests` and `TextBlob`.

2.  **Updated `DEFAULT_CONFIG`:** Include the new `sentiment_analysis` section.

3.  **New Helper Functions:**

      * `get_sentiment_for_match()`: This function will take two team names, query the News API, and return an average sentiment score for each team.

4.  **Modified `PredictorOutputter.process_league()`:** It will now call `get_sentiment_for_match()` for each fixture and add the sentiment scores to the DataFrame.

5.  **Modified `PredictorOutputter.save_predictions_to_db()`:** This is the most critical change. The prediction logic will now use a weighted average of the form score and the sentiment score. The formula will be:

    $final\_score = (\\alpha \\times form\_score) + (\\beta \\times sentiment\_score)$

    Where $\\alpha$ and $\\beta$ are the weights from your config file (`prediction_weight_form` and `prediction_weight_sentiment`).

Below is the updated and annotated script. I've marked new or significantly changed sections with `### --- NEW/MODIFIED --- ###`.

-----

### **Updated `standalone_predictor_outputter.py`**

```python
#!/usr/bin/env python
# ... (all your existing imports) ...
import requests
from textblob import TextBlob
from datetime import datetime, timezone, timedelta

# ... (rest of your imports, logging, etc.) ...

### --- MODIFIED --- ###
# Update default configuration to include sentiment analysis settings
DEFAULT_CONFIG = {
    'output_directory': 'data/predictions',
    'database': {
        'enabled': True,
        'log_operations': True
    },
    'file_processing': {
        'enabled': True,
        'fixtures_dir': 'data/fixtures',
        'processed_dir': 'processed_data'
    },
    'sentiment_analysis': {
        'enabled': True,
        'news_api_key': os.getenv('NEWS_API_KEY', None), # Best practice: use env var
        'prediction_weight_form': 0.7,
        'prediction_weight_sentiment': 0.3
    }
}

class PredictorOutputterError(Exception):
    """Custom exception for PredictorOutputter errors."""
    pass


### --- NEW FUNCTION --- ###
def get_sentiment_for_match(home_team: str, away_team: str, api_key: str) -> Tuple[Optional[float], Optional[float], int]:
    """
    Fetches news and calculates sentiment for the home and away teams.

    Args:
        home_team: Name of the home team.
        away_team: Name of the away team.
        api_key: API key for NewsAPI.org.

    Returns:
        A tuple of (home_sentiment, away_sentiment, articles_analyzed).
    """
    if not api_key:
        logger.warning("News API key not provided. Skipping sentiment analysis.")
        return None, None, 0

    def get_team_sentiment(team_name: str) -> Optional[float]:
        sentiments = []
        # Search for news in the last 7 days
        from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
        query = f'"{team_name}" football'
        url = (f'https://newsapi.org/v2/everything?'
               f'q={query}&'
               f'from={from_date}&'
               f'language=en&'
               f'sortBy=relevancy&'
               f'apiKey={api_key}')
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            
            if not articles:
                return None

            for article in articles[:10]: # Analyze up to 10 most relevant articles
                text = f"{article['title']}. {article['description'] or ''}"
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
            
            return sum(sentiments) / len(sentiments) if sentiments else None
        except requests.RequestException as e:
            logger.error(f"Failed to fetch news for {team_name}: {e}")
            return None

    home_sentiment = get_team_sentiment(home_team)
    away_sentiment = get_team_sentiment(away_team)
    
    # We can refine how we count articles, here it's just a flag that it worked
    articles_analyzed = 10 if home_sentiment is not None or away_sentiment is not None else 0
    
    return home_sentiment, away_sentiment, articles_analyzed


class PredictorOutputter:
    def __init__(self, config: Dict[str, Any]):
        # ... (your existing __init__ logic) ...
        self.leagues_filepath = config.get('leagues_filepath', 'leagues.json')
        
        ### --- MODIFIED --- ###
        # Load sentiment analysis configuration
        self.sentiment_config = config.get('sentiment_analysis', {})
        self.sentiment_enabled = self.sentiment_config.get('enabled', False)
        self.news_api_key = self.sentiment_config.get('news_api_key')
        self.weight_form = float(self.sentiment_config.get('prediction_weight_form', 0.7))
        self.weight_sentiment = float(self.sentiment_config.get('prediction_weight_sentiment', 0.3))

        logger.info("Sentiment Analysis Configuration:")
        logger.info(f"  - Enabled: {self.sentiment_enabled}")
        logger.info(f"  - News API Key Loaded: {'Yes' if self.news_api_key else 'No'}")
        logger.info(f"  - Form Weight: {self.weight_form}, Sentiment Weight: {self.weight_sentiment}")
        
        # ... (rest of your __init__ logic) ...

    # ... (all your _validate_path_param, _load_leagues_data, etc. methods remain the same) ...

    ### --- MODIFIED --- ###
    def process_league(self, league_id, high_form_file_path, fixtures_file_path):
        # ... (your existing logic to load high_form_teams and fixtures) ...
        high_form_teams = self.load_high_form_teams(high_form_file_path)
        if high_form_teams.empty:
            # ...
            return pd.DataFrame()
        
        fixtures = self.load_fixtures(fixtures_file_path)
        if fixtures.empty:
            # ...
            return pd.DataFrame()
            
        # ... (your existing logic for win_rate_dict and adding league info) ...

        fixtures['home_win_rate'] = fixtures['home_team_id'].map(win_rate_dict)
        fixtures['away_win_rate'] = fixtures['away_team_id'].map(win_rate_dict)
        
        ### --- NEW --- ###
        # Add sentiment analysis if enabled
        if self.sentiment_enabled:
            sentiments = fixtures.apply(
                lambda row: get_sentiment_for_match(
                    row['home_team_name'],
                    row['away_team_name'],
                    self.news_api_key
                ),
                axis=1
            )
            fixtures[['home_team_sentiment', 'away_team_sentiment', 'articles_analyzed']] = pd.DataFrame(sentiments.tolist(), index=fixtures.index)
            logger.info(f"Completed sentiment analysis for {len(fixtures)} fixtures in league {league_id}.")
        else:
            fixtures['home_team_sentiment'] = None
            fixtures['away_team_sentiment'] = None
            fixtures['articles_analyzed'] = 0

        flagged_fixtures = fixtures[
            fixtures['home_win_rate'].notnull() | fixtures['away_win_rate'].notnull()
        ].copy()
        
        ### --- MODIFIED --- ###
        # Update output columns to include sentiment
        output_columns = [
            'league_id', 'league_name', 'country', 'match_id', 'date', 'time', 
            'home_team_name', 'home_win_rate', 'home_team_sentiment',
            'away_team_name', 'away_win_rate', 'away_team_sentiment',
            'articles_analyzed'
        ]
        
        for col in output_columns:
            if col not in flagged_fixtures.columns:
                flagged_fixtures[col] = None

        return flagged_fixtures[output_columns]

    ### --- MODIFIED --- ###
    def save_predictions_to_db(self, flagged_matches):
        # ... (your existing setup logic for this method) ...
        try:
            session = get_db_session()
            predictions_saved = 0
            
            for _, match in flagged_matches.iterrows():
                # ... (your existing logic to find the fixture in the DB) ...
                if fixture is None:
                    # ...
                    continue
                
                # --- Get form scores ---
                home_form_score = match.get('home_win_rate', 0)
                if pd.isna(home_form_score): home_form_score = 0
                
                away_form_score = match.get('away_win_rate', 0)
                if pd.isna(away_form_score): away_form_score = 0

                # --- Get sentiment scores ---
                home_sentiment = match.get('home_team_sentiment')
                if pd.isna(home_sentiment): home_sentiment = 0.0 # Neutral sentiment if not available
                
                away_sentiment = match.get('away_team_sentiment')
                if pd.isna(away_sentiment): away_sentiment = 0.0 # Neutral sentiment if not available

                # --- Combine scores using weights ---
                # Normalize sentiment from [-1, 1] to [0, 1] for easier combination
                # A simple way is (score + 1) / 2
                home_sentiment_norm = (home_sentiment + 1) / 2
                away_sentiment_norm = (away_sentiment + 1) / 2

                # Calculate combined score
                home_combined_score = (self.weight_form * home_form_score) + (self.weight_sentiment * home_sentiment_norm)
                away_combined_score = (self.weight_form * away_form_score) + (self.weight_sentiment * away_sentiment_norm)

                # --- Simple probability model based on combined scores ---
                # This part can be made more sophisticated
                total_score = home_combined_score + away_combined_score
                if total_score > 0:
                    home_win_prob = home_combined_score / total_score
                    away_win_prob = away_combined_score / total_score
                else:
                    home_win_prob = 0.5
                    away_win_prob = 0.5
                
                # Assume draw probability is a fixed portion for now, or derive it
                draw_prob = 1.0 - (home_win_prob + away_win_prob)
                if draw_prob < 0: # Normalize if sum > 1
                    factor = 1 / (home_win_prob + away_win_prob)
                    home_win_prob *= factor
                    away_win_prob *= factor
                    draw_prob = 0.0

                existing_prediction = session.query(Prediction).filter(Prediction.fixture_id == fixture.id).first()
                
                if existing_prediction:
                    # Update existing prediction
                    existing_prediction.home_win_probability = home_win_prob
                    existing_prediction.draw_probability = draw_prob
                    existing_prediction.away_win_probability = away_win_prob
                    existing_prediction.home_team_form_score = home_form_score # Original form score
                    existing_prediction.away_team_form_score = away_form_score # Original form score
                    existing_prediction.home_team_sentiment = home_sentiment  # Original sentiment
                    existing_prediction.away_team_sentiment = away_sentiment  # Original sentiment
                    existing_prediction.sentiment_articles_analyzed = match.get('articles_analyzed')
                    existing_prediction.confidence_score = max(home_win_prob, away_win_prob)
                    existing_prediction.prediction_date = datetime.now(timezone.utc)
                    existing_prediction.updated_at = datetime.now(timezone.utc)
                    logger.debug(f"Updated prediction for fixture {fixture.id}")
                else:
                    # Create new prediction
                    new_prediction = Prediction(
                        fixture_id=fixture.id,
                        home_win_probability=home_win_prob,
                        draw_probability=draw_prob,
                        away_win_probability=away_win_prob,
                        home_team_form_score=home_form_score,
                        away_team_form_score=away_form_score,
                        home_team_sentiment=home_sentiment,
                        away_team_sentiment=away_sentiment,
                        sentiment_articles_analyzed=match.get('articles_analyzed'),
                        confidence_score=max(home_win_prob, away_win_prob),
                        algorithm_version="hybrid_form_sentiment_v1",
                        features_used=json.dumps(["recent_win_rate", "news_sentiment"]),
                        prediction_date=datetime.now(timezone.utc)
                    )
                    session.add(new_prediction)
                    logger.debug(f"Created new prediction for fixture {fixture.id}")
                
                predictions_saved += 1
            
            if predictions_saved > 0:
                session.commit()
                logger.info(f"Saved {predictions_saved} predictions to database.")
            
            session.close()
            return predictions_saved
        except Exception as e:
            # ... (your existing error handling) ...
            return 0

    # ... (the rest of your script: run_predictor_outputter, load_config, main) ...
    # No changes are needed for the other methods.
```

### **Final Steps and Considerations**

1.  **API Key Management:** Do not hardcode your News API key. The code above uses `os.getenv('NEWS_API_KEY', None)`. Set this as an environment variable in your production environment for security.
2.  **Caching:** The `get_sentiment_for_match` function makes live API calls every time it runs. For a given match, the news sentiment won't change drastically every hour. You should implement a caching layer (e.g., using a simple dictionary, a file-based cache, or Redis) to store news results for a few hours to avoid redundant API calls and stay within your rate limits.
3.  **Error Handling:** The new code includes basic error handling for the API requests, but you can make it more robust (e.g., handling specific HTTP error codes like 429 for rate limiting).
4.  **Tuning Weights:** The weights `0.7` for form and `0.3` for sentiment are just a starting point. The fun part is tuning these values. You could even make them dynamic based on how much news is available or how close the match is.
5.  **Refining Sentiment:** `TextBlob` is general-purpose. You might find that news headlines are often neutral. You could refine the analysis by:
      * Focusing on specific keywords (e.g., "injury," "confident," "scandal," "new signing").
      * Using a more domain-specific model (like a sports-tuned transformer model) if you find the results are not discriminative enough.

This implementation provides a solid foundation for including sentiment analysis in your `FormFinder` app, making your predictions much more sophisticated.