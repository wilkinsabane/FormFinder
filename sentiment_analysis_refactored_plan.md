Of course. Here is a meticulous, three-phase plan to refactor your script, focusing on modern practices, modularity, robustness, and enhanced verbosity, while using a `config.yaml` file for configuration.

-----

### \#\# Phase 1: Establish a Modern Foundation

This phase focuses on creating a robust and secure configuration system and structuring the project for modularity.

**Step 1.1: Create the `config.yaml` File**
Centralize all your settings into a clean, human-readable YAML file.

  * **Action:** Create a file named `config.yaml` in the root of your project with the following structure. This keeps your credentials and parameters separate from your application logic.

    ```yaml
    # config.yaml

    database:
      enabled: true
      log_operations: true

    file_processing:
      enabled: false # Set to false to prioritize the database-first approach
      fixtures_dir: 'data/fixtures'
      processed_dir: 'processed_data'

    sentiment_analysis:
      enabled: true
      news_api_key: "YOUR_SECRET_API_KEY_HERE" # Paste your key here
      cache_hours: 24
      # Weights for combining form and sentiment scores
      prediction_weights:
        form: 0.7
        sentiment: 0.3

    output_options:
      directory: 'data/predictions'
      log_level: 'DEBUG' # Control verbosity: DEBUG, INFO, WARNING
    ```

**Step 1.2: Install Required Libraries**
You'll need libraries for YAML parsing and data validation.

  * **Action:** Install `PyYAML` and `Pydantic` in your Conda environment. `Pydantic` will validate your `config.yaml` at runtime, preventing errors from misconfiguration.

    ```bash
    pip install pyyaml pydantic
    ```

**Step 1.3: Create a `SentimentAnalyzer` Class**
To make your code modular, we will extract all sentiment-related logic into its own dedicated class. This follows the **Single Responsibility Principle**.

  * **Action:** Create a new file, perhaps `formfinder/sentiment.py`, and define a `SentimentAnalyzer` class. This class will handle API calls, caching, and analysis, completely independent of the `PredictorOutputter`.

      * The `__init__` method will take the `sentiment_analysis` config section and initialize the cache.
      * It will contain the methods `_fetch_team_articles` and `_analyze_articles_sentiment`.
      * It will have a primary public method, `get_match_sentiment`, which will be called by the main script.

-----

### \#\# Phase 2: Implement the Refactored Logic

This phase involves rewriting parts of your script to use the new modular structure and to incorporate the robustness and verbosity enhancements.

**Step 2.1: Implement Pydantic Configuration Models**
Create Python classes that define the exact structure and data types of your `config.yaml`.

  * **Action:** In a new file like `formfinder/config_models.py`, define your Pydantic models. This ensures your configuration is loaded correctly and is type-safe.

    ```python
    # formfinder/config_models.py
    from pydantic import BaseModel, Field
    from typing import Literal

    class SentimentWeights(BaseModel):
        form: float = Field(..., ge=0, le=1) # ge=0 means >= 0
        sentiment: float = Field(..., ge=0, le=1)

    class SentimentConfig(BaseModel):
        enabled: bool
        news_api_key: str
        cache_hours: int = Field(..., gt=0) # gt=0 means > 0
        prediction_weights: SentimentWeights

    class AppConfig(BaseModel):
        sentiment_analysis: SentimentConfig
        #... define models for database, file_processing, etc.
    ```

**Step 2.2: Refactor the Main Script (`standalone_predictor_outputter.py`)**
Update the script to load the YAML file, validate it with Pydantic, and use the new `SentimentAnalyzer` class.

  * **Action:**
    1.  **Modify `load_config`:** Change this function to read `config.yaml`, parse it with `PyYAML`, and validate the result using your Pydantic `AppConfig` model. This function should return a Pydantic object, not a dictionary.

    2.  **Update `PredictorOutputter.__init__`:** It should now accept the Pydantic `AppConfig` object. It will instantiate the `SentimentAnalyzer` class, passing the relevant config section to it.

        ```python
        # In PredictorOutputter.__init__
        self.config = config # This is now a Pydantic object
        self.sentiment_analyzer = None
        if self.config.sentiment_analysis.enabled:
            logger.info("✅ Sentiment analysis is enabled. Initializing analyzer.")
            self.sentiment_analyzer = SentimentAnalyzer(self.config.sentiment_analysis)
        else:
            logger.warning("Sentiment analysis is disabled in the configuration.")
        ```

    3.  **Update `process_league_from_dataframes`:** Instead of calling its own sentiment methods, it will now call the `SentimentAnalyzer` instance. This makes the code cleaner and easier to read.

        ```python
        # Inside the loop over matches
        if self.sentiment_analyzer:
            # The analyzer handles its own logic, including caching and retries
            home_sentiment, away_sentiment, count = self.sentiment_analyzer.get_match_sentiment(
                home_team_name=match['home_team_name'],
                away_team_name=match['away_team_name']
            )
            #... update dataframe
        ```

**Step 2.3: Enhance Logging for Extreme Verbosity**
Add detailed log messages at every critical step of the process.

  * **Action:** Sprinkle `logger.debug()` and `logger.info()` calls throughout your code.

      * **Start/End of Functions:** `logger.debug("Entering 'process_league' for league_id: %s", league_id)` and `logger.debug("Finished 'process_league'.")`
      * **Data Shape:** `logger.info("Found %d upcoming fixtures and %d high-form teams for league %s.", len(fixtures), len(high_form_teams), league_id)`
      * **Decisions:** `logger.debug("Sentiment weights: Form=%.2f, Sentiment=%.2f", weight_form, weight_sentiment)`
      * **Cache Logic:** `logger.debug("CACHE HIT for team: %s", team_name)` and `logger.debug("CACHE MISS for team: %s. Fetching from API.", team_name)`
      * **API Retries:** `logger.warning("API rate limit hit. Retrying in %d seconds... (Attempt %d/%d)", wait_time, attempt, max_retries)`

-----

### \#\# Phase 3: Verification and Final Polish

This phase ensures the refactored system works correctly and is easy to maintain.

**Step 3.1: Write Unit Tests for the `SentimentAnalyzer`**
A modular design makes testing much easier.

  * **Action:**
    1.  Create a test file `tests/test_sentiment.py`.
    2.  Write a test that checks if the `SentimentAnalyzer` correctly parses its configuration.
    3.  Using Python's `unittest.mock`, "mock" the `requests.get` call. This allows you to test your retry and caching logic *without* actually hitting the live API. You can simulate a `429` error and assert that your code waits and retries correctly.

**Step 3.2: Full Execution and Log Review**
Run the entire pipeline and meticulously review the logs.

  * **Action:**
    1.  Delete your old log file to start fresh.
    2.  Run the main script: `python standalone_predictor_outputter.py --config config.yaml`.
    3.  Open the new log file. It should be "extremely verbose," telling a complete story of the execution: which config was loaded, how many leagues were processed, which teams were pulled from the cache vs. the API, and the final predictions being saved. The logs are your proof that the system is working as designed.