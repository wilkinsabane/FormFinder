#!/usr/bin/env python3
"""
Fix initialization issues in enhanced_predictor.py:
1. MarkovFeatureGenerator expects only lookback_window, not db_session
2. SentimentAnalyzer.get_sentiment_for_match expects team names, not team IDs
"""

import logging
from sqlalchemy import text
from formfinder.database import DatabaseManager, get_db_session
from formfinder.config import load_config, get_config
from formfinder.logger import get_logger

logger = get_logger(__name__)

def test_markov_initialization():
    """Test MarkovFeatureGenerator initialization."""
    try:
        from formfinder.markov_feature_generator import MarkovFeatureGenerator
        
        # Test correct initialization (only lookback_window)
        markov_gen = MarkovFeatureGenerator(lookback_window=10)
        logger.info("✅ MarkovFeatureGenerator initialized correctly")
        
        # Test if it has the expected methods
        if hasattr(markov_gen, 'generate_team_features'):
            logger.info("✅ MarkovFeatureGenerator has generate_team_features method")
        else:
            logger.warning("⚠️ MarkovFeatureGenerator missing generate_team_features method")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ MarkovFeatureGenerator initialization failed: {e}")
        return False

def test_sentiment_analyzer():
    """Test SentimentAnalyzer initialization and method signature."""
    try:
        from formfinder.sentiment import SentimentAnalyzer
        
        # Test initialization with dummy API key
        sentiment_analyzer = SentimentAnalyzer()
        logger.info("✅ SentimentAnalyzer initialized correctly")
        
        # Check method signature
        import inspect
        sig = inspect.signature(sentiment_analyzer.get_sentiment_for_match)
        params = list(sig.parameters.keys())
        logger.info(f"✅ get_sentiment_for_match parameters: {params}")
        
        # Expected: ['self', 'home_team', 'away_team', 'match_date', 'days_back']
        expected_params = ['home_team', 'away_team', 'match_date', 'days_back']
        actual_params = [p for p in params if p != 'self']
        
        if actual_params == expected_params:
            logger.info("✅ Method signature matches expected parameters")
        else:
            logger.warning(f"⚠️ Method signature mismatch. Expected: {expected_params}, Got: {actual_params}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ SentimentAnalyzer test failed: {e}")
        return False

def get_team_name_mapping():
    """Get mapping of team IDs to team names."""
    try:
        with get_db_session() as session:
            result = session.execute(text("""
                SELECT id, name 
                FROM teams 
                ORDER BY id
                LIMIT 10
            """)).fetchall()
            
            logger.info("Sample team ID to name mapping:")
            for team_id, team_name in result:
                logger.info(f"  {team_id}: {team_name}")
                
            return dict(result)
            
    except Exception as e:
        logger.error(f"❌ Failed to get team name mapping: {e}")
        return {}

def check_enhanced_predictor_issues():
    """Check the specific issues in enhanced_predictor.py."""
    logger.info("🔍 Checking enhanced_predictor.py initialization issues...")
    
    # Issue 1: MarkovFeatureGenerator initialization
    logger.info("\n1. Testing MarkovFeatureGenerator initialization:")
    markov_ok = test_markov_initialization()
    
    # Issue 2: SentimentAnalyzer method signature
    logger.info("\n2. Testing SentimentAnalyzer method signature:")
    sentiment_ok = test_sentiment_analyzer()
    
    # Issue 3: Team ID to name mapping
    logger.info("\n3. Getting team name mapping for sentiment analysis:")
    team_mapping = get_team_name_mapping()
    
    # Summary
    logger.info("\n📊 Summary of issues found:")
    if not markov_ok:
        logger.error("❌ MarkovFeatureGenerator: Initialization expects only lookback_window parameter")
    if not sentiment_ok:
        logger.error("❌ SentimentAnalyzer: get_sentiment_for_match expects team names, not IDs")
    if not team_mapping:
        logger.error("❌ Team mapping: Cannot retrieve team names for sentiment analysis")
        
    if markov_ok and sentiment_ok and team_mapping:
        logger.info("✅ All components can be initialized correctly")
        return True
    else:
        logger.error("❌ Issues found that need to be fixed")
        return False

def main():
    """Main function to check and identify initialization issues."""
    try:
        # Load configuration
        load_config()
        config = get_config()
        
        logger.info("🚀 Starting initialization issue analysis...")
        
        # Check issues
        success = check_enhanced_predictor_issues()
        
        if success:
            logger.info("\n🎉 Analysis complete - ready to fix enhanced_predictor.py")
        else:
            logger.error("\n💥 Analysis complete - issues identified that need fixing")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    main()