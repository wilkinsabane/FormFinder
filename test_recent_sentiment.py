#!/usr/bin/env python3
"""Test sentiment analysis with recent fixtures."""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and initialize config first
from formfinder.config import load_config, get_config
load_config()

# Now import enhanced predictor
from enhanced_predictor import EnhancedGoalPredictor
from formfinder.database import DatabaseManager, Fixture
from formfinder.sentiment import SentimentAnalyzer

def test_recent_sentiment():
    """Test sentiment analysis with recent fixtures."""
    print("Testing Sentiment Analysis with Recent Fixtures")
    print("=" * 50)
    
    try:
        # Test direct sentiment analyzer first
        print("\n1. Testing direct sentiment analyzer...")
        sentiment_analyzer = SentimentAnalyzer("ff008e7b4e9b4041ab44c50a729d7885", 24)
        
        # Test with recent date
        recent_date = datetime.now() - timedelta(days=1)
        result = sentiment_analyzer.get_sentiment_for_match(
            home_team="Arsenal",
            away_team="Chelsea", 
            match_date=recent_date
        )
        
        print(f"Direct sentiment test results:")
        print(f"  Arsenal sentiment: {result.home_sentiment:.3f} ({result.home_article_count} articles)")
        print(f"  Chelsea sentiment: {result.away_sentiment:.3f} ({result.away_article_count} articles)")
        
        # Test enhanced predictor with recent fixtures
        print("\n2. Testing enhanced predictor with recent fixtures...")
        predictor = EnhancedGoalPredictor()
        print(f"Sentiment analyzer available: {predictor.sentiment_analyzer is not None}")
        
        # Get recent fixtures
        db_manager = DatabaseManager()
        with db_manager.get_session() as session:
            # Look for fixtures from the last 30 days
            recent_cutoff = datetime.now() - timedelta(days=30)
            
            recent_fixtures = session.query(Fixture).filter(
                Fixture.status == 'finished',
                Fixture.match_date >= recent_cutoff
            ).order_by(Fixture.match_date.desc()).limit(5).all()
            
            if not recent_fixtures:
                print("❌ No recent finished fixtures found")
                # Try any fixture from the last year
                year_cutoff = datetime.now() - timedelta(days=365)
                recent_fixtures = session.query(Fixture).filter(
                    Fixture.status == 'finished',
                    Fixture.match_date >= year_cutoff
                ).order_by(Fixture.match_date.desc()).limit(5).all()
                
                if recent_fixtures:
                    print(f"Found {len(recent_fixtures)} fixtures from the last year")
                else:
                    print("❌ No fixtures found from the last year")
                    return
            else:
                print(f"Found {len(recent_fixtures)} recent fixtures")
            
            for i, fixture in enumerate(recent_fixtures):
                print(f"\n📋 Testing fixture {i+1}: {fixture.id}")
                print(f"   {fixture.home_team.name} vs {fixture.away_team.name}")
                print(f"   Date: {fixture.match_date}")
                
                # Extract features
                features = predictor.extract_enhanced_features(fixture.id)
                
                if features:
                    home_sentiment = features.get('home_team_sentiment', 'NOT_FOUND')
                    away_sentiment = features.get('away_team_sentiment', 'NOT_FOUND')
                    
                    print(f"   Home sentiment: {home_sentiment}")
                    print(f"   Away sentiment: {away_sentiment}")
                    
                    if home_sentiment != 0.0 or away_sentiment != 0.0:
                        print("   ✅ Non-zero sentiment values found!")
                        break
                    else:
                        print("   ⚠️ Zero sentiment values")
                else:
                    print("   ❌ Failed to extract features")
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recent_sentiment()