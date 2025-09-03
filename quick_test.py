#!/usr/bin/env python3
"""
Quick test script for FormFinder Notifier

Run this script to quickly test the new notifier system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def quick_test():
    """Run a quick test of the notifier system."""
    print("🚀 FormFinder Notifier Quick Test")
    print("=" * 40)
    
    try:
        # Import the notifier
        from formfinder.notifier import Notifier, DatabaseService
        from formfinder.config import get_config
        
        # Test configuration loading
        print("⚙️  Loading configuration...")
        config = get_config()
        print(f"✅ Configuration loaded - Database: {config.database.type}")
        
        # Test database connection
        print("📊 Testing database connection...")
        db = DatabaseService()
        
        # Try to get predictions from the last 7 days
        predictions = db.get_latest_predictions(days_back=7, min_confidence=0.5)
        print(f"✅ Found {len(predictions)} predictions from last 7 days")
        
        if predictions:
            print("\n📋 Sample predictions:")
            for pred in predictions[:3]:
                print(f"  {pred.home_team} vs {pred.away_team}")
                print(f"  League: {pred.league_name}")
                print(f"  Date: {pred.match_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"  Confidence: {pred.confidence_score:.1%}")
                if pred.predicted_score:
                    print(f"  Predicted: {pred.predicted_score}")
                print()
        else:
            print("ℹ️  No predictions found - database may be empty or predictions below confidence threshold")
        
        # Test notifier initialization
        print("🔧 Testing notifier initialization...")
        notifier = Notifier()
        print("✅ Notifier initialized successfully")
        
        # Test message formatting with mock data if no predictions
        print("✍️  Testing message formatting...")
        if predictions:
            test_predictions = predictions[:2]
        else:
            # Create mock data for testing
            from formfinder.notifier import PredictionData
            from datetime import datetime, timedelta
            test_predictions = [
                PredictionData(
                    home_team="Arsenal",
                    away_team="Chelsea",
                    home_win_probability=0.65,
                    away_win_probability=0.20,
                    draw_probability=0.15,
                    confidence_score=0.85,
                    match_date=datetime.now() + timedelta(days=1),
                    league_name="Premier League",
                    predicted_score="2-1"
                ),
                PredictionData(
                    home_team="Real Madrid",
                    away_team="Barcelona",
                    home_win_probability=0.55,
                    away_win_probability=0.30,
                    draw_probability=0.15,
                    confidence_score=0.78,
                    match_date=datetime.now() + timedelta(days=1),
                    league_name="La Liga",
                    predicted_score="3-2"
                )
            ]
        
        message = notifier.formatter.format_predictions(test_predictions)
        print(f"✅ Email subject: {message.subject}")
        print(f"✅ Email length: {len(message.email_body)} chars")
        print(f"✅ SMS length: {len(message.sms_body)} chars")
        print(f"✅ Telegram length: {len(message.telegram_body)} chars")
        
        # Test notification channels
        print("📱 Testing notification channels...")
        channels = []
        if notifier.config.get('email', {}).get('enabled', False):
            channels.append("Email")
        if notifier.config.get('sms', {}).get('enabled', False):
            channels.append("SMS")
        if notifier.config.get('telegram', {}).get('enabled', False):
            channels.append("Telegram")
        
        if channels:
            print(f"✅ Active channels: {', '.join(channels)}")
        else:
            print("ℹ️  No notification channels enabled - check notifier_config.json")
        
        print("\n🎉 Quick test completed successfully!")
        print("\nNext steps:")
        print("- Run 'python test_notifier.py' for full testing")
        print("- Run 'python setup_notifier.py' for setup assistance")
        print("- Run 'python examples/use_notifier.py' for example usage")
        print("- Check logs in: data/logs/notifier.log")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Troubleshooting:")
        print("- Ensure your database is populated with predictions")
        print("- Check if data/formfinder.db exists")
        print("- Run data processing scripts first if needed")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(quick_test())