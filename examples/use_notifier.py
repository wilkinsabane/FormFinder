#!/usr/bin/env python3
"""
Example usage of the FormFinder Notifier System

This script demonstrates how to use the new modern notifier system
to fetch predictions from the database and send notifications.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from formfinder.notifier import Notifier, DatabaseService


async def main():
    """Main example function."""
    print("🚀 FormFinder Notifier Example")
    print("=" * 50)
    
    # Initialize the notifier
    notifier = Notifier()
    
    # Load configuration first
    from formfinder.config import load_config
    load_config()
    
    # Test database connection and fetch predictions
    print("📊 Fetching predictions from database...")
    db_service = DatabaseService()
    
    try:
        # Get today's predictions
        predictions = db_service.get_today_predictions(min_confidence=0.7)
        print(f"✅ Found {len(predictions)} predictions for today")
        
        # Display predictions
        if predictions:
            print("\n📋 Today's Predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred.home_team} vs {pred.away_team}")
                print(f"     League: {pred.league_name}")
                print(f"     Date: {pred.match_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"     Confidence: {pred.confidence_score:.1%}")
                print(f"     Probabilities: Home {pred.home_win_probability:.1%} | "
                      f"Draw {pred.draw_probability:.1%} | "
                      f"Away {pred.away_win_probability:.1%}")
                if pred.predicted_score:
                    print(f"     Predicted Score: {pred.predicted_score}")
                print()
        else:
            print("ℹ️  No high-confidence predictions found for today")
        
        # Send notifications
        print("📤 Sending notifications...")
        results = await notifier.send_notifications(predictions)
        
        print("\n📊 Notification Results:")
        for channel, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {channel.title()}: {status}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your database is running and configured correctly")
        return False
    
    print("\n🎉 Example completed successfully!")
    return True


if __name__ == "__main__":
    asyncio.run(main())