#!/usr/bin/env python3
"""
Test script for the FormFinder Notifier System

This script provides comprehensive testing for the new notifier system
including database connectivity, configuration validation, and notification testing.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from formfinder.notifier import (
    Notifier, DatabaseService, MessageFormatter, 
    ConfigManager, EmailNotifier, SMSNotifier, TelegramNotifier
)


class NotifierTester:
    """Comprehensive test suite for the notifier system."""
    
    def __init__(self):
        self.results = {}
    
    async def test_database_connection(self):
        """Test database connectivity."""
        print("🔍 Testing database connection...")
        try:
            # Load configuration first
            from formfinder.config import load_config
            load_config()
            
            db_service = DatabaseService()
            predictions = db_service.get_today_predictions(min_confidence=0.5)
            print(f"✅ Database connection successful - Found {len(predictions)} predictions")
            self.results['database'] = True
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.results['database'] = False
            return False
    
    async def test_configuration(self):
        """Test configuration loading and validation."""
        print("🔍 Testing configuration...")
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Validate required sections
            required_sections = ['notifications', 'email', 'sms', 'telegram']
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                print(f"❌ Missing configuration sections: {missing_sections}")
                self.results['configuration'] = False
                return False
            
            print("✅ Configuration loaded successfully")
            self.results['configuration'] = True
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.results['configuration'] = False
            return False
    
    async def test_message_formatting(self):
        """Test message formatting for different channels."""
        print("🔍 Testing message formatting...")
        try:
            formatter = MessageFormatter()
            
            # Create mock prediction data
            from formfinder.notifier import PredictionData
            mock_predictions = [
                PredictionData(
                    home_team="Arsenal",
                    away_team="Chelsea",
                    predicted_total_goals=2.5,
                    over_2_5_probability=0.65,
                    confidence_score=0.85,
                    match_date=datetime.now(),
                    league_name="Premier League",
                    predicted_score="2-1"
                ),
                PredictionData(
                    home_team="Real Madrid",
                    away_team="Barcelona",
                    predicted_total_goals=3.0,
                    over_2_5_probability=0.75,
                    confidence_score=0.78,
                    match_date=datetime.now(),
                    league_name="La Liga",
                    predicted_score="3-2"
                )
            ]
            
            message = formatter.format_predictions(mock_predictions)
            
            # Test all message formats
            formats = ['email_body', 'sms_body', 'telegram_body']
            for fmt in formats:
                content = getattr(message, fmt)
                if not content or len(content.strip()) == 0:
                    print(f"❌ Empty {fmt}")
                    self.results['formatting'] = False
                    return False
            
            print("✅ Message formatting successful")
            self.results['formatting'] = True
            return True
            
        except Exception as e:
            print(f"❌ Message formatting test failed: {e}")
            self.results['formatting'] = False
            return False
    
    async def test_notification_channels(self):
        """Test notification channel initialization."""
        print("🔍 Testing notification channels...")
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Test channel initialization
            channels = {
                'email': EmailNotifier(config),
                'sms': SMSNotifier(config),
                'telegram': TelegramNotifier(config)
            }
            
            for name, channel in channels.items():
                if hasattr(channel, 'config') and channel.config:
                    print(f"✅ {name.title()} channel initialized")
                else:
                    print(f"ℹ️  {name.title()} channel disabled or not configured")
            
            self.results['channels'] = True
            return True
            
        except Exception as e:
            print(f"❌ Channel test failed: {e}")
            self.results['channels'] = False
            return False
    
    async def test_full_notification_flow(self):
        """Test the complete notification flow."""
        print("🔍 Testing full notification flow...")
        try:
            notifier = Notifier()
            
            # Test with mock data
            from formfinder.notifier import PredictionData
            mock_predictions = [
                PredictionData(
                    home_team="Test Team 1",
                    away_team="Test Team 2",
                    predicted_total_goals=2.5,
                    over_2_5_probability=0.70,
                    confidence_score=0.90,
                    match_date=datetime.now(),
                    league_name="Test League",
                    predicted_score="2-0"
                )
            ]
            
            # Test mode (dry run)
            print("  📧 Testing email formatting...")
            message = notifier.formatter.format_predictions(mock_predictions)
            print(f"  📧 Email subject: {message.subject}")
            print(f"  📧 Email length: {len(message.email_body)} chars")
            
            print("  📱 Testing SMS formatting...")
            print(f"  📱 SMS length: {len(message.sms_body)} chars")
            
            print("  📱 Testing Telegram formatting...")
            print(f"  📱 Telegram length: {len(message.telegram_body)} chars")
            
            self.results['full_flow'] = True
            return True
            
        except Exception as e:
            print(f"❌ Full flow test failed: {e}")
            self.results['full_flow'] = False
            return False
    
    async def run_all_tests(self):
        """Run all tests and display summary."""
        print("🧪 FormFinder Notifier Test Suite")
        print("=" * 50)
        
        tests = [
            self.test_database_connection,
            self.test_configuration,
            self.test_message_formatting,
            self.test_notification_channels,
            self.test_full_notification_flow
        ]
        
        for test in tests:
            await test()
            print()
        
        # Display summary
        print("📊 Test Summary")
        print("=" * 30)
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for test_name, passed_test in self.results.items():
            status = "✅ PASSED" if passed_test else "❌ FAILED"
            print(f"{test_name.title()}: {status}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your notifier system is ready to use.")
        else:
            print("⚠️  Some tests failed. Please check the configuration and setup.")
        
        return passed == total


async def main():
    """Main test runner."""
    tester = NotifierTester()
    return await tester.run_all_tests()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)