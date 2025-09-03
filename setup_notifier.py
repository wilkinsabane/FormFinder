#!/usr/bin/env python3
"""
Setup script for FormFinder Notifier System

This script helps you configure the new notifier system with Telegram support,
including setting up Telegram bots and testing the configuration.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from formfinder.notifier import ConfigManager


class NotifierSetup:
    """Interactive setup assistant for the FormFinder notifier."""
    
    def __init__(self):
        self.config_path = Path("notifier_config.json")
        self.config_manager = ConfigManager()
        
    def display_welcome(self):
        """Display welcome message."""
        print("🚀 FormFinder Notifier Setup")
        print("=" * 50)
        print("This script will help you configure the new notifier system")
        print("with support for Email, SMS, and Telegram notifications.")
        print()
        
    def setup_telegram(self):
        """Guide user through Telegram bot setup."""
        print("📱 Telegram Bot Setup")
        print("-" * 30)
        print("1. Create a Telegram bot:")
        print("   • Open Telegram and search for @BotFather")
        print("   • Send: /newbot")
        print("   • Follow the prompts to create your bot")
        print("   • Save the bot token provided")
        print()
        
        print("2. Get your chat ID:")
        print("   • Add your new bot to a group or send it a message")
        print("   • Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
        print("   • Find your chat_id in the response")
        print()
        
        print("3. Enable notifications:")
        print("   • Set 'enabled': true in the telegram section")
        print("   • Add your bot_token and chat_ids")
        print()
        
        input("Press Enter when you've completed the Telegram setup...")
        
    def configure_notifications(self):
        """Configure notification preferences."""
        print("⚙️  Notification Configuration")
        print("-" * 30)
        
        config = self.config_manager.load_config()
        
        # Basic settings
        print("Current settings:")
        print(f"  Min confidence: {config['notifications']['min_confidence']}")
        print(f"  Max predictions: {config['notifications']['max_predictions']}")
        print()
        
        # Enable/disable channels
        print("Notification channels:")
        print(f"  Email enabled: {config['email']['enabled']}")
        print(f"  SMS enabled: {config['sms']['enabled']}")
        print(f"  Telegram enabled: {config['telegram']['enabled']}")
        print()
        
        # Ask for Telegram token if not set
        if not config['telegram']['enabled']:
            setup_telegram = input("Would you like to enable Telegram notifications? (y/n): ").lower()
            if setup_telegram == 'y':
                bot_token = input("Enter your Telegram bot token: ")
                chat_ids = input("Enter your chat IDs (comma-separated): ")
                
                config['telegram']['enabled'] = True
                config['telegram']['bot_token'] = bot_token
                config['telegram']['chat_ids'] = [cid.strip() for cid in chat_ids.split(',')]
                
                self.config_manager.save_config(config)
                print("✅ Telegram configuration updated!")
        
    def create_example_predictions(self):
        """Create example prediction data for testing."""
        print("📊 Creating example data...")
        
        # Create a simple example file
        example_data = {
            "example_date": datetime.now().isoformat(),
            "predictions": [
                {
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "home_win_probability": 0.65,
                    "away_win_probability": 0.20,
                    "draw_probability": 0.15,
                    "confidence_score": 0.85,
                    "league": "Premier League",
                    "predicted_score": "2-1",
                    "match_date": datetime.now().strftime("%Y-%m-%d")
                },
                {
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "home_win_probability": 0.55,
                    "away_win_probability": 0.30,
                    "draw_probability": 0.15,
                    "confidence_score": 0.78,
                    "league": "La Liga",
                    "predicted_score": "3-2",
                    "match_date": datetime.now().strftime("%Y-%m-%d")
                }
            ]
        }
        
        with open("example_predictions.json", "w") as f:
            json.dump(example_data, f, indent=2)
            
        print("✅ Example data created: example_predictions.json")
        
    def create_systemd_service(self):
        """Create a systemd service file for automatic notifications."""
        print("🖥️  Creating systemd service...")
        
        service_content = f"""[Unit]
Description=FormFinder Notifier Service
After=network.target

[Service]
Type=oneshot
User={os.getenv('USER', 'formfinder')}
WorkingDirectory={Path.cwd()}
Environment=PYTHONPATH={Path.cwd()}
ExecStart={sys.executable} -m formfinder.notifier --run-once
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"""
        
        timer_content = """[Unit]
Description=Run FormFinder Notifier daily
Requires=formfinder-notifier.service

[Timer]
OnCalendar=*-*-* 08:00:00
Persistent=true

[Install]
WantedBy=timers.target"""
        
        with open("formfinder-notifier.service", "w") as f:
            f.write(service_content)
            
        with open("formfinder-notifier.timer", "w") as f:
            f.write(timer_content)
            
        print("✅ Systemd files created:")
        print("   • formfinder-notifier.service")
        print("   • formfinder-notifier.timer")
        print()
        print("To enable automatic notifications:")
        print("  sudo cp formfinder-notifier.* /etc/systemd/system/")
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable formfinder-notifier.timer")
        print("  sudo systemctl start formfinder-notifier.timer")
        
    def create_logs_directory(self):
        """Create necessary directories."""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        print("✅ Logs directory created: logs/")
        
    def display_next_steps(self):
        """Display next steps for the user."""
        print("\n🎯 Next Steps")
        print("-" * 30)
        print("1. Test the notifier:")
        print("   python test_notifier.py")
        print()
        
        print("2. Run a manual notification:")
        print("   python -m formfinder.notifier --test-mode")
        print()
        
        print("3. Configure your settings:")
        print("   Edit notifier_config.json with your preferences")
        print()
        
        print("4. Set up Telegram (if desired):")
        print("   • Follow the Telegram setup guide above")
        print("   • Enable telegram in notifier_config.json")
        print()
        
        print("5. Schedule automatic notifications:")
        print("   • Use systemd timers (Linux)")
        print("   • Use Windows Task Scheduler (Windows)")
        print("   • Use cron (Unix-like systems)")
        
    def run_setup(self):
        """Run the complete setup process."""
        self.display_welcome()
        
        # Create necessary directories
        self.create_logs_directory()
        
        # Setup configuration
        self.configure_notifications()
        
        # Telegram setup
        self.setup_telegram()
        
        # Create example data
        self.create_example_predictions()
        
        # Create systemd files
        self.create_systemd_service()
        
        # Display next steps
        self.display_next_steps()


async def main():
    """Main setup runner."""
    setup = NotifierSetup()
    setup.run_setup()


if __name__ == "__main__":
    asyncio.run(main())