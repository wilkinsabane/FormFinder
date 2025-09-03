"""
Modern FormFinder Notification System

A comprehensive notification system that fetches predictions from the database
and sends notifications via email, SMS (Twilio), and Telegram.

Features:
- Database-driven predictions
- Multi-channel notifications (Email, SMS, Telegram)
- Robust error handling and retry logic
- Configurable notification preferences
- Rich message formatting per channel
- Comprehensive logging
- Modular architecture for easy extension
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta, date, UTC
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import aiohttp
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session, aliased

try:
    from .database import get_db_session, Prediction, Fixture, Team, League
    from .config import get_config, load_config
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from formfinder.database import get_db_session, Prediction, Fixture, Team, League
    from formfinder.config import get_config, load_config

# Configure logging
LOG_DIR = Path('data/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
NOTIFIER_LOG_FILE = LOG_DIR / 'notifier.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(NOTIFIER_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = Path('notifier_config.json')
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


@dataclass
class NotificationMessage:
    """Represents a formatted message for different notification channels."""
    subject: str
    email_body: str
    sms_body: str
    telegram_body: str
    telegram_parse_mode: str = "HTML"


@dataclass
class PredictionData:
    """Structured prediction data for notifications."""
    home_team: str
    away_team: str
    predicted_total_goals: float
    over_2_5_probability: float
    confidence_score: float
    match_date: datetime
    league_name: str
    home_team_form: Optional[float] = None
    away_team_form: Optional[float] = None
    predicted_score: Optional[str] = None


class DatabaseService:
    """Service for database operations related to predictions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.DatabaseService')
    
    def get_latest_predictions(
        self, 
        days_back: int = 1,
        min_confidence: float = 0.7,
        limit: Optional[int] = None
    ) -> List[PredictionData]:
        """
        Fetch latest predictions from database.
        
        Args:
            days_back: Number of days to look back for predictions
            min_confidence: Minimum confidence score threshold
            limit: Maximum number of predictions to return
            
        Returns:
            List of PredictionData objects
        """
        try:
            with get_db_session() as session:
                cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
                
                home_team = aliased(Team)
                away_team = aliased(Team)
                
                query = session.query(Prediction).join(Fixture).join(League).join(home_team, Fixture.home_team_id == home_team.id).join(away_team, Fixture.away_team_id == away_team.id).filter(
                    and_(
                        Prediction.prediction_date >= cutoff_date,
                        Prediction.confidence_score >= min_confidence
                    )
                ).order_by(desc(Prediction.confidence_score))
                
                if limit:
                    query = query.limit(limit)
                
                predictions = query.all()
                
                return [
                    PredictionData(
                        home_team=p.fixture.home_team.name,
                        away_team=p.fixture.away_team.name,
                        predicted_total_goals=p.predicted_total_goals or 0.0,
                        over_2_5_probability=p.over_2_5_probability or 0.0,
                        confidence_score=p.confidence_score,
                        match_date=p.fixture.match_date,
                        league_name=p.fixture.league.name,
                        home_team_form=p.home_team_form_score,
                        away_team_form=p.away_team_form_score,
                        predicted_score=f"{p.predicted_home_score:.0f}-{p.predicted_away_score:.0f}" 
                                       if p.predicted_home_score and p.predicted_away_score else None
                    )
                    for p in predictions
                ]
                
        except Exception as e:
            self.logger.error(f"Error fetching predictions from database: {e}")
            raise
    
    def get_today_predictions(self, min_confidence: float = 0.7) -> List[PredictionData]:
        """Get predictions for today's matches."""
        try:
            with get_db_session() as session:
                # Use local timezone since fixture dates are stored as naive datetimes
                today = date.today()
                today_start = datetime.combine(today, datetime.min.time())
                today_end = datetime.combine(today + timedelta(days=1), datetime.min.time())
                
                home_team = aliased(Team)
                away_team = aliased(Team)
                
                predictions = session.query(Prediction).join(Fixture).join(League).join(home_team, Fixture.home_team_id == home_team.id).join(away_team, Fixture.away_team_id == away_team.id).filter(
                    and_(
                        Fixture.match_date >= today_start,
                        Fixture.match_date < today_end,
                        Prediction.confidence_score >= min_confidence
                    )
                ).order_by(desc(Prediction.confidence_score)).all()
                
                return [
                    PredictionData(
                        home_team=p.fixture.home_team.name,
                        away_team=p.fixture.away_team.name,
                        predicted_total_goals=p.predicted_total_goals or 0.0,
                        over_2_5_probability=p.over_2_5_probability or 0.0,
                        confidence_score=p.confidence_score,
                        match_date=p.fixture.match_date,
                        league_name=p.fixture.league.name,
                        home_team_form=p.home_team_form_score,
                        away_team_form=p.away_team_form_score,
                        predicted_score=f"{p.predicted_home_score:.0f}-{p.predicted_away_score:.0f}" 
                                       if p.predicted_home_score and p.predicted_away_score else None
                    )
                    for p in predictions
                ]
                
        except Exception as e:
            self.logger.error(f"Error fetching today's predictions: {e}")
            raise


class MessageFormatter:
    """Handles formatting of messages for different notification channels."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.MessageFormatter')
    
    def format_predictions(self, predictions: List[PredictionData]) -> NotificationMessage:
        """
        Format predictions for different notification channels.
        
        Args:
            predictions: List of prediction data
            
        Returns:
            NotificationMessage object with formatted content
        """
        if not predictions:
            return self._format_no_predictions()
        
        subject_date = datetime.now().strftime("%B %d, %Y")
        subject = f"🏆 FormFinder Predictions - {subject_date}"
        
        email_body = self._format_email_body(predictions)
        sms_body = self._format_sms_body(predictions)
        telegram_body = self._format_telegram_body(predictions)
        
        return NotificationMessage(
            subject=subject,
            email_body=email_body,
            sms_body=sms_body,
            telegram_body=telegram_body
        )
    
    def _format_no_predictions(self) -> NotificationMessage:
        """Format message when no predictions are available."""
        date_str = datetime.now().strftime("%B %d, %Y")
        return NotificationMessage(
            subject=f"FormFinder Update - {date_str}",
            email_body="No high-confidence predictions found for today.",
            sms_body="FormFinder: No predictions today.",
            telegram_body="📊 <b>FormFinder Update</b>\n\nNo high-confidence predictions found for today."
        )
    
    def _format_email_body(self, predictions: List[PredictionData]) -> str:
        """Format detailed email body."""
        lines = [
            f"FormFinder has identified {len(predictions)} high-confidence prediction(s) for {datetime.now().strftime('%B %d, %Y')}:",
            "",
            "=" * 60,
            ""
        ]
        
        for i, pred in enumerate(predictions, 1):
            lines.extend([
                f"🎯 PREDICTION #{i}",
                f"📅 {pred.match_date.strftime('%A, %B %d at %H:%M')}",
                f"🏆 {pred.league_name}",
                f"⚽ {pred.home_team} vs {pred.away_team}",
                f"📊 Confidence: {pred.confidence_score:.1%}",
                f"⚽ Predicted Total Goals: {pred.predicted_total_goals:.1f}",
                f"🎯 Over 2.5 Goals: {pred.over_2_5_probability:.1%}"
            ])
            
            if pred.predicted_score:
                lines.append(f"🔮 Predicted Score: {pred.predicted_score}")
            
            if pred.home_team_form and pred.away_team_form:
                lines.extend([
                    f"📈 Form Scores: {pred.home_team} {pred.home_team_form:.1f} | {pred.away_team} {pred.away_team_form:.1f}"
                ])
            
            lines.extend(["", "-" * 40, ""])
        
        lines.extend([
            "",
            "💡 These predictions are based on advanced form analysis and machine learning models.",
            "⚠️  Always gamble responsibly. These are predictions, not guarantees."
        ])
        
        return "\n".join(lines)
    
    def _format_sms_body(self, predictions: List[PredictionData]) -> str:
        """Format concise SMS body (160 chars max)."""
        if len(predictions) == 1:
            pred = predictions[0]
            score = f" {pred.predicted_score}" if pred.predicted_score else ""
            return f"FormFinder: {pred.home_team} vs {pred.away_team}{score} ({pred.confidence_score:.0%})"
        else:
            return f"FormFinder: {len(predictions)} predictions ready. Check email/telegram for details."
    
    def _format_telegram_body(self, predictions: List[PredictionData]) -> str:
        """Format rich Telegram message."""
        lines = [
            "📊 <b>FormFinder Predictions</b>",
            f"<i>{datetime.now().strftime('%A, %B %d, %Y')}</i>",
            "",
            f"🔍 <b>{len(predictions)} High-Confidence Predictions</b>",
            ""
        ]
        
        for i, pred in enumerate(predictions, 1):
            lines.extend([
                f"🎯 <b>Prediction #{i}</b>",
                f"📅 <b>{pred.match_date.strftime('%H:%M')}</b> - {pred.league_name}",
                f"⚽ <b>{pred.home_team}</b> vs <b>{pred.away_team}</b>",
                f"📊 Confidence: <b>{pred.confidence_score:.1%}</b>",
                f"⚽ Total Goals: <b>{pred.predicted_total_goals:.1f}</b>",
                f"🎯 Over 2.5: <b>{pred.over_2_5_probability:.0%}</b>"
            ])
            
            if pred.predicted_score:
                lines.append(f"🔮 Score: <b>{pred.predicted_score}</b>")
            
            lines.append("")
        
        lines.extend([
            "⚠️ <i>Remember to gamble responsibly. These are predictions, not guarantees.</i>",
            "",
            "📱 <a href='#'>View Full Analysis</a>"
        ])
        
        return "\n".join(lines)


class EmailNotifier:
    """Handles email notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('email', {})
        self.logger = logging.getLogger(__name__ + '.EmailNotifier')
    
    async def send(self, message: NotificationMessage) -> bool:
        """Send email notification with retry logic."""
        if not self.config.get('enabled', False):
            self.logger.info("Email notifications disabled")
            return False
        
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.info(f"Sending email (attempt {attempt + 1}/{MAX_RETRIES})")
                
                msg = MIMEMultipart('alternative')
                msg['From'] = self.config['username']
                msg['To'] = ', '.join(self.config['to_emails'])
                msg['Subject'] = self.config['subject']
                
                # Add HTML part for rich formatting
                html_part = MIMEText(message.email_body, 'html')
                msg.attach(html_part)
                
                # Send email
                with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                    server.starttls()
                    server.login(self.config['username'], self.config['password'])
                    server.send_message(msg)
                
                self.logger.info("Email sent successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Email send failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
        
        return False


class SMSNotifier:
    """Handles SMS notifications via Twilio."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('sms', {})
        self.logger = logging.getLogger(__name__ + '.SMSNotifier')
    
    async def send(self, message: NotificationMessage) -> bool:
        """Send SMS notification with retry logic."""
        if not self.config.get('enabled', False):
            self.logger.info("SMS notifications disabled")
            return False
        
        try:
            from twilio.rest import Client
        except ImportError:
            self.logger.error("Twilio library not installed. Install with: pip install twilio")
            return False
        
        twilio_config = self.config.get('twilio', {})
        client = Client(
            twilio_config['twilio_account_sid'],
            twilio_config['twilio_auth_token']
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.info(f"Sending SMS (attempt {attempt + 1}/{MAX_RETRIES})")
                
                message_obj = client.messages.create(
                    body=message.sms_body,
                    from_=twilio_config['from_number'],
                    to=self.config['to_numbers'][0]
                )
                
                self.logger.info(f"SMS sent successfully. SID: {message_obj.sid}")
                return True
                
            except Exception as e:
                self.logger.error(f"SMS send failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
        
        return False


class TelegramNotifier:
    """Handles Telegram notifications via Bot API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('telegram', {})
        self.logger = logging.getLogger(__name__ + '.TelegramNotifier')
    
    async def send(self, message: NotificationMessage) -> bool:
        """Send Telegram notification with retry logic."""
        if not self.config.get('enabled', False):
            self.logger.info("Telegram notifications disabled")
            return False
        
        bot_token = self.config['bot_token']
        chat_ids = self.config['chat_ids']
        
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.info(f"Sending Telegram message (attempt {attempt + 1}/{MAX_RETRIES})")
                
                success_count = 0
                for chat_id in chat_ids:
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    payload = {
                        'chat_id': chat_id,
                        'text': message.telegram_body,
                        'parse_mode': message.telegram_parse_mode,
                        'disable_web_page_preview': True
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=payload) as response:
                            if response.status == 200:
                                success_count += 1
                            else:
                                error_text = await response.text()
                                self.logger.error(f"Telegram API error for chat {chat_id}: {response.status} - {error_text}")
                
                if success_count > 0:
                    self.logger.info(f"Telegram message sent successfully to {success_count} chat(s)")
                    return True
                
            except Exception as e:
                self.logger.error(f"Telegram send failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
        
        return False


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__ + '.ConfigManager')
    
    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            if not self.config_file.exists():
                self.logger.warning(f"Config file {self.config_file} not found, creating default")
                self._create_default_config()
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self._validate_config(config)
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _create_default_config(self):
        """Create default configuration file."""
        default_config = {
            "notifications": {
                "min_confidence": 0.7,
                "max_predictions": 10,
                "days_back": 1
            },
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "your_email@gmail.com",
                "sender_password": "your_app_password",
                "receiver_email": "receiver@gmail.com",
                "subject_prefix": "[FormFinder]"
            },
            "sms": {
                "enabled": False,
                "service_provider": "twilio",
                "twilio_account_sid": "your_account_sid",
                "twilio_auth_token": "your_auth_token",
                "twilio_from_number": "+1234567890",
                "receiver_phone_number": "+0987654321"
            },
            "telegram": {
                "enabled": False,
                "bot_token": "your_bot_token",
                "chat_id": "your_chat_id"
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure."""
        required_sections = ['notifications', 'email', 'sms', 'telegram']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")


class Notifier:
    """Main notification orchestrator."""
    
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_manager = ConfigManager(config_file)
        self.db_service = DatabaseService()
        self.formatter = MessageFormatter()
        self.logger = logging.getLogger(__name__)
    
    async def send_notifications(self, predictions: Optional[List[PredictionData]] = None) -> Dict[str, bool]:
        """
        Send notifications for predictions.
        
        Args:
            predictions: Optional list of predictions. If None, fetches from database.
            
        Returns:
            Dictionary with notification channel as key and success status as value
        """
        try:
            config = self.config_manager.load_config()
            
            # Get predictions if not provided
            if predictions is None:
                predictions = self.db_service.get_latest_predictions(
                    days_back=config['notifications']['days_back'],
                    min_confidence=config['notifications']['min_confidence'],
                    limit=config['notifications']['max_predictions']
                )
            
            self.logger.info(f"Sending notifications for {len(predictions)} predictions")
            
            # Format message
            message = self.formatter.format_predictions(predictions)
            
            # Initialize notifiers
            notifiers = {
                'email': EmailNotifier(config),
                'sms': SMSNotifier(config),
                'telegram': TelegramNotifier(config)
            }
            
            # Send notifications concurrently
            tasks = []
            for channel, notifier in notifiers.items():
                task = asyncio.create_task(notifier.send(message))
                tasks.append((channel, task))
            
            # Wait for all notifications to complete
            results = {}
            for channel, task in tasks:
                try:
                    results[channel] = await task
                except Exception as e:
                    self.logger.error(f"Error in {channel} notification: {e}")
                    results[channel] = False
            
            self.logger.info(f"Notifications completed: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error sending notifications: {e}")
            return {'error': str(e)}
    
    async def send_daily_notifications(self) -> Dict[str, bool]:
        """Send daily notifications for today's predictions."""
        try:
            config = self.config_manager.load_config()
            predictions = self.db_service.get_today_predictions(
                min_confidence=config['notifications']['min_confidence']
            )
            
            return await self.send_notifications(predictions)
            
        except Exception as e:
            self.logger.error(f"Error in daily notifications: {e}")
            return {'error': str(e)}
    
    def run_sync(self):
        """Run notifications synchronously (for backward compatibility)."""
        return asyncio.run(self.send_daily_notifications())


def main():
    """Main entry point for the notifier."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FormFinder Notification System')
    parser.add_argument('--config', default=CONFIG_FILE, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Test mode (dry run)')
    
    args = parser.parse_args()
    
    # Ensure configuration is loaded before any database operations
    try:
        load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    notifier = Notifier(args.config)
    
    if args.test:
        print("🧪 Running in test mode...")
        predictions = notifier.db_service.get_today_predictions()
        message = notifier.formatter.format_predictions(predictions)
        print(f"\n📧 Email Body:\n{message.email_body}")
        print(f"\n📱 SMS Body:\n{message.sms_body}")
        print(f"\n📱 Telegram Body:\n{message.telegram_body}")
    else:
        print("🚀 Starting FormFinder notifications...")
        results = asyncio.run(notifier.send_daily_notifications())
        print(f"✅ Notifications completed: {results}")


if __name__ == "__main__":
    main()