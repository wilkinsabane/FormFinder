# 🏆 FormFinder Modern Notification System

A comprehensive, modern, and robust notification system that fetches predictions directly from the database and sends notifications via email, SMS (Twilio), and Telegram.

## ✨ Features

- **Database-Driven**: Fetches predictions directly from the database instead of CSV files
- **Multi-Channel Support**: Email, SMS (via Twilio), and Telegram notifications
- **Modern Architecture**: Modular, object-oriented design with proper separation of concerns
- **Robust Error Handling**: Comprehensive error handling with retry logic
- **Rich Formatting**: Optimized message formatting for each notification channel
- **Async Support**: Concurrent notification sending using asyncio
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Easy Configuration**: JSON-based configuration with validation
- **Test Mode**: Built-in test mode for dry runs

## 🚀 Quick Start

### 1. Configuration

Update your `notifier_config.json` file with the following structure:

```json
{
  "notifications": {
    "min_confidence": 0.75,
    "max_predictions": 8,
    "days_back": 1
  },
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your_email@gmail.com",
    "sender_password": "your_app_password",
    "receiver_email": "receiver_email@gmail.com",
    "subject_prefix": "🏆 [FormFinder]"
  },
  "sms": {
    "enabled": false,
    "twilio_account_sid": "your_twilio_sid",
    "twilio_auth_token": "your_twilio_token",
    "twilio_from_number": "+1234567890",
    "receiver_phone_number": "+0987654321"
  },
  "telegram": {
    "enabled": false,
    "bot_token": "your_bot_token",
    "chat_id": "your_chat_id"
  }
}
```

### 2. Telegram Bot Setup (Optional)

1. **Create a Telegram Bot**:
   - Message @BotFather on Telegram
   - Use `/newbot` to create a new bot
   - Save the bot token

2. **Get Your Chat ID**:
   - Add your bot to a group/channel
   - Send a message to the group
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your chat ID in the response

### 3. Run the Notifier

#### Basic Usage
```python
from formfinder.notifier import Notifier

# Create notifier instance
notifier = Notifier()

# Send daily notifications
results = asyncio.run(notifier.send_daily_notifications())
print(f"Results: {results}")
```

#### Command Line Usage
```bash
# Run daily notifications
python -m formfinder.notifier

# Test mode (dry run)
python -m formfinder.notifier --test

# Custom config file
python -m formfinder.notifier --config custom_config.json
```

## 📊 Database Schema Integration

The notifier directly fetches predictions from your database using the following models:

- **Prediction**: Contains prediction data with confidence scores
- **Fixture**: Match information including teams and match dates
- **Team**: Team details and names
- **League**: League information

### Query Examples

```python
# Get today's high-confidence predictions
from formfinder.notifier import DatabaseService

db_service = DatabaseService()
predictions = db_service.get_today_predictions(min_confidence=0.8)

# Get predictions from last 3 days
predictions = db_service.get_latest_predictions(days_back=3, limit=20)
```

## 🔧 Advanced Usage

### Custom Notification Channels

You can easily extend the system to add new notification channels:

```python
from formfinder.notifier import NotificationMessage

class CustomNotifier:
    def __init__(self, config):
        self.config = config
    
    async def send(self, message: NotificationMessage) -> bool:
        # Your custom implementation
        return True
```

### Batch Processing

```python
# Process multiple days
from datetime import datetime, timedelta

async def process_weekly_predictions():
    notifier = Notifier()
    
    for day_offset in range(7):
        date = datetime.now() - timedelta(days=day_offset)
        predictions = notifier.db_service.get_latest_predictions(
            days_back=day_offset,
            min_confidence=0.75
        )
        
        if predictions:
            await notifier.send_notifications(predictions)
```

## 📱 Message Formatting

### Email Format
- Detailed match information
- Probability breakdowns
- Form analysis
- Predicted scores (when available)
- Professional formatting with emojis

### SMS Format
- Concise, under 160 characters
- Essential information only
- Optimized for mobile reading

### Telegram Format
- Rich HTML formatting
- Interactive elements
- Compact yet comprehensive
- Emojis for visual appeal

## 🛠️ Configuration Reference

### Notification Settings
| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `min_confidence` | float | 0.75 | Minimum confidence score for predictions |
| `max_predictions` | int | 8 | Maximum predictions to include in notifications |
| `days_back` | int | 1 | How many days back to fetch predictions |

### Email Settings
| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `enabled` | bool | Yes | Enable/disable email notifications |
| `smtp_server` | string | Yes | SMTP server address |
| `smtp_port` | int | Yes | SMTP server port |
| `sender_email` | string | Yes | Sender email address |
| `sender_password` | string | Yes | Sender email password/app password |
| `receiver_email` | string | Yes | Recipient email address |

### SMS Settings (Twilio)
| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `enabled` | bool | Yes | Enable/disable SMS notifications |
| `twilio_account_sid` | string | Yes | Twilio Account SID |
| `twilio_auth_token` | string | Yes | Twilio Auth Token |
| `twilio_from_number` | string | Yes | Twilio phone number |
| `receiver_phone_number` | string | Yes | Recipient phone number |

### Telegram Settings
| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `enabled` | bool | Yes | Enable/disable Telegram notifications |
| `bot_token` | string | Yes | Telegram bot token |
| `chat_id` | string | Yes | Telegram chat/channel ID |

## 🔍 Logging and Monitoring

The system provides comprehensive logging:

- **Log Location**: `data/logs/notifier.log`
- **Log Level**: Configurable (INFO, DEBUG, WARNING, ERROR)
- **File Rotation**: Automatic log file rotation
- **Structured Logging**: JSON-formatted logs for monitoring tools

### Log Analysis

```bash
# View recent notifications
tail -f data/logs/notifier.log

# Search for specific errors
grep "ERROR" data/logs/notifier.log

# Monitor notification success rates
grep "sent successfully" data/logs/notifier.log
```

## 🧪 Testing

### Test Mode
```bash
# Dry run to see what would be sent
python -m formfinder.notifier --test
```

### Unit Tests
```bash
# Run specific tests
pytest tests/test_notifier.py -v

# Test with coverage
pytest tests/test_notifier.py --cov=formfinder.notifier
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check database configuration
   - Ensure database is running
   - Verify connection credentials

2. **Email Delivery Issues**
   - Check SMTP settings
   - Verify app-specific passwords
   - Check spam folders

3. **Telegram Bot Issues**
   - Verify bot token
   - Check chat ID format
   - Ensure bot has access to the chat

4. **SMS Delivery Issues**
   - Verify Twilio credentials
   - Check phone number formats
   - Review Twilio account balance

### Debug Mode

Enable debug logging by setting:

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

## 🔒 Security Best Practices

- **Use environment variables** for sensitive data
- **Enable 2FA** on all accounts
- **Use app-specific passwords** for email
- **Rotate API keys** regularly
- **Monitor usage** and set up alerts

## 📈 Performance Optimization

- **Connection pooling** for database queries
- **Async processing** for concurrent notifications
- **Caching** for frequently accessed data
- **Rate limiting** to respect API limits

## 🔄 Migration Guide

### From Old Notifier

1. **Backup existing configuration**
2. **Update notifier_config.json** with new structure
3. **Test in dry-run mode** first
4. **Gradually enable new channels**
5. **Monitor logs** for any issues

### Data Migration

The new system automatically uses your existing database, so no data migration is required.

## 📞 Support

For issues or questions:
1. Check the logs in `data/logs/notifier.log`
2. Run in test mode: `python -m formfinder.notifier --test`
3. Verify configuration settings
4. Check database connectivity

## 📝 Changelog

### v2.0.0 (Current)
- Complete rewrite with modern architecture
- Database-driven predictions
- Telegram notifications
- Async processing
- Comprehensive error handling
- Rich message formatting
- Modular design
- Enhanced logging

### v1.0.0 (Legacy)
- CSV-based predictions
- Basic email/SMS support
- Synchronous processing