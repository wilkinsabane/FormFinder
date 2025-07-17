import os
import os
import json
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import logging

# Configure logging for the notifier script (appends to a shared log or its own)
LOG_DIR = 'data/logs' # Consistent with other scripts
os.makedirs(LOG_DIR, exist_ok=True)
NOTIFIER_LOG_FILE = os.path.join(LOG_DIR, 'notifier.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler(NOTIFIER_LOG_FILE),
        logging.StreamHandler()
    ]
)

CONFIG_FILE = 'notifier_config.json'

def load_config():
    """Loads notifier configuration from JSON file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logging.info("Notifier configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"CRITICAL: Notifier configuration file '{CONFIG_FILE}' not found.")
        return None
    except json.JSONDecodeError:
        logging.error(f"CRITICAL: Error decoding notifier configuration file '{CONFIG_FILE}'.")
        return None

def get_latest_prediction_file(predictions_dir):
    """Finds the latest prediction CSV file for today."""
    today_str = datetime.now().strftime("%Y%m%d")
    expected_filename = f"predictions_{today_str}.csv"
    filepath = os.path.join(predictions_dir, expected_filename)
    if os.path.exists(filepath):
        logging.info(f"Found prediction file for today: {filepath}")
        return filepath
    else:
        logging.warning(f"Prediction file for today not found: {filepath}")
        # Fallback: try to find any predictions_*.csv if today's isn't there
        try:
            all_pred_files = [f for f in os.listdir(predictions_dir) if f.startswith("predictions_") and f.endswith(".csv")]
            if all_pred_files:
                latest_file = sorted(all_pred_files, reverse=True)[0]
                logging.warning(f"Falling back to latest available prediction file: {latest_file}")
                return os.path.join(predictions_dir, latest_file)
        except Exception as e:
            logging.error(f"Error searching for fallback prediction files: {e}")
        return None


def format_predictions_for_display(df, max_matches=None):
    """Formats the predictions DataFrame into a string list for messages."""
    if df.empty:
        return "No high-potential matches found."

    lines = []
    df_display = df.head(max_matches) if max_matches else df

    for _, row in df_display.iterrows():
        home_team = row.get('home_team_name', 'N/A')
        home_rate = f" (HR: {row.get('home_win_rate', 'N/A'):.2f})" if pd.notna(row.get('home_win_rate')) else ""
        away_team = row.get('away_team_name', 'N/A')
        away_rate = f" (AR: {row.get('away_win_rate', 'N/A'):.2f})" if pd.notna(row.get('away_win_rate')) else ""
        match_time = row.get('time', '')
        match_date = row.get('date', '')
        lines.append(f"- {match_date} {match_time}: {home_team}{home_rate} vs {away_team}{away_rate}")
    
    if max_matches and len(df) > max_matches:
        lines.append(f"...and {len(df) - max_matches} more matches.")
        
    return "\n".join(lines)

def send_email(config, subject, body, attachment_path=None):
    """Sends an email notification."""
    if not config['email']['enabled']:
        logging.info("Email notifications are disabled in config.")
        return

    logging.info(f"Attempting to send email to {config['email']['receiver_email']}")
    msg = MIMEMultipart()
    msg['From'] = config['email']['sender_email']
    msg['To'] = config['email']['receiver_email']
    msg['Subject'] = f"{config['email']['subject_prefix']} {subject}"

    msg.attach(MIMEText(body, 'plain'))

    if attachment_path and os.path.exists(attachment_path):
        try:
            with open(attachment_path, "rb") as attachment:
                part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(part)
            logging.info(f"Attached file: {attachment_path}")
        except Exception as e:
            logging.error(f"Failed to attach file {attachment_path}: {e}")

    try:
        with smtplib.SMTP(config['email']['smtp_server'], config['email']['smtp_port']) as server:
            server.starttls()
            server.login(config['email']['sender_email'], config['email']['sender_password'])
            server.sendmail(config['email']['sender_email'], config['email']['receiver_email'], msg.as_string())
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def send_sms(config, body):
    """Sends an SMS notification (example using Twilio)."""
    if not config['sms']['enabled']:
        logging.info("SMS notifications are disabled in config.")
        return

    logging.info(f"Attempting to send SMS to {config['sms']['receiver_phone_number']}")
    if config['sms']['service_provider'] == 'twilio':
        try:
            from twilio.rest import Client # Make sure 'twilio' is installed: pip3 install twilio
            client = Client(config['sms']['twilio_account_sid'], config['sms']['twilio_auth_token'])
            message = client.messages.create(
                body=body,
                from_=config['sms']['twilio_from_number'],
                to=config['sms']['receiver_phone_number']
            )
            logging.info(f"SMS sent successfully via Twilio. SID: {message.sid}")
        except ImportError:
            logging.error("Twilio library not installed. Please install with 'pip3 install twilio'.")
        except Exception as e:
            logging.error(f"Failed to send SMS via Twilio: {e}")
    else:
        logging.warning(f"SMS service provider '{config['sms']['service_provider']}' not supported by this script. Implement sending logic.")


class Notifier:
    """A class to handle notifications for FormFinder predictions."""
    
    def __init__(self):
        pass
    
    def send_notifications(self, predictions):
        """Send notifications with predictions data.
        
        Args:
            predictions: List of prediction dictionaries or DataFrame
        """
        try:
            logging.info(f"Sending notifications for {len(predictions) if predictions else 0} predictions")
            
            if not predictions:
                logging.info("No predictions to send notifications for")
                return
            
            # Convert predictions to DataFrame if it's a list
            if isinstance(predictions, list):
                predictions_df = pd.DataFrame(predictions)
            else:
                predictions_df = predictions
            
            # Load configuration
            config = load_config()
            if not config:
                logging.error("Cannot send notifications without valid configuration")
                return
            
            # Create email content
            subject_date = datetime.now().strftime("%B %d, %Y")
            email_subject = f"Soccer Form Predictions for {subject_date}"
            
            if predictions_df.empty:
                email_body_content = "No high-potential matches were flagged for today."
                sms_body_content = f"FormFinder {subject_date[:6]}: No flagged matches."
            else:
                email_body_content = f"FormFinder has identified the following {len(predictions_df)} high-potential prediction(s) for {subject_date}:\n\n"
                email_body_content += format_predictions_for_display(predictions_df, config.get('max_matches_in_email_body'))
                
                sms_body_content = f"FormFinder {subject_date[:6]}: {len(predictions_df)} prediction(s). "
                sms_body_content += format_predictions_for_display(predictions_df.head(config.get('max_matches_in_sms', 1))).replace('\n', ' ')
                if len(sms_body_content) > 160:
                    sms_body_content = sms_body_content[:157] + "..."
            
            # Send notifications
            send_email(config, email_subject, email_body_content)
            send_sms(config, sms_body_content)
            
            logging.info("Notifications sent successfully")
            
        except Exception as e:
            logging.error(f"Error sending notifications: {e}")
    
    def run(self):
        """Main method to run the notifier."""
        logging.info("Notifier script started.")
        config = load_config()
        if not config:
            logging.critical("Notifier cannot proceed without valid configuration.")
            return

        prediction_file_path = get_latest_prediction_file(config.get('predictions_dir', 'data/predictions'))

        if not prediction_file_path:
            logging.warning("No prediction file found. Sending a notification about missing data.")
            email_subject = "Prediction File Not Found"
            email_body = f"The FormFinder prediction file for {datetime.now().strftime('%Y-%m-%d')} was not found in {config.get('predictions_dir', 'data/predictions')}."
            sms_body = f"FormFinder: No prediction CSV for {datetime.now().strftime('%d/%m')}."
            
            send_email(config, email_subject, email_body)
            send_sms(config, sms_body)
            logging.info("Notifier script finished: No data file.")
            return

        try:
            predictions_df = pd.read_csv(prediction_file_path)
            logging.info(f"Successfully loaded predictions from {prediction_file_path}. Matches found: {len(predictions_df)}")
        except Exception as e:
            logging.error(f"Failed to read or parse prediction file {prediction_file_path}: {e}")
            email_subject = "Error Processing Prediction File"
            email_body = f"FormFinder encountered an error trying to read/parse {prediction_file_path}:\n\n{e}"
            sms_body = f"FormFinder: Error reading predictions CSV for {datetime.now().strftime('%d/%m')}."
            
            send_email(config, email_subject, email_body)
            send_sms(config, sms_body)
            logging.info("Notifier script finished: Error with data file.")
            return
            
        file_date_str = os.path.basename(prediction_file_path).replace("predictions_", "").replace(".csv", "")
        try:
            # Try to parse YYYYMMDD from filename for a nicer date in subject
            subject_date = datetime.strptime(file_date_str, "%Y%m%d").strftime("%B %d, %Y")
        except ValueError:
            subject_date = datetime.now().strftime("%B %d, %Y") # Fallback to current date

        email_subject = f"Soccer Form Predictions for {subject_date}"
        
        if predictions_df.empty:
            email_body_content = "No high-potential matches were flagged for today."
            sms_body_content = f"FormFinder {subject_date[:6]}: No flagged matches."
        else:
            email_body_content = f"FormFinder has identified the following {len(predictions_df)} high-potential match(es) for {subject_date}:\n\n"
            email_body_content += format_predictions_for_display(predictions_df, config.get('max_matches_in_email_body'))
            
            sms_body_content = f"FormFinder {subject_date[:6]}: {len(predictions_df)} match(es). "
            sms_body_content += format_predictions_for_display(predictions_df.head(config.get('max_matches_in_sms', 1))).replace('\n', ' ') # Make it single line for SMS
            if len(sms_body_content) > 160: # Truncate if too long for standard SMS
                sms_body_content = sms_body_content[:157] + "..."

        # Send Email
        send_email(config, email_subject, email_body_content, attachment_path=prediction_file_path)

        # Send SMS
        send_sms(config, sms_body_content)

        logging.info("Notifier script finished.")

def main():
    """Legacy main function for backward compatibility."""
    notifier = Notifier()
    notifier.run()

if __name__ == "__main__":
    main()