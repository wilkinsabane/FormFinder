from formfinder.database import DatabaseManager
from formfinder.config import load_config
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# Load configuration first
load_config()

# Initialize database
db = DatabaseManager()
session = db.get_session()

try:
    from formfinder.database import Prediction, Fixture
    
    # Get today's date in different formats
    today = date.today()
    print(f"Today (local): {today}")
    
    # UTC timezone filtering (like notifier)
    UTC = ZoneInfo('UTC')
    today_start_utc = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    today_end_utc = today_start_utc + timedelta(days=1)
    print(f"Today start (UTC): {today_start_utc}")
    print(f"Today end (UTC): {today_end_utc}")
    
    # Local timezone filtering
    today_start_local = datetime.combine(today, datetime.min.time())
    today_end_local = datetime.combine(today + timedelta(days=1), datetime.min.time())
    print(f"Today start (local): {today_start_local}")
    print(f"Today end (local): {today_end_local}")
    
    # Check the fixture for today
    fixture = session.query(Fixture).filter(Fixture.id == 4938).first()
    if fixture:
        print(f"\nFixture 4938 match_date: {fixture.match_date}")
        print(f"Type: {type(fixture.match_date)}")
        print(f"Has timezone: {fixture.match_date.tzinfo is not None if hasattr(fixture.match_date, 'tzinfo') else 'N/A'}")
        
        # Test different filtering approaches
        print(f"\nFiltering tests:")
        print(f"UTC filter (match_date >= {today_start_utc}): {fixture.match_date >= today_start_utc}")
        print(f"UTC filter (match_date < {today_end_utc}): {fixture.match_date < today_end_utc}")
        print(f"Local filter (match_date >= {today_start_local}): {fixture.match_date >= today_start_local}")
        print(f"Local filter (match_date < {today_end_local}): {fixture.match_date < today_end_local}")
    
    # Test the exact query from notifier
    predictions_utc = session.query(Prediction).join(Fixture).filter(
        Fixture.match_date >= today_start_utc,
        Fixture.match_date < today_end_utc,
        Prediction.confidence_score >= 0.7
    ).all()
    print(f"\nUTC query found {len(predictions_utc)} predictions")
    
    # Test with local timezone
    predictions_local = session.query(Prediction).join(Fixture).filter(
        Fixture.match_date >= today_start_local,
        Fixture.match_date < today_end_local,
        Prediction.confidence_score >= 0.7
    ).all()
    print(f"Local query found {len(predictions_local)} predictions")
    
finally:
    session.close()