from formfinder.database import DatabaseManager
from formfinder.config import load_config
from datetime import datetime, date

# Load configuration first
load_config()

# Initialize database
db = DatabaseManager()
session = db.get_session()

try:
    from formfinder.database import Prediction, Fixture
    
    # Get today's date
    today = date.today()
    print(f"Checking predictions for today: {today}")
    
    # Query predictions for today
    predictions = session.query(Prediction).join(Fixture).filter(
        Fixture.match_date >= today,
        Fixture.match_date < datetime.combine(today.replace(day=today.day+1), datetime.min.time())
    ).all()
    
    print(f"Found {len(predictions)} predictions for today in database")
    
    for p in predictions:
        fixture = session.query(Fixture).filter(Fixture.id == p.fixture_id).first()
        print(f"Fixture {p.fixture_id}: {fixture.match_date} - confidence_score = {p.confidence_score}")
        print(f"  over_2_5_probability = {p.over_2_5_probability}")
        
    # Also check if there are any predictions with confidence_score >= 0.7
    high_conf_predictions = session.query(Prediction).filter(
        Prediction.confidence_score >= 0.7
    ).all()
    
    print(f"\nTotal predictions with confidence >= 0.7: {len(high_conf_predictions)}")
    
finally:
    session.close()