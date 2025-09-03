from scripts.database_feature_engine import DatabaseFeatureEngine
from formfinder.database import get_db_session

# Get the feature columns that the training system expects
db = get_db_session()
engine = DatabaseFeatureEngine(db)

print("Expected feature columns from DatabaseFeatureEngine:")
print(f"Total features: {len(engine.feature_columns)}")
for i, (key, value) in enumerate(engine.feature_columns.items(), 1):
    print(f"{i:2d}. {key} -> {value}")

db.close()