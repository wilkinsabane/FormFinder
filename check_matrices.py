#!/usr/bin/env python3
from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

load_config()

with get_db_session() as session:
    result = session.execute(text('SELECT COUNT(*) FROM markov_transition_matrices'))
    count = result.scalar()
    print(f'Transition matrices: {count}')
    
    # Also check markov_features
    result2 = session.execute(text('SELECT COUNT(*) FROM markov_features'))
    count2 = result2.scalar()
    print(f'Markov features: {count2}')