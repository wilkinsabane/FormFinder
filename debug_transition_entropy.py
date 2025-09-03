#!/usr/bin/env python3
"""
Debug script to investigate transition entropy calculation issues.
This script will check why transition entropy values are identical for different teams.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from formfinder.config import load_config
from formfinder.database import get_db_session
from formfinder.markov_feature_generator import MarkovFeatureGenerator
from formfinder.markov_transition_calculator import MarkovTransitionCalculator
from sqlalchemy import text
from datetime import datetime, timedelta
import numpy as np

# Load configuration
load_config()

def debug_transition_entropy():
    """Debug transition entropy calculation for multiple teams."""
    
    with get_db_session() as session:
        print("🔍 Debugging Transition Entropy Calculation")
        print("=" * 50)
        
        # Get some recent fixtures with different teams
        fixtures_query = """
            SELECT DISTINCT f.home_team_id, f.away_team_id, f.match_date, f.league_id,
                   ht.name as home_team_name, at.name as away_team_name
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.home_score IS NOT NULL 
                AND f.away_score IS NOT NULL
                AND f.match_date >= CURRENT_DATE - INTERVAL '60 days'
            ORDER BY f.match_date DESC
            LIMIT 10
        """
        
        fixtures = session.execute(text(fixtures_query)).fetchall()
        
        if not fixtures:
            print("❌ No fixtures found for testing")
            return
            
        print(f"Found {len(fixtures)} fixtures to test\n")
        
        # Initialize components
        markov_generator = MarkovFeatureGenerator(db_session=session, lookback_window=10)
        transition_calculator = MarkovTransitionCalculator()
        
        entropy_values = []
        team_data = []
        
        for fixture in fixtures:
            home_team_id = fixture.home_team_id
            away_team_id = fixture.away_team_id
            match_date = fixture.match_date
            league_id = fixture.league_id
            home_team_name = fixture.home_team_name
            away_team_name = fixture.away_team_name
            
            print(f"\n📊 Testing: {home_team_name} vs {away_team_name}")
            print(f"   Date: {match_date}, League: {league_id}")
            
            # Test home team
            print(f"\n🏠 Home Team: {home_team_name} (ID: {home_team_id})")
            
            # Check if team has performance states
            states_query = """
                SELECT COUNT(*) as state_count, 
                       MIN(state_date) as earliest_state,
                       MAX(state_date) as latest_state
                FROM team_performance_states 
                WHERE team_id = :team_id AND league_id = :league_id
            """
            
            state_info = session.execute(text(states_query), {
                'team_id': home_team_id,
                'league_id': league_id
            }).fetchone()
            
            print(f"   States available: {state_info.state_count}")
            if state_info.state_count > 0:
                print(f"   Date range: {state_info.earliest_state} to {state_info.latest_state}")
            
            # Calculate transition entropy
            home_entropy = markov_generator.calculate_transition_entropy(
                team_id=home_team_id,
                league_id=league_id,
                context='home'
            )
            
            print(f"   Transition Entropy (home context): {home_entropy:.6f}")
            
            # Get transition matrix for detailed analysis
            transition_matrix = transition_calculator.get_transition_matrix(
                team_id=home_team_id,
                league_id=league_id,
                context='home'
            )
            
            if transition_matrix:
                print(f"   Transition matrix found: {len(transition_matrix)} states")
                
                # Calculate entropy for each state
                state_entropies = []
                for from_state in markov_generator.STATES:
                    if from_state in transition_matrix:
                        state_entropy = transition_calculator.get_transition_entropy(
                            transition_matrix, from_state
                        )
                        state_entropies.append(state_entropy)
                        print(f"     {from_state} -> entropy: {state_entropy:.6f}")
                
                if state_entropies:
                    avg_entropy = np.mean(state_entropies)
                    print(f"   Average state entropy: {avg_entropy:.6f}")
                    print(f"   Entropy variance: {np.var(state_entropies):.6f}")
            else:
                print(f"   No transition matrix found - using default")
                default_entropy = np.log2(len(markov_generator.STATES)) / 2
                print(f"   Default entropy: {default_entropy:.6f}")
            
            entropy_values.append(home_entropy)
            team_data.append({
                'team_id': home_team_id,
                'team_name': home_team_name,
                'entropy': home_entropy,
                'context': 'home',
                'states_count': state_info.state_count
            })
            
            # Test away team
            print(f"\n✈️ Away Team: {away_team_name} (ID: {away_team_id})")
            
            away_state_info = session.execute(text(states_query), {
                'team_id': away_team_id,
                'league_id': league_id
            }).fetchone()
            
            print(f"   States available: {away_state_info.state_count}")
            
            away_entropy = markov_generator.calculate_transition_entropy(
                team_id=away_team_id,
                league_id=league_id,
                context='away'
            )
            
            print(f"   Transition Entropy (away context): {away_entropy:.6f}")
            
            entropy_values.append(away_entropy)
            team_data.append({
                'team_id': away_team_id,
                'team_name': away_team_name,
                'entropy': away_entropy,
                'context': 'away',
                'states_count': away_state_info.state_count
            })
        
        # Analyze results
        print("\n" + "=" * 50)
        print("📈 ENTROPY ANALYSIS RESULTS")
        print("=" * 50)
        
        if entropy_values:
            unique_values = set(entropy_values)
            print(f"Total entropy values calculated: {len(entropy_values)}")
            print(f"Unique entropy values: {len(unique_values)}")
            print(f"Min entropy: {min(entropy_values):.6f}")
            print(f"Max entropy: {max(entropy_values):.6f}")
            print(f"Mean entropy: {np.mean(entropy_values):.6f}")
            print(f"Std deviation: {np.std(entropy_values):.6f}")
            
            if len(unique_values) == 1:
                print("\n⚠️ WARNING: All entropy values are identical!")
                print(f"   All values = {list(unique_values)[0]:.6f}")
                
                # Check if this is the default value
                default_entropy = np.log2(len(markov_generator.STATES)) / 2
                if abs(list(unique_values)[0] - default_entropy) < 1e-6:
                    print(f"   This matches the default entropy value: {default_entropy:.6f}")
                    print("   This suggests no transition matrices are being found")
            
            print("\n📊 Entropy by team:")
            for team in team_data:
                print(f"   {team['team_name']} ({team['context']}): {team['entropy']:.6f} (states: {team['states_count']})")
        
        # Check transition matrix data
        print("\n" + "=" * 50)
        print("🔍 TRANSITION MATRIX DATA CHECK")
        print("=" * 50)
        
        matrix_count_query = """
            SELECT COUNT(*) as total_matrices,
                   COUNT(DISTINCT team_id) as unique_teams,
                   COUNT(DISTINCT home_away_context) as unique_contexts
            FROM markov_transition_matrices
        """
        
        matrix_info = session.execute(text(matrix_count_query)).fetchone()
        print(f"Total transition matrix records: {matrix_info.total_matrices}")
        print(f"Unique teams with matrices: {matrix_info.unique_teams}")
        print(f"Unique contexts: {matrix_info.unique_contexts}")
        
        if matrix_info.total_matrices == 0:
            print("\n❌ No transition matrices found in database!")
            print("   This explains why all entropy values are identical (using defaults)")
            print("   Need to populate transition matrices first")
        
        # Sample some transition matrix data
        sample_query = text("""
            SELECT team_id, from_state, to_state, transition_probability, 
                   home_away_context, total_transitions
            FROM markov_transition_matrices
            ORDER BY team_id, from_state, to_state
            LIMIT 20
        """)
        
        sample_matrices = session.execute(sample_query).fetchall()
        
        if sample_matrices:
            print("\n📋 Sample transition matrix data:")
            for record in sample_matrices[:10]:  # Show first 10
                print(f"   Team {record.team_id} ({record.home_away_context}): {record.from_state} -> {record.to_state} = {record.transition_probability:.4f}")
        
if __name__ == "__main__":
    debug_transition_entropy()