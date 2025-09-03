#!/usr/bin/env python3
"""Check goal_difference column in standings table."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text

def main():
    load_config()
    
    with get_db_session() as session:
        # Check goal_difference column values
        print("Checking goal_difference column in standings table...")
        
        # Get total count
        total_count = session.execute(text(
            "SELECT COUNT(*) FROM standings"
        )).fetchone()[0]
        print(f"Total standings records: {total_count}")
        
        # Check distinct goal_difference values
        distinct_values = session.execute(text(
            "SELECT DISTINCT goal_difference FROM standings ORDER BY goal_difference"
        )).fetchall()
        print(f"\nDistinct goal_difference values: {[row[0] for row in distinct_values]}")
        
        # Count records with goal_difference = 0
        zero_count = session.execute(text(
            "SELECT COUNT(*) FROM standings WHERE goal_difference = 0"
        )).fetchone()[0]
        print(f"Records with goal_difference = 0: {zero_count}")
        
        # Count records with goal_difference != 0
        non_zero_count = session.execute(text(
            "SELECT COUNT(*) FROM standings WHERE goal_difference != 0"
        )).fetchone()[0]
        print(f"Records with goal_difference != 0: {non_zero_count}")
        
        # Check some sample records with goals_for and goals_against
        print("\nSample standings with goals data:")
        sample_data = session.execute(text(
            "SELECT team_id, goals_for, goals_against, goal_difference FROM standings LIMIT 10"
        )).fetchall()
        
        for team_id, goals_for, goals_against, goal_diff in sample_data:
            calculated_diff = goals_for - goals_against if goals_for is not None and goals_against is not None else None
            print(f"Team {team_id}: GF={goals_for}, GA={goals_against}, GD={goal_diff}, Calculated={calculated_diff}")
        
        # Check if there are any records where calculated goal_difference doesn't match stored value
        print("\nChecking for mismatched goal_difference calculations...")
        mismatched = session.execute(text(
            """SELECT team_id, goals_for, goals_against, goal_difference, 
                      (goals_for - goals_against) as calculated_diff
               FROM standings 
               WHERE goals_for IS NOT NULL AND goals_against IS NOT NULL 
                 AND goal_difference != (goals_for - goals_against)
               LIMIT 10"""
        )).fetchall()
        
        if mismatched:
            print(f"Found {len(mismatched)} records with mismatched goal_difference:")
            for team_id, gf, ga, stored_gd, calc_gd in mismatched:
                print(f"  Team {team_id}: Stored GD={stored_gd}, Should be {calc_gd} (GF={gf}, GA={ga})")
        else:
            print("No mismatched goal_difference calculations found.")
        
        # Check for NULL values in goals columns
        null_goals_for = session.execute(text(
            "SELECT COUNT(*) FROM standings WHERE goals_for IS NULL"
        )).fetchone()[0]
        null_goals_against = session.execute(text(
            "SELECT COUNT(*) FROM standings WHERE goals_against IS NULL"
        )).fetchone()[0]
        
        print(f"\nRecords with NULL goals_for: {null_goals_for}")
        print(f"Records with NULL goals_against: {null_goals_against}")

if __name__ == '__main__':
    main()